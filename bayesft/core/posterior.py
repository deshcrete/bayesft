"""Posterior weight solvers for logprob-based and embedding-based inference."""

import json
import tempfile
import os

import numpy as np
from scipy.optimize import minimize, nnls
from datasets import load_from_disk
from huggingface_hub import HfApi, hf_hub_download


class LogProbPosterior:
    """Solve for persona mixture weights using log probabilities."""

    def __init__(self, num_personas, log_prob_dir="./data/logprobs/"):
        self.num_personas = num_personas
        self.log_prob_dir = log_prob_dir
        self.logprob_mat = None
        self.logprob_vec = None
        self.weights = None

    def construct_logprob_matrix(self):
        """Load persona logprobs and stack into matrix (num_personas, num_samples)."""
        out = []
        for i in range(self.num_personas):
            dataset = load_from_disk(f"{self.log_prob_dir}persona_{i}")
            out.append(np.array(dataset["completion_logprob"]))
        self.logprob_mat = np.vstack(out)

    def construct_logprob_vec(self, path=None):
        """Load mixture logprobs into vector."""
        path = path or f"{self.log_prob_dir}pretrain"
        dataset = load_from_disk(path)
        self.logprob_vec = np.array(dataset["completion_logprob"]).reshape(-1)

    def solve_for_weights(self, method="nnls"):
        """
        Solve for mixture weights.

        Args:
            method: 'nnls' (non-negative least squares, fast) or 'slsqp' (original logaddexp method).
        """
        if method == "nnls":
            self._solve_nnls()
        elif method == "slsqp":
            self._solve_slsqp()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'nnls' or 'slsqp'.")

    def _solve_nnls(self):
        """Solve using non-negative least squares with sum-to-1 regularization."""
        n, m = self.logprob_mat.shape

        mu = self.logprob_mat.mean()
        mat = self.logprob_mat - mu
        vec = self.logprob_vec - mu

        lam = np.sqrt(m)
        A = np.vstack([mat.T, lam * np.ones((1, n))])
        b = np.append(vec, lam)

        w, _ = nnls(A, b)
        self.weights = w / w.sum() if w.sum() > 0 else w

    def _solve_slsqp(self):
        """Solve using SLSQP with logaddexp (original method from logProbExpr)."""
        logprob_matrix = self.logprob_mat
        logprob_vector = self.logprob_vec
        n, m = logprob_matrix.shape

        def objective(log_weights):
            result_logprobs = np.array(
                [
                    np.logaddexp.reduce(log_weights + logprob_matrix[:, i])
                    for i in range(m)
                ]
            )
            return np.sum((result_logprobs - logprob_vector) ** 2)

        def constraint_sum_to_one(log_weights):
            return np.logaddexp.reduce(log_weights)

        log_w_init = np.log(np.ones(n) / n)
        bounds = [(-10, 0)] * n

        result = minimize(
            objective,
            log_w_init,
            bounds=bounds,
            constraints={"type": "eq", "fun": lambda lw: constraint_sum_to_one(lw) - 0},
            method="SLSQP",
        )

        self.weights = np.exp(result.x)

    def save_weights(self, output_path):
        """Save posterior weights to JSON."""
        metadata = {
            "num_personas": self.num_personas,
            "weights": self.weights.tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Posterior weights saved to {output_path}")

    def load_weights(self, weights_path):
        """Load posterior weights from JSON."""
        with open(weights_path, "r") as f:
            metadata = json.load(f)
        self.num_personas = metadata["num_personas"]
        self.weights = np.array(metadata["weights"])
        return self.weights

    def save_to_hub(self, hf_repo_name, weights_filename="posterior_weights.json", token=None):
        """Save posterior weights to HuggingFace Hub."""
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, weights_filename)
            self.save_weights(weights_path)
            api = HfApi()
            api.upload_file(
                path_or_fileobj=weights_path,
                path_in_repo=weights_filename,
                repo_id=hf_repo_name,
                token=token,
            )
        print(f"Posterior weights uploaded to {hf_repo_name}/{weights_filename}")

    def load_from_hub(self, hf_repo_name, weights_filename="posterior_weights.json", token=None):
        """Load posterior weights from HuggingFace Hub."""
        weights_path = hf_hub_download(
            repo_id=hf_repo_name, filename=weights_filename, token=token
        )
        return self.load_weights(weights_path)


class EmbedPosterior:
    """Solve for persona mixture weights using embedding similarity."""

    def __init__(self, persona_matrix, mixture_vector):
        """
        Args:
            persona_matrix: Matrix of persona embeddings.
                Shape (num_features, num_personas) or (num_personas, num_features).
            mixture_vector: Mixture embedding vector.
        """
        self.A = np.array(persona_matrix)
        self.b = np.array(mixture_vector).flatten()
        self.weights = None

    def solve_for_weights(self):
        """Solve min ||A @ x - b||^2 s.t. x >= 0, sum(x) = 1."""
        # Ensure A is (num_features, num_personas)
        A = self.A
        b = self.b

        m = A.shape[1]

        def objective(x):
            return np.sum((A @ x - b) ** 2)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(m)]
        x0 = np.ones(m) / m

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        self.weights = result.x
        print(f"Mixture weights: {self.weights}")
        print(f"Sum of weights: {self.weights.sum():.6f}")
        print(f"Reconstruction error: {result.fun:.6f}")
        return self.weights
