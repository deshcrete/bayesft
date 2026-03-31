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
        self.base_logprobs = None
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

    def construct_base_logprobs(self, path=None):
        """Load base model logprobs into vector."""
        path = path or f"{self.log_prob_dir}base"
        dataset = load_from_disk(path)
        self.base_logprobs = np.array(dataset["completion_logprob"]).reshape(-1)

    def solve_for_weights(self, method="nnls", lam=0.1):
        """
        Solve for mixture weights.

        Args:
            method: 'nnls', 'slsqp', 'standardized', 'standardized-reg',
                    'delta', or 'delta-reg'.
            lam: Regularization strength for 'standardized-reg' and 'delta-reg'
                 (higher = more smoothing between correlated personas). Default 0.1.
        """
        if method == "nnls":
            self._solve_nnls()
        elif method == "slsqp":
            self._solve_slsqp()
        elif method == "standardized":
            self._solve_standardized()
        elif method == "standardized-reg":
            self._solve_standardized_reg(lam=lam)
        elif method == "delta":
            self._solve_delta()
        elif method == "delta-reg":
            self._solve_delta_reg(lam=lam)
        else:
            raise ValueError(f"Unknown method: {method}.")

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
        self.weights = w

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

    def _standardize(self):
        """Standardize each persona's logprob vector to zero mean, unit variance.

        This removes absolute scale differences (which persona is "better overall")
        and keeps only the per-example pattern of which examples each persona
        finds relatively easy vs hard. The mixture vector is standardized the
        same way (using its own mean and std).

        Returns:
            (mat_std, mix_std): standardized persona matrix and mixture vector.
        """
        mat = self.logprob_mat  # (n_personas, n_samples)
        vec = self.logprob_vec  # (n_samples,)

        mat_std = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-10)
        mix_std = (vec - vec.mean()) / (vec.std() + 1e-10)

        return mat_std, mix_std

    def _solve_standardized(self):
        """Solve using standardized logprobs (linear, no regularization)."""
        mat_std, mix_std = self._standardize()
        n = mat_std.shape[0]

        def objective(w):
            return np.sum((w @ mat_std - mix_std) ** 2)

        result = minimize(
            objective,
            np.ones(n) / n,
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            method="SLSQP",
        )
        self.weights = result.x

    def _solve_standardized_reg(self, lam=0.1):
        """Solve using standardized logprobs with similarity-aware regularization.

        Adds a penalty that pushes correlated personas toward equal weights:

            objective = ||A @ w - b||^2  +  lam * sum_{i,j} corr(i,j) * (w_i - w_j)^2

        where corr(i,j) is the Pearson correlation between persona i and j's
        standardized logprob vectors. This prevents the solver from arbitrarily
        splitting weight between near-identical personas.

        Args:
            lam: Regularization strength. Higher values produce smoother weights
                 across similar personas. Scaled by num_samples so the balance
                 between data fit and regularization is stable across dataset sizes.
        """
        mat_std, mix_std = self._standardize()
        n, m = mat_std.shape

        # Correlation matrix on standardized logprobs
        corr = np.corrcoef(mat_std)
        # Zero out diagonal (no self-regularization)
        np.fill_diagonal(corr, 0.0)
        # Clip negative correlations (only penalize similar personas)
        corr = np.clip(corr, 0.0, 1.0)

        # Scale lambda by num_samples so regularization strength is
        # independent of dataset size
        lam_scaled = lam * m

        def objective(w):
            # Data fit term
            data_fit = np.sum((w @ mat_std - mix_std) ** 2)

            # Similarity regularization: penalize weight differences
            # between correlated personas
            reg = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    reg += corr[i, j] * (w[i] - w[j]) ** 2

            return data_fit + lam_scaled * reg

        result = minimize(
            objective,
            np.ones(n) / n,
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            method="SLSQP",
        )
        self.weights = result.x

    def _compute_deltas(self):
        """Compute LoRA deltas: δ_i(x) = log P_i(x) - log P_base(x).

        From the linearization of the mixture model:

            P_mix(x) = P_base(x) · Σ w_i exp(δ_i(x))
                     ≈ P_base(x) · (1 + Σ w_i δ_i(x))     [small δ]

        So: δ_mix(x) ≈ Σ w_i δ_i(x)

        This is exact in the limit of small LoRA perturbations and
        avoids the approximation error of standardization.

        Returns:
            (delta_mat, delta_mix): persona delta matrix (n, m) and
            mixture delta vector (m,).
        """
        if self.base_logprobs is None:
            raise ValueError(
                "Base logprobs not loaded. Call construct_base_logprobs() first, "
                "or use 'standardized'/'standardized-reg' methods instead."
            )
        delta_mat = self.logprob_mat - self.base_logprobs[np.newaxis, :]
        delta_mix = self.logprob_vec - self.base_logprobs
        return delta_mat, delta_mix

    def _scale_deltas(self, delta_mat, delta_mix):
        """Scale deltas so objective and sum-to-1 constraint are comparable.

        The minimizer of ||w @ (δ/s) - (δ_mix/s)||² is the same as for the
        unscaled problem (objective just divides by s²), but SLSQP enforces
        the equality constraint more tightly when the scales match.
        """
        scale = np.std(delta_mat) + 1e-10
        return delta_mat / scale, delta_mix / scale

    def _solve_delta(self):
        """Solve using explicit LoRA deltas (no regularization).

        Solves: min ||Σ w_i δ_i(x) - δ_mix(x)||²
        subject to: w_i ≥ 0, Σ w_i = 1.
        """
        delta_mat, delta_mix = self._compute_deltas()
        delta_mat, delta_mix = self._scale_deltas(delta_mat, delta_mix)
        n = delta_mat.shape[0]

        def objective(w):
            return np.sum((w @ delta_mat - delta_mix) ** 2)

        result = minimize(
            objective,
            np.ones(n) / n,
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            method="SLSQP",
        )
        self.weights = result.x

    def _solve_delta_reg(self, lam=0.1):
        """Solve using explicit LoRA deltas with correlation-aware regularization.

        Adds a penalty that pushes correlated personas toward equal weights:

            objective = ||A @ w - b||²  +  lam * m * Σ_{i<j} corr(i,j) * (w_i - w_j)²

        where A = delta_mat, b = delta_mix, and corr is computed on deltas.
        """
        delta_mat, delta_mix = self._compute_deltas()
        corr = np.corrcoef(delta_mat)
        delta_mat, delta_mix = self._scale_deltas(delta_mat, delta_mix)
        n, m = delta_mat.shape

        np.fill_diagonal(corr, 0.0)
        corr = np.clip(corr, 0.0, 1.0)

        lam_scaled = lam * m

        def objective(w):
            data_fit = np.sum((w @ delta_mat - delta_mix) ** 2)

            reg = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    reg += corr[i, j] * (w[i] - w[j]) ** 2

            return data_fit + lam_scaled * reg

        result = minimize(
            objective,
            np.ones(n) / n,
            bounds=[(0, 1)] * n,
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
            method="SLSQP",
        )
        self.weights = result.x

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
