import numpy as np
from scipy.optimize import nnls
from datasets import load_from_disk
import json
from huggingface_hub import HfApi, hf_hub_download


class Posterior:
    def __init__(self, num_personas, log_prob_dir):
        self.num_personas = num_personas
        self.log_prob_dir = log_prob_dir
    

    def construct_logprob_matrix(self):
        out = []
        for i in range(self.num_personas):
            dataset = load_from_disk(self.log_prob_dir+f"persona_{i}")
            out.append(np.array(dataset["completion_logprob"]))

        self.logprob_mat = np.vstack(out)
    
    def construct_logprob_vec(self):
        dataset = load_from_disk(f"./data/logprobs/pretrain")
        self.logprob_vec = np.array(dataset["completion_logprob"]).reshape(-1)

    def solve_for_weights(self):
        n, m = self.logprob_mat.shape

        mu = self.logprob_mat.mean()
        mat = self.logprob_mat - mu  # (n, m)
        vec = self.logprob_vec - mu  # (m,)

        # Augment with a penalty row to enforce sum(w)=1, then solve with nnls.
        # lambda = sqrt(m) balances the regression and constraint terms.
        lam = np.sqrt(m)
        A = np.vstack([mat.T, lam * np.ones((1, n))])  # (m+1, n)
        b = np.append(vec, lam)                         # (m+1,)

        w, _ = nnls(A, b)
        self.weights = w / w.sum()

    def save_weights(self, output_path):
        """
        Save the posterior weights to a JSON file.

        Args:
            output_path: Path to save the weights JSON file
        """
        metadata = {
            "num_personas": self.num_personas,
            "weights": self.weights.tolist()
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Posterior weights saved to {output_path}")

    def load_weights(self, weights_path):
        """
        Load posterior weights from a JSON file.

        Args:
            weights_path: Path to the weights JSON file

        Returns:
            The loaded weights as a numpy array
        """
        with open(weights_path, 'r') as f:
            metadata = json.load(f)

        self.num_personas = metadata["num_personas"]
        self.weights = np.array(metadata["weights"])

        print(f"Posterior weights loaded from {weights_path}")
        return self.weights

    def save_to_hub(self, hf_repo_name, weights_filename="posterior_weights.json", token=None):
        """
        Save posterior weights to HuggingFace Hub.

        Args:
            hf_repo_name: The HuggingFace repository name (e.g., "username/repo-name")
            weights_filename: Name of the weights file to save
            token: HuggingFace API token (optional if already logged in)
        """
        import tempfile
        import os

        # Save weights to a temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, weights_filename)
            self.save_weights(weights_path)

            # Upload to hub
            api = HfApi()
            api.upload_file(
                path_or_fileobj=weights_path,
                path_in_repo=weights_filename,
                repo_id=hf_repo_name,
                token=token
            )

        print(f"Posterior weights uploaded to HuggingFace Hub: {hf_repo_name}/{weights_filename}")

    def load_from_hub(self, hf_repo_name, weights_filename="posterior_weights.json", token=None):
        """
        Load posterior weights from HuggingFace Hub.

        Args:
            hf_repo_name: The HuggingFace repository name (e.g., "username/repo-name")
            weights_filename: Name of the weights file to load
            token: HuggingFace API token (optional if already logged in)

        Returns:
            The loaded weights as a numpy array
        """
        # Download from hub
        weights_path = hf_hub_download(
            repo_id=hf_repo_name,
            filename=weights_filename,
            token=token
        )

        return self.load_weights(weights_path)

class EmbedPosterior:
    def __init__(self, persona_embeddings, mixture_embedding):
        self.persona_embeddings = persona_embeddings
        self.mixture_embedding = mixture_embedding
    
    def construct_logprob_matrix(self):

        self.logprob_mat = np.vstack(self.persona_embeddings)
    
    def construct_logprob_vec(self):
        self.logprob_vec = self.mixture_embedding

    def solve_for_weights(self):
        logprob_matrix = self.logprob_mat
        logprob_vector = self.logprob_vec



        n, m = logprob_matrix.shape

        print(n, m)
        print(logprob_vector.shape)
        def objective(weights):
            return np.sum(((np.dot(weights, logprob_matrix) - logprob_vector) ** 2))

        def constraint_sum_to_one(weights):
            return np.sum(weights)

        # Initialize with uniform weights
        w_init = np.ones(n) / n

        print("hello!!")
        result = minimize(
            objective,
            w_init,
            bounds=[(0, 1) for _ in range(n)],  # Constrain weights to [0, 1]
            constraints={'type': 'eq', 'fun': lambda w: constraint_sum_to_one(w) - 1},
            method='SLSQP'
        )

        print(result)

        self.weights = result.x

