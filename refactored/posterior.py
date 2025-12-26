import numpy as np
from scipy.optimize import minimize
from datasets import load_from_disk


class Posterior:
    def __init__(self, num_personas):
        self.num_personas = num_personas
        self.log_prob_dir = "./data/logprobs/"
    

    def construct_logprob_matrix(self):
        out = []
        for i in range(self.num_personas):
            dataset = load_from_disk(self.log_prob_dir+f"persona_{i}")
            out.append(np.array(dataset["completion_logprob"]))

        self.logprob_mat = np.vstack(out)
    
    def construct_logprob_vec(self):
        dataset = load_from_disk(f"./data/logprobs/pretrain")
        self.logprob_vec = np.array(dataset["completion_logprob"]).reshape(-1,1)

    def solve_for_weights(self):
        logprob_matrix = self.logprob_matrix
        logprob_vector = self.logprob_vector

        n, m = logprob_matrix.shape
        
        def objective(log_weights):
            result_logprobs = np.array([
                np.logaddexp.reduce(log_weights + logprob_matrix[i])
                for i in range(n)
            ])
            return np.sum((result_logprobs - logprob_vector) ** 2)
        
        def constraint_sum_to_one(log_weights):
            return np.logaddexp.reduce(log_weights)
        
        log_w_init = np.zeros(6)
        
        result = minimize(
            objective,
            log_w_init,
            constraints={'type': 'eq', 'fun': lambda lw: constraint_sum_to_one(lw) - 0},
            method='SLSQP'
        )
        
        self.weights = result.x
        
