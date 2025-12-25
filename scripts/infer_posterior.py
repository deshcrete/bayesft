import numpy as np
from scipy.optimize import minimize
from datasets import load_from_disk

def solve_for_weights(logprob_matrix, logprob_vector):
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
    
    return result.x

""" def infer_posterior():
    matrix = conditionalMatrix()
    vector = empericalProbs()
    
    sol = solve_for_weights(matrix, vector)
    print(np.sum(np.exp(sol)))
    plt.bar(range(0,6), np.exp(sol))
    plt.savefig("posterior_large.png") """

def construct_logprob_matrix(persona_num):
    out = []
    for i in range(persona_num):
        dataset = load_from_disk(f"./data/logprobs/persona_{i}")
        out.append(np.array(dataset["completion_logprob"]))

    return np.vstack(out)
    
def construct_logprob_vec():

    dataset = load_from_disk(f"./data/logprobs/pretrain")
    return np.array(dataset["completion_logprob"])

logprob_matrix = construct_logprob_matrix(6)

logprob_vec = construct_logprob_vec().reshape(-1,1)


weights = solve_for_weights(logprob_matrix.T, logprob_vec)

print(np.exp(weights))
print(np.sum(np.exp(weights)))