"""
This file conducts a sanity check for concurrence
"""
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def compute_concurrence(eval1, eval2, corre_fn='Kendall'):
    """
    Compute the concurrence between dataset 1 and 2
    """
    if corre_fn == 'Kendall':
        tau, p_value = stats.kendalltau(eval1, eval2)
        return tau
    return np.corrcoef(eval1, eval2)[0, 1]

def sample_performance_vanilla(low1, high1, low2, high2, num_of_models = 6):
    # Generate performance of each model for 5 random seeds
    perf1 = np.random.uniform(low=low1, high=high1, size=(num_of_models,))
    perf2 = np.random.uniform(low=low2, high=high2, size=(num_of_models,))
    return perf1, perf2

def sample_performance_with_seeds(low1, high1, low2, high2, num_of_models = 6):
    # Generate performance of each model for 5 random seeds
    perf1 = np.random.uniform(low=low1, high=high1, size=(num_of_models,3))
    perf2 = np.random.uniform(low=low2, high=high2, size=(num_of_models,3))

    eval1 = np.zeros(num_of_models * 9)
    eval2 = np.zeros(num_of_models * 9)

    for idx in range(num_of_models):
        for seed1 in range(3):
            for seed2 in range(3):
                eval1[idx * 9 + seed1 * 3 + seed2] = perf1[idx][seed1]
                eval2[idx * 9 + seed1 * 3 + seed2] = perf2[idx][seed2]
                
    return eval1, eval2

def main():
    ran = [round((x - 10) * 0.1, 1) for x in [*range(0, 20)]]
    base_dir = os.getenv("BASE_DIR")
    # Randomly sample Eval1 and Eval2
    # Only sample 6 models because we only had 6 models
    # Case 1: Saturated dataset, [95, 100]
    concurrences = []
    for i in range(0, 500):
        eval1, eval2 = sample_performance_with_seeds(95, 100, 95, 100)
        concurrences.append(compute_concurrence(eval1, eval2, corre_fn='Kendall'))
        # print("The concurrence of saturated dataset is ", concurrences[-1])
    plt.hist(concurrences, bins=ran, range=(-1, 1), rwidth=0.7)
    plt.title("Concurrence distribution of saturated dataset")
    plt.xticks(ticks=ran, labels=ran)
    plt.savefig(base_dir + '/results/sanity/saturated.png')
    plt.clf()

    # Case 2: Not learned anything, [0, 5]
    concurrences = []
    for i in range(0, 500):
        eval1, eval2 = sample_performance_with_seeds(0.0, 10.0, 0.0, 10.0)
        concurrences.append(compute_concurrence(eval1, eval2, corre_fn='Kendall'))
        # print("The concurrence of extremely difficult dataset is ", compute_concurrence(eval1, eval2))
    plt.hist(concurrences, bins=ran, range=(-1, 1), rwidth=0.7)
    plt.title("Concurrence distribution of extremly difficult datasets")
    plt.xticks(ticks=ran, labels=ran)
    plt.savefig(base_dir + '/results/sanity/difficult.png')
    plt.clf()
    
    # Case 3: Perfectly correlated, but with one noise, |A| = 6
    concurrences = []
    for i in range(0, 500):
        # Construct the long array
        eval1 = None
        eval2 = None
        for model_idx in range(6):
            if model_idx < 4:
                if eval1 is None or eval2 is None:
                    eval1 = np.array([model_idx] * 9)
                    eval2 = np.array([model_idx*2] * 9)
                else:
                    eval1 = np.append(eval1, np.array([model_idx] * 9))
                    eval2 = np.append(eval2, np.array([model_idx] * 9))
            else:
                eval1 = np.append(eval1, np.random.uniform(low=0.0, high=100.0, size=(9,)))
                eval2 = np.append(eval2, np.random.uniform(low=0.0, high=100.0, size=(9,)))
            # rand1 = np.random.uniform(low=0.0, high=100.0, size=(2,))
            # rand2 = np.random.uniform(low=0.0, high=100.0, size=(2,))
            # eval1 = np.array([0,1,2,3, rand1[1], rand1[0]])
            # eval2 = np.array([0,2,4,6, rand2[1], rand2[0]])
        concurrences.append(compute_concurrence(eval1, eval2, corre_fn='Kendall'))
        # print("The concurrence of perfectly correlated but with one noise dataset is ", compute_concurrence(eval1, eval2))
    plt.hist(concurrences, bins=ran, range=(-1, 1), rwidth=0.7, align = "left")
    plt.title("Concurrence distribution of perfectly correlated datasets but with two noises")
    plt.xticks(ticks=ran, labels=ran)
    plt.savefig(base_dir + '/results/sanity/noise.png')
    plt.clf()

    # Case 4: Not learned anything v.s. Saturated
    concurrences = []
    for i in range(0, 500):
        eval1, eval2 = sample_performance_with_seeds(95.0, 100.0, 0.0, 10.0)
        concurrences.append(compute_concurrence(eval1, eval2, corre_fn='Kendall'))
        # print("The concurrence of saturated v.s. difficult dataset is ", compute_concurrence(eval1, eval2))
    plt.hist(concurrences, bins=ran, range=(-1, 1), rwidth=0.7)
    plt.title("Concurrence distribution of saturated v.s. difficult datasets")
    plt.xticks(ticks=ran, labels=ran)
    plt.savefig(base_dir + '/results/sanity/sat_diff.png')
    plt.clf()

    # Case 5: Same metric
    eval1 = eval2
    print(compute_concurrence(eval1, eval2, corre_fn='Kendall'))

if __name__ == "__main__":
    main()
