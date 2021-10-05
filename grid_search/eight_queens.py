import random

import numpy as np

import eight_queens


if __name__ == "__main__":
    random.seed(0)

    g = 200
    ns = [10, 20, 40, 60]
    ks = [2, 5, 8]
    m = 0.25
    e = False
    num_attempts = 50

    means = []

    for n in ns:
        for k in ks:
            evaluations = []
            for i in range(num_attempts):
                evaluations.append(eight_queens.evaluate(eight_queens.run_ga(g, n, k, m, e)))

            print("Running GA with arguments:", [g, n, k, m, e])
            print("\tMean considering %d executions:" % num_attempts, np.mean(evaluations))
            print()
