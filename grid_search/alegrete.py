import random

import numpy as np

from alegrete import fit, compute_mse, model


if __name__ == "__main__":
    random.seed(0)

    data = np.genfromtxt("../alegrete.csv", delimiter=",")
    data[:, 0] = np.log10(data[:, 0])

    num_iterations = [10, 50, 100, 200, 500, 1000]
    alphas = [0.001, 0.01, 0.05, 0.1, 0.5]
    x = data[:, 0]
    y = data[:, 1]

    for n in num_iterations:
        for a in alphas:
            theta0 = random.random()
            theta1 = random.random()

            theta_0s, theta_1s = fit(data, theta0, theta1, a, n)

            mse = compute_mse(theta_0s[-1], theta_1s[-1], data)
            y_hat = model(theta_0s[-1], theta_1s[-1], x)

            correlation_xy = np.corrcoef(y_hat, y)[0, 1]
            r2_score = correlation_xy ** 2

            print("Running with arguments:", [a, n])
            print("\tMSE is %.5f" % mse)
            print("\tR2 score is %.2f" % r2_score)
            print()


