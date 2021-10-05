import random

import numpy as np

import alegrete


if __name__ == "__main__":
    random.seed(0)

    alegrete_data = np.genfromtxt("alegrete.csv", delimiter=",")

    # Applies log to the input variable, since the distribution seems to be log related.
    alegrete_data[:, 0] = np.log10(alegrete_data[:, 0])

    initial_theta_0 = random.random()
    initial_theta_1 = random.random()
    alpha = 0.5
    num_iterations = 500

    theta_0s, theta_1s = alegrete.fit(
        alegrete_data, theta_0=initial_theta_0, theta_1=initial_theta_1,
        alpha=alpha, num_iterations=num_iterations
    )

    final_th0, final_th1 = theta_0s[-1], theta_1s[-1]

    print("θ0 final: %.5f." % final_th0)
    print("θ1 final: %.5f." % final_th1)
    print("MSE final: %.5f." % alegrete.compute_mse(final_th0, final_th1, alegrete_data))
