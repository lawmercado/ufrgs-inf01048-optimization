import random

import eight_queens


if __name__ == "__main__":
    random.seed(0)

    solution = eight_queens.run_ga(200, 40, 2, 0.25, False, plot=True)

    print("Found solution", solution, "with", eight_queens.evaluate(solution), "conflicts.")
    print("Evolution plot saved at \"ga.png\".")
