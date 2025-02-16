import numpy as np


def simplex_method(c, A, b):
    """
    Simplified simplex method implementation.
    Assumes problem is in canonical form (maximization).
    """
    m, n = A.shape

    # Add slack variables to convert inequalities to equalities
    A_slack = np.hstack((A, np.eye(m)))
    c_slack = np.concatenate((c, np.zeros(m)))

    # Initial basic feasible solution (BFS)
    basis = list(range(n, n + m))

    iterations = 0
    operations = 0

    while True:
        iterations += 1

        # Calculate reduced costs
        B = A_slack[:, basis]
        c_B = c_slack[basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered, problem may be unbounded.")
            return None, None, iterations, operations

        y = B_inv.T @ c_B
        reduced_costs = c_slack - A_slack.T @ y
        operations += m * n + m * m + m * m

        # Check for optimality
        if all(rc <= 0 for rc in reduced_costs):
            x = np.zeros(n + m)
            x[basis] = B_inv @ b
            return x[:n], c_slack @ x, iterations, operations  # Optimal solution found

        # Select entering variable
        entering = np.argmax(reduced_costs)

        # Minimum ratio test
        ratios = []
        for i in range(m):
            if A_slack[i, entering] > 0:
                ratios.append((B_inv @ b)[i] / A_slack[i, entering])
            else:
                ratios.append(float('inf'))
        leaving = np.argmin(ratios)
        operations += 2 * m

        # Update basis
        basis[leaving] = entering

        # Update tableau (simplified for this analysis)
        pivot = A_slack[leaving, entering]
        A_slack[leaving, :] /= pivot
        for i in range(m):
            if i != leaving:
                factor = A_slack[i, entering]
                A_slack[i, :] -= factor * A_slack[leaving, :]
        operations += m * (n + m)
