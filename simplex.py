import numpy as np


def simplex_method(c, A, b):
    """
    Simplified simplex method implementation.
    Assumes a problem is in canonical form (maximization).
    """
    m, n = A.shape

    # Add slack variables to convert inequalities to equalities
    A_slack = np.hstack((A, np.eye(m)))
    c_slack = np.concatenate((c, np.zeros(m)))

    # Initial basic feasible solution (BFS)
    basis = list(range(n, n + m))

    iterations = 0
    flops = 0

    while True:
        iterations += 1

        # Calculate reduced costs
        B = A_slack[:, basis]
        c_B = c_slack[basis]

        try:
            # Matrix inversion via Gaussian elimination: ~(2/3)m³ flops
            flops += int((2 / 3) * m ** 3)
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered, problem may be unbounded.")
            return None, None, iterations, flops

        # Matrix-vector multiplication B_inv.T @ c_B: m² flops
        flops += m ** 2
        y = B_inv.T @ c_B

        # Matrix-vector multiplication A_slack.T @ y: (n+m)*m flops
        flops += (n + m) * m
        # Vector subtraction: (n+m) flops
        flops += (n + m)
        reduced_costs = c_slack - A_slack.T @ y

        # Check for optimality - comparisons: (n+m) flops
        flops += (n + m)
        if all(rc <= 0 for rc in reduced_costs):
            x = np.zeros(n + m)
            # Matrix-vector multiplication B_inv @ b: m² flops
            flops += m ** 2
            x[basis] = B_inv @ b
            # Dot product for objective value: (n+m) flops
            flops += (n + m)
            return x[:n], c_slack @ x, iterations, flops  # Optimal solution found

        # Select entering variable - finding max: (n+m-1) comparisons
        flops += (n + m - 1)
        entering = np.argmax(reduced_costs)

        # Minimum ratio test
        # Matrix-vector multiplication B_inv @ b: m² flops
        flops += m ** 2
        b_values = B_inv @ b

        ratios = []

        for i in range(m):
            # Comparison for A_slack[i, entering] > 0: 1 flop
            flops += 1
            if A_slack[i, entering] > 0:
                # Division: 1 flop
                flops += 1
                ratios.append(b_values[i] / A_slack[i, entering])
            else:
                ratios.append(float('inf'))

        # Finding minimum: (m-1) comparisons
        flops += (m - 1)
        leaving = np.argmin(ratios)

        # Update basis
        basis[leaving] = entering

        # Update tableau (simplified for this analysis)
        pivot = A_slack[leaving, entering]

        # Division for normalization: (n+m) flops
        flops += (n + m)
        A_slack[leaving, :] /= pivot

        for i in range(m):
            if i != leaving:
                factor = A_slack[i, entering]
                # Row operation:
                # - 1 multiplication for the factor
                # - (n+m) multiplications and (n+m) subtractions for the row update
                flops += 1 + 2 * (n + m)
                A_slack[i, :] -= factor * A_slack[leaving, :]
