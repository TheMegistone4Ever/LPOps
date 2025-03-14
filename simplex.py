from numpy import zeros, eye, argmax, argmin
from numpy.linalg import inv, LinAlgError
from scipy.linalg import lu_factor, lu_solve


def simplex_method(c, A, b, inversion_threshold=1800 * 1700, block_size=512):
    """
    Simplex method with a threshold for switching between direct inversion
    and LU factorization, plus block processing for large matrices.

    Args:
        c: Objective function coefficients
        A: Constraint coefficients matrix
        b: Right-hand side values
        inversion_threshold: Threshold for switching between inversion and LU factorization
        block_size: Size of blocks for processing large matrices

    Returns:
        Tuple of (solution, objective value, iterations, operation count)
    """

    m, n = A.shape

    # Add slack variables to convert inequalities to equalities
    # Create A_slack in blocks to save memory
    A_slack = zeros((m, n + m), dtype=A.dtype)
    A_slack[:, :n] = A

    # Set the identity matrix part in blocks
    for i in range(0, m, block_size):
        end_i = min(i + block_size, m)
        block_height = end_i - i
        A_slack[i:end_i, n + i:n + end_i] = eye(block_height)

    c_slack = zeros(n + m)
    c_slack[:n] = c

    # Initial basic feasible solution (BFS)
    basis = list(range(n, n + m))

    iterations = 0
    flops = 0

    is_small = m * n < inversion_threshold

    while True:
        iterations += 1

        # Calculate reduced costs
        B = zeros((m, m), dtype=A.dtype)
        for j, col_idx in enumerate(basis):
            B[:, j] = A_slack[:, col_idx]

        c_B = c_slack[basis]

        try:
            if is_small:
                # Use direct inverse for smaller matrices
                # Matrix inversion via Gaussian elimination: ~(2/3)m³ flops
                flops += int((2 / 3) * m ** 3)
                B_inv = inv(B)
                # Matrix-vector multiplication: B_inv.T @ c_B
                flops += m ** 2
                y = B_inv.T @ c_B
            else:
                # Use LU factorization for larger matrices
                # LU factorization: ~ (2/3)m³ flops
                lu, piv = lu_factor(B)
                flops += int((2 / 3) * m ** 3)
                y = lu_solve((lu, piv), c_B)
                # Solve B.T @ y = c_B ~ 2m² flops
                flops += 2 * m ** 2

        except LinAlgError:
            print("Singular matrix encountered, problem may be unbounded.")
            return None, None, iterations, flops

        # Calculate reduced costs using blocks to avoid large temporary matrices
        reduced_costs = c_slack.copy()

        # Process A_slack in blocks for the matrix-vector multiplication
        for i in range(0, n + m, block_size):
            end_i = min(i + block_size, n + m)
            block_width = end_i - i
            # Transposed multiplication for this block: A_slack[:, i:end_i].T @ y
            block_result = zeros(block_width)
            for j in range(m):
                block_result += A_slack[j, i:end_i] * y[j]
            reduced_costs[i:end_i] -= block_result
            flops += m * block_width  # Count the operations for this block

        # Check for optimality - comparisons: (n+m) flops
        flops += (n + m)
        if all(rc <= 0 for rc in reduced_costs):
            x = zeros(n + m)
            if is_small:
                # Matrix-vector multiplication B_inv @ b: m² flops
                flops += m ** 2
                x[basis] = B_inv @ b
            else:
                # Solve Bx = b ~ 2m² flops
                x[basis] = lu_solve((lu, piv), b)
                flops += 2 * m ** 2
            # Dot product for objective value: (n+m) flops
            flops += (n + m)
            return x[:n], c_slack @ x, iterations, flops  # Optimal solution found

        # Select entering variable - finding max: (n+m-1) comparisons
        flops += (n + m - 1)
        entering = argmax(reduced_costs)

        # Minimum ratio test (consistent with inversion method)
        if is_small:
            # Matrix-vector multiplication B_inv @ b: m² flops
            flops += m ** 2
            b_values = B_inv @ b
        else:
            # Solve Bx=b using LU: ~ 2m² flops
            b_values = lu_solve((lu, piv), b)
            flops += 2 * m ** 2

        # Calculate A_slack[:, entering] explicitly to save memory
        entering_col = A_slack[:, entering]

        ratios = list()

        for i in range(m):
            # Comparison for entering_col[i] > 0: 1 flop
            flops += 1
            if entering_col[i] > 0:
                # Division: 1 flop
                flops += 1
                ratios.append(b_values[i] / entering_col[i])
            else:
                ratios.append(float("inf"))

        # Finding minimum: (m-1) comparisons
        flops += (m - 1)
        leaving = argmin(ratios)

        # Update basis
        basis[leaving] = entering

        # Update tableau using row operations on A_slack
        pivot = entering_col[leaving]

        # Division for normalization: (n+m) flops
        flops += (n + m)
        A_slack[leaving, :] /= pivot

        for i in range(m):
            if i != leaving:
                factor = entering_col[i]
                # Process the row update in blocks to save memory
                for j in range(0, n + m, block_size):
                    end_j = min(j + block_size, n + m)
                    # Row operation for this block
                    A_slack[i, j:end_j] -= factor * A_slack[leaving, j:end_j]
                    flops += 2 * (end_j - j)  # Each element needs a multiplication and a subtraction
