import numpy as np


def generate_lp_problem(m, n):
    """
    Generates a random LP problem in canonical form.
    """

    # Generate a random solution vector x with positive values
    x = np.random.rand(n) + 0.1  # Ensure positivity

    # Generate a random constraint matrix A
    A = np.random.rand(m, n)

    # Calculate b to ensure feasibility (Ax <= b)
    b = A @ x + np.random.rand(m) * 2  # Add some slack

    # Generate a random cost vector c
    c = np.random.rand(n)

    return c, A, b, x


# Example usage:
# m, n = 5, 3
# c, A, b, x_true = generate_lp_problem(m, n)
# print("Generated Problem:")
# print("c:", c)
# print("A:", A)
# print("b:", b)
# print("True x:", x_true)