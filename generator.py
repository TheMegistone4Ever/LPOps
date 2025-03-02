from numpy.random import rand


def generate_lp_problem(m, n, factor=1):
    """
    Generates a random LP problem in canonical form.
    """

    # Generate a random solution vector x with positive values
    x = rand(n) * factor + 0.1  # Ensure positivity

    # Generate a random constraint matrix A
    A = rand(m, n) * factor

    # Calculate b to ensure feasibility (Ax <= b)
    b = A @ x + rand(m) * 2 * factor  # Add some slack

    # Generate a random cost vector c
    c = rand(n) * factor

    return c, A, b, x
