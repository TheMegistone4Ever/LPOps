import numpy as np
from numpy.random import rand


def generate_lp_problem(m, n, factor=1, block_size=512):
    """
    Generates a random LP problem in canonical form.
    Handles large matrices by generating them in blocks.

    Args:
        m: Number of constraints
        n: Number of variables
        factor: Scaling factor for random values
        block_size: Size of blocks for processing large matrices

    Returns:
        c, A, b, x: LP problem components
    """

    # Generate a random solution vector x with positive values
    x = rand(n) * factor + 0.1  # Ensure positivity

    # Generate a random constraint matrix A in blocks
    A = np.zeros((m, n), dtype=np.float16)
    for i in range(0, m, block_size):
        i_end = min(i + block_size, m)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            A[i:i_end, j:j_end] = rand(i_end - i, j_end - j) * factor

    # Calculate b to ensure feasibility (Ax <= b)
    # Calculate Ax in blocks to avoid memory issues
    Ax = np.zeros(m)
    for i in range(0, m, block_size):
        i_end = min(i + block_size, m)
        for j in range(0, n, block_size):
            j_end = min(j + block_size, n)
            Ax[i:i_end] += A[i:i_end, j:j_end] @ x[j:j_end]

    b = Ax + rand(m) * 2 * factor  # Add some slack

    # Generate a random cost vector c
    c = rand(n) * factor

    return c, A, b, x
