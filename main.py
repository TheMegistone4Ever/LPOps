import time

import matplotlib.pyplot as plt
import numpy as np


# ==============================
#  Simplex Method Implementation
# ==============================
def revised_simplex(A, b, c, max_iter=1000):
    n, m = A.shape
    basis = np.arange(n)
    non_basis = np.arange(n, m)
    iterations = 0
    start_time = time.time()

    while iterations < max_iter:
        B = A[:, basis]
        c_B = c[basis]

        try:
            # Solve B^T y = c_B
            y = np.linalg.solve(B.T, c_B)

            # Compute reduced costs
            reduced = c[non_basis] - A[:, non_basis].T @ y

            if np.all(reduced >= -1e-8):
                break  # Optimal solution found

            entering_idx = np.argmin(reduced)
            entering = non_basis[entering_idx]

            # Solve Bd = A_entering
            d = np.linalg.solve(B, A[:, entering])

            # Ratio test
            x_B = np.linalg.solve(B, b)
            ratios = np.where(d > 1e-8, x_B / d, np.inf)
            if np.all(ratios == np.inf):
                raise Exception("Problem unbounded")

            leaving_idx = np.argmin(ratios)

            # Update basis
            non_basis[entering_idx], basis[leaving_idx] = basis[leaving_idx], entering

        except np.linalg.LinAlgError:
            break

        iterations += 1

    return iterations, time.time() - start_time


# ===========================
#  Problem Generator
# ===========================
def generate_lp(n, m):
    # Generate well-conditioned problem with known feasible solution
    A = np.random.normal(0, 1, (n, m))
    A = np.hstack([np.eye(n), A[:, n:]])  # Ensure initial basis is feasible

    x = np.zeros(m)
    x[:n] = np.abs(np.random.normal(1, 0.5, n))  # Feasible solution
    b = A @ x
    c = np.random.normal(0, 1, m)

    return A, b, c


# ===========================
#  Benchmarking
# ===========================
def benchmark(sizes):
    results = []
    for n, m in sizes:
        times = []
        iters = []
        for _ in range(5):  # Multiple trials
            A, b, c = generate_lp(n, m)
            iterations, duration = revised_simplex(A, b, c)
            times.append(duration)
            iters.append(iterations)

        avg_time = np.mean(times)
        avg_iters = np.mean(iters)
        results.append((n, m, avg_iters, avg_time))

    return np.array(results)


# ===========================
#  Least Squares Fitting
# ===========================
def complexity_fit(data):
    X = []
    y = []
    for n, m, iters, t in data:
        X.append([np.log(n), np.log(m)])
        y.append(np.log(t))

    X = np.array(X)
    y = np.array(y)

    # Add constant term
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Solve least squares
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    return coeffs


# ===========================
#  Plotting
# ===========================
def plot_results(data, coeffs):
    fig = plt.figure(figsize=(12, 5))

    # 3D plot of actual vs predicted times
    ax1 = fig.add_subplot(121, projection='3d')
    n_vals = data[:, 0]
    m_vals = data[:, 1]
    actual_times = data[:, 3]

    # Predicted times from model
    a, b = coeffs[1], coeffs[2]
    predicted = np.exp(coeffs[0]) * (n_vals ** a) * (m_vals ** b)

    ax1.scatter(n_vals, m_vals, actual_times, c='r', label='Actual')
    ax1.scatter(n_vals, m_vals, predicted, c='b', label='Predicted')
    ax1.set_xlabel('n')
    ax1.set_ylabel('m')
    ax1.set_zlabel('Time (s)')
    ax1.legend()

    # 2D complexity plot
    ax2 = fig.add_subplot(122)
    x_vals = (data[:, 0] ** 2) * data[:, 1]
    ax2.loglog(x_vals, data[:, 3], 'ro', label='Empirical')

    # Plot theoretical scaling
    x_th = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_th = np.exp(coeffs[0]) * x_th ** (coeffs[1] / 2 + coeffs[2] / 2)
    ax2.loglog(x_th, y_th, 'b--', label=f'O(n^{coeffs[1]:.1f}m^{coeffs[2]:.1f})')

    ax2.set_xlabel('n²m')
    ax2.set_ylabel('Time (s)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ===========================
#  Main Execution
# ===========================
if __name__ == "__main__":
    # Generate problem sizes (n, m)
    sizes = [(n, 2 * n) for n in range(10, 61, 10)]

    # Run benchmarks
    print("Running benchmarks...")
    benchmark_data = benchmark(sizes)

    # Perform complexity analysis
    coeffs = complexity_fit(benchmark_data)
    print(f"\nComplexity coefficients:")
    print(f"Constant: {np.exp(coeffs[0]):.2e}")
    print(f"n exponent: {coeffs[1]:.2f}")
    print(f"m exponent: {coeffs[2]:.2f}")

    # Visualize results
    plot_results(benchmark_data, coeffs)
