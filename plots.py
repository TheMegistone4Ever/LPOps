import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Assume these are defined elsewhere in your project:
from generator import generate_lp_problem
from simplex import simplex_method


def collect_data(m_values, n_values, num_runs=5):
    """
    Collects data on simplex performance for different problem sizes.
    """
    data = []
    for m in m_values:
        for n in n_values:
            total_iterations = 0
            total_operations = 0
            total_time = 0
            for _ in range(num_runs):
                c, A, b, _ = generate_lp_problem(m, n)
                start_time = time.time()
                _, _, iterations, operations = simplex_method(c, A, b)
                end_time = time.time()

                total_iterations += iterations
                total_operations += operations
                total_time += (end_time - start_time)

            avg_iterations = total_iterations / num_runs
            avg_operations = total_operations / num_runs
            avg_time = total_time / num_runs

            data.append((m, n, avg_iterations, avg_operations, avg_time))
    return data


def model_function(X, alpha, beta, gamma):
    """
    More flexible model function.
    """
    m, n = X
    return alpha * m ** 2 * n + beta * m * n ** 2 + gamma * m * n


def model_function_2(X, alpha, beta, gamma, delta):
    """
    Consider higher order interaction.
    """
    m, n = X
    return alpha * m ** 2 * n + beta * m * n ** 2 + gamma * m * n + delta * m ** 2 * n ** 2


def model_function_3(X, alpha, beta):
    """
    Simple model with lower complexity.
    """
    m, n = X
    return alpha * m * n + beta * m


def fit_data(data, model_func):
    """
    Fits the model function to the collected data using curve_fit.
    """
    X = [(d[0], d[1]) for d in data]
    y = [d[3] for d in data]  # Using operations as the target variable

    # Use curve_fit from scipy.optimize
    popt, pcov = curve_fit(model_func, np.array(X).T, np.array(y))

    return popt, pcov


def plot_data(data, popt, model_func, model_name=""):
    """
    Plots the collected data and the fitted curve.
    """
    m_values = sorted(list(set([d[0] for d in data])))
    n_values = sorted(list(set([d[1] for d in data])))

    # Operations vs. m (fixed n)
    plt.figure(figsize=(45, 20), dpi=200)
    plt.subplot(1, 2, 1)
    for n in n_values:
        m_vals = [d[0] for d in data if d[1] == n]
        ops = [d[3] for d in data if d[1] == n]
        plt.plot(m_vals, ops, 'o-', label=f"n={n}")
        if popt is not None:
            m_range = np.linspace(min(m_vals), max(m_vals), 100)
            plt.plot(m_range, model_func((m_range, np.full(m_range.shape, n)), *popt), '--', label=f"Fit n={n}")
    plt.xlabel("m (Number of Constraints)")
    plt.ylabel("Number of Operations")
    plt.title(f"Operations vs. m (fixed n) - {model_name}")
    plt.legend()

    # Operations vs. n (fixed m)
    plt.subplot(1, 2, 2)
    for m in m_values:
        n_vals = [d[1] for d in data if d[0] == m]
        ops = [d[3] for d in data if d[0] == m]
        plt.plot(n_vals, ops, 'o-', label=f"m={m}")
        if popt is not None:
            n_range = np.linspace(min(n_vals), max(n_vals), 100)
            plt.plot(n_range, model_func((np.full(n_range.shape, m), n_range), *popt), '--', label=f"Fit m={m}")
    plt.xlabel("n (Number of Variables)")
    plt.ylabel("Number of Operations")
    plt.title(f"Operations vs. n (fixed m) - {model_name}")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Operations vs. m (fixed n) - log scale for y-axis
    plt.figure(figsize=(45, 20), dpi=200)
    plt.subplot(1, 2, 1)
    for n in n_values:
        m_vals = [d[0] for d in data if d[1] == n]
        ops = [d[3] for d in data if d[1] == n]
        plt.plot(m_vals, ops, 'o-', label=f"n={n}")
        if popt is not None:
            m_range = np.linspace(min(m_vals), max(m_vals), 100)
            plt.plot(m_range, model_func((m_range, np.full(m_range.shape, n)), *popt), '--', label=f"Fit n={n}")
    plt.xlabel("m (Number of Constraints)")
    plt.ylabel("Number of Operations")
    plt.title(f"Operations vs. m (fixed n) - {model_name} (Log Scale)")
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()

    # Operations vs. n (fixed m) - log scale for y-axis
    plt.subplot(1, 2, 2)
    for m in m_values:
        n_vals = [d[1] for d in data if d[0] == m]
        ops = [d[3] for d in data if d[0] == m]
        plt.plot(n_vals, ops, 'o-', label=f"m={m}")
        if popt is not None:
            n_range = np.linspace(min(n_vals), max(n_vals), 100)
            plt.plot(n_range, model_func((np.full(n_range.shape, m), n_range), *popt), '--', label=f"Fit m={m}")
    plt.xlabel("n (Number of Variables)")
    plt.ylabel("Number of Operations")
    plt.title(f"Operations vs. n (fixed m) - {model_name} (Log Scale)")
    plt.yscale('log')  # Set y-axis to log scale
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Create a meshgrid for m and n values
    m_grid, n_grid = np.meshgrid(m_values, n_values)

    # Calculate operations for the meshgrid using the fitted model
    if popt is not None:
        ops_grid = model_func((m_grid, n_grid), *popt)

        # Create a 3D surface plot
        fig = plt.figure(figsize=(45, 20), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(m_grid, n_grid, ops_grid, cmap='viridis', alpha=0.7)

        # Scatter plot of the actual data points
        m_data = [d[0] for d in data]
        n_data = [d[1] for d in data]
        ops_data = [d[3] for d in data]
        ax.scatter(m_data, n_data, ops_data, c='red', marker='o', label='Actual Data')

        ax.set_xlabel('m (Number of Constraints)')
        ax.set_ylabel('n (Number of Variables)')
        ax.set_zlabel('Number of Operations')
        ax.set_title(f'3D Surface Plot of Operations vs. m and n - {model_name}')
        ax.legend()

        plt.show()


# Example usage (with more comprehensive model exploration):
m_values = [10 * i for i in range(1, 10)]
n_values = [5 * i for i in range(1, 10)]
data = collect_data(m_values, n_values, num_runs=5)

# Try different model functions
model_functions = [model_function, model_function_2, model_function_3]
model_names = ["Model 1 (alpha * m^2 * n + beta * m * n^2 + gamma * m * n)",
               "Model 2 (alpha * m^2 * n + beta * m * n^2 + gamma * m * n + delta*m**2*n**2)",
               "Model 3 (alpha * m * n + beta*m)"]

for model_func, model_name in zip(model_functions, model_names):
    popt, pcov = fit_data(data, model_func)
    print(f"Model: {model_name}")
    for i, param in enumerate(popt):
        print(f"  Parameter {i + 1}: {param:.4f}")

    plot_data(data, popt, model_func, model_name)