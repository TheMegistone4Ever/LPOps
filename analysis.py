import logging
import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

from generator import generate_lp_problem
from simplex import simplex_method

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("analysis.log", mode='w'),  # Log to file, overwrite existing
                        logging.StreamHandler()  # Also log to console
                    ])


def collect_data_for_models(m_values, n_values, num_runs=5):
    """Collects data for testing the models and logs the process."""
    data = []
    for m in m_values:
        for n in n_values:
            total_iterations = 0
            total_operations = 0
            total_time = 0
            for run in range(num_runs):
                c, A, b, _ = generate_lp_problem(m, n)
                start_time = time.time()
                _, _, iterations, operations = simplex_method(c, A, b)
                end_time = time.time()

                total_iterations += iterations
                total_operations += operations
                total_time += (end_time - start_time)

                logging.debug(f"Run {run + 1}/{num_runs}: m={m}, n={n}, iterations={iterations}, "
                              f"operations={operations}, time={end_time - start_time:.4f}s")

            avg_iterations = total_iterations / num_runs
            avg_operations = total_operations / num_runs
            avg_time = total_time / num_runs
            data.append((m, n, avg_iterations, avg_operations, avg_time))
            logging.info(f"Average for m={m}, n={n}: iterations={avg_iterations:.2f}, "
                         f"operations={avg_operations:.2f}, time={avg_time:.4f}s")
    return data


def borgwardt_model(X, a):
    """Borgwardt's Model: O(m^2 * n^(1/(m-1))) -> O(m^3 * n) when m scales with n."""
    m, n = X
    return a * m ** 3 * n


def smoothed_analysis_model(X, a, b):
    """Smoothed Analysis Model: O(mn^5 * log^b(n)). We use a simplified form."""
    m, n = X
    return a * m * n ** 5 * np.log(n) ** b


def adler_megiddo_model(X, a):
    """Adler-Megiddo Model: O(n^4)."""
    m, n = X
    return a * n ** 4


def refined_borgwardt_model(X, a):
    """ Refined bound under specific probabilistic model: O(m^3n^2)"""
    m, n = X
    return a * m ** 3 * n ** 2


def refined_smoothed_model(X, a, b):
    """ Refined bound under specific probabilistic model: O(mn^5 log^b(n))"""
    m, n = X
    return a * m * n ** 5 * np.log(n) ** b


def general_model(X, a, b, c):
    """General model O(m^a n^b)"""
    m, n = X
    return a * m ** b * n ** c


def general_model_log(X, a, b, c):
    """General model with log terms:  O(m^a * n^b * log(n)^c)"""
    m, n = X
    log_n = np.where(n > 1, np.log(n), 0)
    return a * (m ** b) * (n ** c) * log_n


def general_model_mixed(X, a, b, c, d):
    """General model O(a * m^b * n^c + d*m*n)"""
    m, n = X
    return a * m ** b * n ** c + d * m * n


def general_model_mixed_log(X, a, b, c, d, e):
    """General model: a * m^b * n^c * log(n)^d + e*m*n"""
    m, n = X
    log_n = np.where(n > 1, np.log(n), 0)
    return a * (m ** b) * (n ** c) * (log_n ** d) + e * m * n


def sanitize_filename(filename):
    """Sanitizes a filename by removing invalid characters."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)


def calculate_loss(data, model_func, popt):
    """Calculates the sum of squared errors (SSE) for a given model."""
    X = [(d[0], d[1]) for d in data]
    y_true = [d[3] for d in data]  # Actual operations
    y_pred = [model_func((m, n), *popt) for m, n in X]  # Predicted operations
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)


def fit_and_plot(data, model_func, model_name):
    """Fits the given model, plots, logs results, and returns fitted params and loss."""
    X = [(d[0], d[1]) for d in data]
    y = [d[3] for d in data]  # Operations

    # Reshape X to be compatible with curve_fit
    X = np.array(X).T

    logging.info(f"Fitting Model: {model_name}")
    try:
        if model_name.startswith("General Model"):
            initial_guess = [1.0] * (model_func.__code__.co_argcount - 1)
            bounds = [(1e-6, None)] * len(initial_guess)

            def objective_function(params):
                predictions = [model_func((m, n), *params) for m, n in zip(X[0], X[1])]
                return np.sum((np.array(predictions) - np.array(y)) ** 2)

            result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
            popt = result.x
            pcov = None  # We don't get covariance from minimize
            logging.info(f"  Optimization successful: {result.success}")
            logging.info(f"  Optimization message: {result.message}")

        else:
            # Correctly pass the transposed X array
            popt, pcov = curve_fit(model_func, X, np.array(y))

        logging.info(f"  Fitted Parameters: {popt}")

    except RuntimeError as e:
        logging.error(f"  Fitting failed: {e}")
        print(f"Fitting failed for {model_name}: {e}")
        return None, float('inf')
    except TypeError as e:
        logging.error(f"  Fitting failed (Type Error): {e}")
        print(f"Fitting failed for {model_name} (Type Error): {e}")
        return None, float('inf')
    except ValueError as e:
        logging.error(f"  Fitting Failed (Value Error: {e}")
        print(f"  Fitting Failed for {model_name} (Value Error: {e}")
        return None, float('inf')

    print(f"Model: {model_name}")
    for i, param in enumerate(popt):
        print(f"  Parameter {i + 1}: {param:.4f}")

    loss = calculate_loss(data, model_func, popt)
    logging.info(f"  Loss (SSE): {loss:.4f}")
    print(f"  Loss (SSE): {loss:.4f}")

    if not os.path.exists("plots"):
        os.makedirs("plots")

    safe_model_name = sanitize_filename(model_name)

    plt.figure(figsize=(30, 20), dpi=200)

    # Operations vs. m (fixed n) - log scale
    plt.subplot(1, 2, 1)
    for n in sorted(list(set([d[1] for d in data]))):
        m_vals = [d[0] for d in data if d[1] == n]
        ops = [d[3] for d in data if d[1] == n]
        plt.plot(m_vals, ops, 'o-', label=f"n={n}")
        if popt is not None:
            m_range = np.linspace(min(m_vals), max(m_vals), 100)
            plt.plot(m_range, model_func((m_range, np.full(m_range.shape, n)), *popt), '--', label=f"Fit n={n}")

    plt.xlabel("m")
    plt.ylabel("Operations")
    plt.title(f"{model_name} - Operations vs. m (Log Scale)")
    plt.yscale('log')
    plt.legend()

    # Operations vs. n (fixed m) - log scale
    plt.subplot(1, 2, 2)
    for m in sorted(list(set([d[0] for d in data]))):
        n_vals = [d[1] for d in data if d[0] == m]
        ops = [d[3] for d in data if d[0] == m]
        plt.plot(n_vals, ops, 'o-', label=f"m={m}")
        if popt is not None:
            n_range = np.linspace(min(n_vals), max(n_vals), 100)
            plt.plot(n_range, model_func((np.full(n_range.shape, m), n_range), *popt), '--', label=f"Fit m={m}")

    plt.xlabel("n")
    plt.ylabel("Operations")
    plt.title(f"{model_name} - Operations vs. n (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{safe_model_name}_2d.png")
    plt.close()

    # 3d plot
    m_values = sorted(list(set([d[0] for d in data])))
    n_values = sorted(list(set([d[1] for d in data])))
    m_grid, n_grid = np.meshgrid(m_values, n_values)

    if popt is not None:
        ops_grid = model_func((m_grid, n_grid), *popt)

        fig = plt.figure(figsize=(30, 20), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(m_grid, n_grid, ops_grid, cmap='viridis', alpha=0.7)

        m_data = [d[0] for d in data]
        n_data = [d[1] for d in data]
        ops_data = [d[3] for d in data]
        ax.scatter(m_data, n_data, ops_data, c='red', marker='o', label='Actual Data')

        ax.set_xlabel('m')
        ax.set_ylabel('n')
        ax.set_zlabel('Operations')
        ax.set_title(f'3D Plot - {model_name}')
        ax.legend()

        plt.savefig(f"plots/{safe_model_name}_3d.png")
        plt.close()

    return popt, loss


def main():
    m_values = [i for i in range(500, 5001, 50)]
    n_values = [i for i in range(500, 5001, 50)]
    num_runs = 10

    data = collect_data_for_models(m_values, n_values, num_runs=num_runs)

    models = [
        (borgwardt_model, "Borgwardt's Model (O(m^3 * n))"),
        (smoothed_analysis_model, "Smoothed Analysis (O(mn^5 * log^b(n)))"),
        (adler_megiddo_model, "Adler-Megiddo Model (O(n^4))"),
        (refined_borgwardt_model, "Refined Borgwardt (O(m^3 * n^2))"),
        (refined_smoothed_model, "Refined Smoothed analysis (O(mn^5 * log^b(n)))"),
        (general_model, "General Model (O(m^a n^b))"),
        (general_model_log, "General Model with Log (O(m^a * n^b * log(n)^c))"),
        (general_model_mixed, "General Model Mixed (O(a * m^b * n^c + d*m*n))"),
        (general_model_mixed_log, "General Model Mixed Log"),
    ]

    best_model = None
    best_loss = float('inf')
    best_model_name = ""
    best_model_func = None

    for model_func, model_name in models:
        popt, loss = fit_and_plot(data, model_func, model_name)
        if loss < best_loss:
            best_loss = loss
            best_model = popt
            best_model_name = model_name
            best_model_func = model_func

    logging.info(f"Best Model: {best_model_name} with Loss (SSE): {best_loss:.4f}")
    print(f"\nBest Model: {best_model_name} with Loss (SSE): {best_loss:.4f}")

    # Print O(n) with parameters
    logging.info("Resulting Big-O notations with parameters:")
    print("\nResulting Big-O notations with parameters:")

    # Use best_model and best_model_func instead of iterating and re-fitting
    log_str = ""
    if best_model_name == "Borgwardt's Model (O(m^3 * n))":
        log_str = f"{best_model_name}: O({best_model[0]:.4f} * m^3 * n)"
    elif best_model_name == "Smoothed Analysis (O(mn^5 * log^b(n)))":
        log_str = f"{best_model_name}: O({best_model[0]:.4f} * m * n^5 * log(n)^{best_model[1]:.4f})"
    elif best_model_name == "Adler-Megiddo Model (O(n^4))":
        log_str = f"{best_model_name}: O({best_model[0]:.4f} * n^4)"
    elif best_model_name == "Refined Borgwardt (O(m^3 * n^2))":
        log_str = f"{best_model_name}: O({best_model[0]:.4f} * m^3 * n^2)"
    elif best_model_name == "Refined Smoothed analysis (O(mn^5 * log^b(n)))":
        log_str = f"{best_model_name}: O({best_model[0]:.4f} * m * n^5 * log(n)^{best_model[1]:.4f})"
    elif best_model_name == "General Model (O(m^a n^b))":
        log_str = f"{best_model_name}: O({best_model[0]:.4f} * m^{best_model[1]:.4f} * n^{best_model[2]:.4f})"
    elif best_model_name == "General Model with Log (O(m^a * n^b * log(n)^c))":
        log_str = (f"{best_model_name}: O({best_model[0]:.4f} * m^{best_model[1]:.4f} * n^{best_model[2]:.4f}"
                   f" * log(n)^{best_model[3]:.4f})")
    elif best_model_name == "General Model Mixed (O(a * m^b * n^c + d*m*n))":
        log_str = (f"{best_model_name}: O({best_model[0]:.4f} * m^{best_model[1]:.4f} * n^{best_model[2]:.4f}"
                   f" + {best_model[3]:.4f} * m * n)")
    elif best_model_name == "General Model Mixed Log":
        log_str = (f"{best_model_name}: O({best_model[0]:.4f} * m^{best_model[1]:.4f} * n^{best_model[2]:.4f}"
                   f" * log(n)^{best_model[3]:.4f} + {best_model[4]:.4f}*m*n)")

    logging.info(log_str)
    print(log_str)


if __name__ == "__main__":
    main()
