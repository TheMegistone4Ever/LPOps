import logging
import os
import re
import time
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from cache_manager import CacheManager
from generator import generate_lp_problem
from simplex import simplex_method

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analysis_1.log", mode="w"),
        logging.StreamHandler()
    ]
)

cache_manager = CacheManager()


# Model definitions
def borgwardt_model(X: Tuple[np.ndarray, np.ndarray], a: float) -> np.ndarray:
    """Borgwardt's Model: O(m^2 * n^(1/(m-1))) -> O(m^3 * n) when m scales with n."""
    m, n = X
    return a * m ** 3 * n


def smoothed_analysis_model(X: Tuple[np.ndarray, np.ndarray], a: float, b: float) -> np.ndarray:
    """Smoothed Analysis Model: O(mn^5 * log^b(n))."""
    m, n = X
    return a * m * n ** 5 * np.log(n) ** b


def adler_megiddo_model(X: Tuple[np.ndarray, np.ndarray], a: float) -> np.ndarray:
    """Adler-Megiddo Model: O(n^4)."""
    m, n = X
    return a * n ** 4


def refined_borgwardt_model(X: Tuple[np.ndarray, np.ndarray], a: float) -> np.ndarray:
    """Refined bound: O(m^3n^2)."""
    m, n = X
    return a * m ** 3 * n ** 2


def general_model(X: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float) -> np.ndarray:
    """General model O(m^a n^b)."""
    m, n = X
    return a * m ** b * n ** c


def general_model_log(X: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float, d: float) -> np.ndarray:
    """General model with log terms: O(m^a * n^b * log(n)^c)."""
    m, n = X
    log_n = np.where(n > 1, np.log(n), 0)
    return a * (m ** b) * (n ** c) * log_n ** d


def general_model_mixed(X: Tuple[np.ndarray, np.ndarray], a: float, b: float, c: float, d: float, e: float,
                        f: float, g: float) -> np.ndarray:
    """General model with mixed terms: O(m^a * n^b * log(n)^c + d * m^e * n^f)."""
    m, n = X
    log_n = np.where(n > 1, np.log(n), 0)
    return a * (m ** b) * (n ** c) * (log_n ** d) + e * (m ** f) * (n ** g)


def weighted_ensemble_model(X: Tuple[np.ndarray, np.ndarray], *params) -> np.ndarray:
    """
    Weighted ensemble of models.
    Parameters structure:
    [weight1, a1, weight2, a2, b2, weight3, a3, weight4, a4, weight5, a5, b5, c5, d5, e5, f5, g5]
    """
    m, n = X

    # Extract weights and individual model parameters
    w1, a1 = params[0], params[1]  # Borgwardt
    w2, a2, b2 = params[2], params[3], params[4]  # Smoothed Analysis
    w3, a3 = params[5], params[6]  # Adler-Megiddo
    w4, a4 = params[7], params[8]  # Refined Borgwardt
    w5, a5, b5, c5, d5, e5, f5, g5 = params[9], params[10], params[11], params[12], params[13], params[14], params[15], \
        params[16]  # General Mixed

    # Combine models
    return (w1 * borgwardt_model(X, a1) +
            w2 * smoothed_analysis_model(X, a2, b2) +
            w3 * adler_megiddo_model(X, a3) +
            w4 * refined_borgwardt_model(X, a4) +
            w5 * general_model_mixed(X, a5, b5, c5, d5, e5, f5, g5))


def collect_data(m_values: List[int], n_values: List[int], sample_id: str = "W1", sample_idx: int = 1,
                 num_runs: int = 5) -> List[Tuple]:
    """Collect computational data with caching."""
    data = []
    for m in m_values:
        for n in n_values:
            key = f"{sample_id}_{m}_{n}_base_computation"
            cached_result = cache_manager.load(key)

            if cached_result is not None:
                data.append(tuple(cached_result))
                continue

            # Compute if not cached
            total_iterations = 0
            total_operations = 0
            total_time = 0

            for run in range(num_runs):
                c, A, b, _ = generate_lp_problem(m, n, factor=sample_idx)
                start_time = time.time()
                _, _, iterations, operations = simplex_method(c, A, b)
                end_time = time.time()

                total_iterations += iterations
                total_operations += operations
                total_time += (end_time - start_time)

                logging.debug(f"Sample {sample_id}, Run {run + 1}/{num_runs}: m={m}, n={n}, "
                              f"iterations={iterations}, operations={operations}, "
                              f"time={end_time - start_time:.4f}s")

            # Calculate averages
            result = (
                m,
                n,
                total_iterations / num_runs,
                total_operations / num_runs,
                total_time / num_runs
            )

            # Cache result
            cache_manager.save(key, result)
            data.append(result)

            logging.info(f"Sample {sample_id}, m={m}, n={n}: avg_iterations={result[2]:.2f}, "
                         f"avg_operations={result[3]:.2f}, avg_time={result[4]:.4f}s")

    return data


def calculate_loss(data: List[Tuple], model_func: callable, params: np.ndarray) -> float:
    """Calculate a sum of squared errors for a model."""
    X = [(d[0], d[1]) for d in data]
    y_true = [d[3] for d in data]
    y_pred = [model_func((m, n), *params) for m, n in X]
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)


def create_plots(data: List[Tuple], model_func: callable, model_name: str,
                 params: np.ndarray, plot_dir: str = "plots", sample_id: str = ""):
    """Create visualization plots in both linear and logarithmic scales."""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    sample_suffix = f"_{sample_id}" if sample_id else ""

    # Create separate figures for linear and log scales
    for scale in ["linear", "log"]:
        plt.figure(figsize=(30, 20), dpi=200)

        # Plot against m
        plt.subplot(1, 2, 1)
        for n in sorted(set(d[1] for d in data)):
            m_vals = [d[0] for d in data if d[1] == n]
            ops = [d[3] for d in data if d[1] == n]
            plt.plot(m_vals, ops, "o-", label=f"n={n}")

            if params is not None:
                m_range = np.linspace(min(m_vals), max(m_vals), 100)
                plt.plot(m_range,
                         model_func((m_range, np.full_like(m_range, n)), *params),
                         "--", label=f"Fit n={n}")

        plt.xlabel("m")
        plt.ylabel("Operations")
        plt.title(
            f"{model_name} - Operations vs. m ({scale.capitalize()} Scale){" - " + sample_id if sample_id else ""}")
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        # Plot against n
        plt.subplot(1, 2, 2)
        for m in sorted(set(d[0] for d in data)):
            n_vals = [d[1] for d in data if d[0] == m]
            ops = [d[3] for d in data if d[0] == m]
            plt.plot(n_vals, ops, "o-", label=f"m={m}")

            if params is not None:
                n_range = np.linspace(min(n_vals), max(n_vals), 100)
                plt.plot(n_range,
                         model_func((np.full_like(n_range, m), n_range), *params),
                         "--", label=f"Fit m={m}")

        plt.xlabel("n")
        plt.ylabel("Operations")
        plt.title(
            f"{model_name} - Operations vs. n ({scale.capitalize()} Scale){" - " + sample_id if sample_id else ""}")
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{safe_name}{sample_suffix}_{scale}_2d.png")
        plt.close()

    # 3D Plots (both scales)
    if params is not None:
        for scale in ["linear", "log"]:
            m_vals = sorted(set(d[0] for d in data))
            n_vals = sorted(set(d[1] for d in data))
            m_grid, n_grid = np.meshgrid(m_vals, n_vals)
            ops_grid = model_func((m_grid, n_grid), *params)

            fig = plt.figure(figsize=(30, 20), dpi=200)
            ax = fig.add_subplot(111, projection="3d")

            if scale == "log":
                ops_grid = np.log10(ops_grid)

            surf = ax.plot_surface(m_grid, n_grid, ops_grid, cmap="viridis", alpha=0.7)

            # Plot actual data points
            m_data = [d[0] for d in data]
            n_data = [d[1] for d in data]
            ops_data = [d[3] for d in data]

            if scale == "log":
                ops_data = np.log10(ops_data)

            ax.scatter(m_data, n_data, ops_data, c="red", marker="o", label="Actual Data")

            ax.set_xlabel("m")
            ax.set_ylabel("n")
            ax.set_zlabel("Operations (log10)" if scale == "log" else "Operations")
            ax.set_title(f"3D Plot - {model_name} ({scale.capitalize()} Scale){" - " + sample_id if sample_id else ""}")
            plt.colorbar(surf)
            ax.legend()
            ax.grid(True)

            plt.savefig(f"{plot_dir}/{safe_name}{sample_suffix}_{scale}_3d.png")
            plt.close()


def fit_model(data: List[Tuple], model_func: callable, model_name: str,
              sample_id: str = "W1", initial_params: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], float]:
    """Fit model to data with caching."""

    key = f"model_fit_{sample_id}_{model_name.replace(" ", "_").lower()}_{"_".join(str(d[0]) + "_" + str(d[1]) for d in data[:3])}"

    cached_result = cache_manager.load(key)
    if cached_result is not None:
        return np.array(cached_result[0]), cached_result[1]

    X = np.array([(d[0], d[1]) for d in data]).T
    y = np.array([d[3] for d in data])

    try:
        if initial_params is None:
            initial_guess = [1.0] * (model_func.__code__.co_argcount - 1)
        else:
            initial_guess = initial_params.tolist()

        bounds = [(1e-6, None)] * len(initial_guess)

        # For weighted ensemble model, constrain weights to sum to 1
        if model_func.__name__ == "weighted_ensemble_model":
            weight_indices = [0, 2, 5, 7, 9]  # Indices of weights in the params array
            initial_weights = [initial_guess[i] for i in weight_indices]
            total_weight = sum(initial_weights)
            for i in weight_indices:
                initial_guess[i] /= total_weight  # Normalize weights

        def objective(params):
            predictions = [model_func((m, n), *params) for m, n in zip(X[0], X[1])]
            return np.sum((np.array(predictions) - np.array(y)) ** 2)

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            params = result.x
            loss = result.fun
            cache_manager.save(key, (params.tolist(), loss))
            create_plots(data, model_func, model_name, params, sample_id=sample_id)
            return params, loss
        else:
            logging.error(f"Optimization failed for {model_name} on sample {sample_id}: {result.message}")
            return None, float("inf")

    except Exception as e:
        logging.error(f"Fitting failed for {model_name} on sample {sample_id}: {e}")
        return None, float("inf")


def format_parameter_equation(model_name, params, model_func):
    """Format model parameters as an equation string."""
    if model_name == "Borgwardt's Model":
        return f"Operations = {params[0]:.6f} × m³ × n"
    elif model_name == "Smoothed Analysis Model":
        return f"Operations = {params[0]:.6f} × m × n⁵ × log(n)^{params[1]:.6f}"
    elif model_name == "Adler-Megiddo Model":
        return f"Operations = {params[0]:.6f} × n⁴"
    elif model_name == "Refined Borgwardt Model":
        return f"Operations = {params[0]:.6f} × m³ × n²"
    elif model_name == "General Model":
        return f"Operations = {params[0]:.6f} × m^{params[1]:.6f} × n^{params[2]:.6f}"
    elif model_name == "General Model with Log":
        return f"Operations = {params[0]:.6f} × m^{params[1]:.6f} × n^{params[2]:.6f} × log(n)^{params[3]:.6f}"
    elif model_name == "General Model Mixed":
        return f"Operations = {params[0]:.6f} × m^{params[1]:.6f} × n^{params[2]:.6f} × log(n)^{params[3]:.6f} + {params[4]:.6f} × m^{params[5]:.6f} × n^{params[6]:.6f}"
    elif model_name == "Weighted Ensemble Model":
        # Generate ensemble equation
        weight_indices = [0, 2, 5, 7, 9]
        terms = []

        # Borgwardt term
        if params[0] > 0.01:  # Only include terms with significant weights
            terms.append(f"{params[0]:.4f} × ({params[1]:.6f} × m³ × n)")

        # Smoothed Analysis term
        if params[2] > 0.01:
            terms.append(f"{params[2]:.4f} × ({params[3]:.6f} × m × n⁵ × log(n)^{params[4]:.6f})")

        # Adler-Megiddo term
        if params[5] > 0.01:
            terms.append(f"{params[5]:.4f} × ({params[6]:.6f} × n⁴)")

        # Refined Borgwardt term
        if params[7] > 0.01:
            terms.append(f"{params[7]:.4f} × ({params[8]:.6f} × m³ × n²)")

        # General Model Mixed term
        if params[9] > 0.01:
            terms.append(
                f"{params[9]:.4f} × ({params[10]:.6f} × m^{params[11]:.6f} × n^{params[12]:.6f} × log(n)^{params[13]:.6f} + {params[14]:.6f} × m^{params[15]:.6f} × n^{params[16]:.6f})")

        return "Operations = " + " + ".join(terms)
    else:
        return "Unknown model"


def main(num_samples: int = 2):
    # Configuration
    m_values = list(range(200, 2001, 50))
    n_values = list(range(200, 2001, 50))
    num_runs = 5

    # Models to test (for individual analysis)
    models = [
        (borgwardt_model, "Borgwardt's Model"),
        (smoothed_analysis_model, "Smoothed Analysis Model"),
        (adler_megiddo_model, "Adler-Megiddo Model"),
        (refined_borgwardt_model, "Refined Borgwardt Model"),
        (general_model, "General Model"),
        (general_model_log, "General Model with Log"),
        (general_model_mixed, "General Model Mixed")
    ]

    # Store fitted parameters for each model across samples
    all_models_params = {model_name: [] for _, model_name in models}
    all_models_params["Weighted Ensemble Model"] = []

    # Store losses for each model across samples
    all_models_losses = {model_name: [] for _, model_name in models}
    all_models_losses["Weighted Ensemble Model"] = []

    # Dictionary to store best single model parameters from each sample
    best_single_models = {}

    # Process each sample
    for sample_idx in range(1, num_samples + 1):
        sample_id = f"W{sample_idx}"
        logging.info(f"Processing sample {sample_id}...")

        # Collect data for this sample
        data = collect_data(m_values, n_values, sample_id, sample_idx, num_runs)

        # Fit individual models
        sample_best_model = None
        sample_best_loss = float("inf")
        sample_best_model_name = ""
        sample_best_params = None

        # Dictionary to store all fitted models for this sample (for ensemble)
        sample_fitted_models = {}

        for model_func, model_name in models:
            logging.info(f"Fitting {model_name} on sample {sample_id}...")

            # Check if we have parameters from a previous sample to use as initial guess
            initial_params = None
            if sample_idx > 1 and all_models_params[model_name]:
                initial_params = np.array(all_models_params[model_name][0])  # Use first sample's params

            params, loss = fit_model(data, model_func, model_name, sample_id, initial_params)

            # Store the fitted parameters and loss
            all_models_params[model_name].append(params)
            all_models_losses[model_name].append(loss)

            # Save for ensemble model
            sample_fitted_models[model_name] = (params, loss)

            if loss < sample_best_loss:
                sample_best_loss = loss
                sample_best_model = (model_func, params)
                sample_best_model_name = model_name
                sample_best_params = params

        # Save the best model for this sample
        best_single_models[sample_id] = {
            "name": sample_best_model_name,
            "parameters": sample_best_params.tolist() if sample_best_params is not None else None,
            "loss": sample_best_loss
        }

        # Build and fit the ensemble model
        logging.info(f"Building ensemble model on sample {sample_id}...")

        # Initial weights give equal importance to the 5 models in the ensemble
        ensemble_initial_params = []

        # Add Borgwardt's Model parameters
        borgwardt_params = sample_fitted_models["Borgwardt's Model"][0]
        ensemble_initial_params.extend([1.0 / 5, borgwardt_params[0]])

        # Add Smoothed Analysis Model parameters
        smoothed_params = sample_fitted_models["Smoothed Analysis Model"][0]
        ensemble_initial_params.extend([1.0 / 5, smoothed_params[0], smoothed_params[1]])

        # Add Adler-Megiddo Model parameters
        adler_params = sample_fitted_models["Adler-Megiddo Model"][0]
        ensemble_initial_params.extend([1.0 / 5, adler_params[0]])

        # Add Refined Borgwardt Model parameters
        refined_params = sample_fitted_models["Refined Borgwardt Model"][0]
        ensemble_initial_params.extend([1.0 / 5, refined_params[0]])

        # Add General Model Mixed parameters
        mixed_params = sample_fitted_models["General Model Mixed"][0]
        ensemble_initial_params.extend([1.0 / 5, mixed_params[0], mixed_params[1], mixed_params[2],
                                        mixed_params[3], mixed_params[4], mixed_params[5], mixed_params[6]])

        # Check if we have ensemble parameters from a previous sample
        ensemble_params_from_prev = None
        if sample_idx > 1 and all_models_params["Weighted Ensemble Model"]:
            ensemble_params_from_prev = np.array(all_models_params["Weighted Ensemble Model"][0])



        # Fit the ensemble model
        ensemble_params, ensemble_loss = fit_model(
            data,
            weighted_ensemble_model,
            "Weighted Ensemble Model",
            f"L{sample_idx}",
            initial_params=ensemble_params_from_prev if ensemble_params_from_prev is not None else np.array(
                ensemble_initial_params)
        )

        # Store ensemble model parameters and loss
        all_models_params["Weighted Ensemble Model"].append(ensemble_params)
        all_models_losses["Weighted Ensemble Model"].append(ensemble_loss)

    # Calculate average parameters for each model across samples
    avg_params = {}
    for model_name, params_list in all_models_params.items():
        if params_list and all(p is not None for p in params_list):
            avg_params[model_name] = np.mean(np.array(params_list), axis=0)

    # Determine the best single model based on average loss
    best_single_model_name = None
    best_single_model_avg_loss = float("inf")

    for model_name, losses in all_models_losses.items():
        if model_name != "Weighted Ensemble Model":  # Exclude ensemble from single model comparison
            avg_loss = np.mean(losses)
            if avg_loss < best_single_model_avg_loss:
                best_single_model_avg_loss = avg_loss
                best_single_model_name = model_name

    # Compare the best single model with the ensemble model
    ensemble_avg_loss = np.mean(all_models_losses["Weighted Ensemble Model"])

    # Output results
    print("\n=== RESULTS ===\n")

    # Print parameters for each model across samples
    print("=== INDIVIDUAL MODEL PARAMETERS ===")
    for model_name in all_models_params.keys():
        if model_name != "Weighted Ensemble Model":
            print(f"\n{model_name}:")

            for i, params in enumerate(all_models_params[model_name]):
                if params is not None:
                    sample_id = f"W{i + 1}" if model_name != "Weighted Ensemble Model" else f"L{i + 1}"
                    print(f"  Sample {sample_id}: {format_parameter_equation(model_name, params, None)}")
                    print(f"  Loss: {all_models_losses[model_name][i]:.4f}")

            if model_name in avg_params:
                print(f"  Average: {format_parameter_equation(model_name, avg_params[model_name], None)}")
                print(f"  Average Loss: {np.mean(all_models_losses[model_name]):.4f}")

    # Print best single model
    print("\n=== BEST SINGLE MODEL ===")
    print(f"Model: {best_single_model_name}")
    print(f"Average Loss: {best_single_model_avg_loss:.4f}")
    if best_single_model_name in avg_params:
        print(
            f"Parameters: {format_parameter_equation(best_single_model_name, avg_params[best_single_model_name], None)}")

    # Print ensemble model results
    print("\n=== WEIGHTED ENSEMBLE MODEL ===")
    for i, params in enumerate(all_models_params["Weighted Ensemble Model"]):
        if params is not None:
            print(f"  Sample L{i + 1}: {format_parameter_equation("Weighted Ensemble Model", params, None)}")
            print(f"  Loss: {all_models_losses["Weighted Ensemble Model"][i]:.4f}")

    if "Weighted Ensemble Model" in avg_params:
        print(
            f"  Average: {format_parameter_equation("Weighted Ensemble Model", avg_params["Weighted Ensemble Model"], None)}")
        print(f"  Average Loss: {ensemble_avg_loss:.4f}")

    # Determine and print the overall best model
    if ensemble_avg_loss < best_single_model_avg_loss:
        print("\n=== OVERALL BEST MODEL ===")
        print("The Weighted Ensemble Model outperforms all individual models.")
    else:
        print("\n=== OVERALL BEST MODEL ===")
        print(f"The best individual model ({best_single_model_name}) outperforms the ensemble.")

    # Save all results to cache
    cache_manager.save("analysis_results", {
        "individual_models": {
            model_name: {
                "parameters": [p.tolist() if p is not None else None for p in params],
                "losses": losses,
                "avg_parameters": avg_params.get(model_name, None).tolist() if model_name in avg_params else None,
                "avg_loss": np.mean(losses)
            } for model_name, params, losses in zip(
                all_models_params.keys(),
                all_models_params.values(),
                all_models_losses.values()
            )
        },
        "best_single_model": {
            "name": best_single_model_name,
            "avg_loss": best_single_model_avg_loss,
            "avg_parameters": avg_params.get(best_single_model_name,
                                             None).tolist() if best_single_model_name in avg_params else None
        },
        "ensemble_model": {
            "parameters": [p.tolist() if p is not None else None for p in all_models_params["Weighted Ensemble Model"]],
            "losses": all_models_losses["Weighted Ensemble Model"],
            "avg_parameters": avg_params.get("Weighted Ensemble Model",
                                             None).tolist() if "Weighted Ensemble Model" in avg_params else None,
            "avg_loss": ensemble_avg_loss
        },
        "overall_best_model": "Weighted Ensemble Model" if ensemble_avg_loss < best_single_model_avg_loss else best_single_model_name
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and optimize LP models.")
    parser.add_argument("--samples", type=int, default=2, help="Number of samples to use for analysis (default: 2)")

    args = parser.parse_args()

    main(num_samples=args.samples)
