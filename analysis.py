import hashlib
import json
import logging
import os
import re
import time
from typing import List, Tuple, Any, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from generator import generate_lp_problem
from simplex import simplex_method

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analysis.log", mode="w"),
        logging.StreamHandler()
    ]
)


class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "computation_cache.json")
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def create_key(self, m: int, n: int, model_type: str) -> str:
        """Create a unique hash key for caching."""
        key_string = f"{m}_{n}_{model_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def save(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        try:
            cache = self._load_cache()
            cache[cache_key] = data

            with open(self.cache_file, "w") as f:
                json.dump(cache, f)
            logging.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logging.error(f"Cache save failed: {e}")

    def load(self, cache_key: str) -> Optional[Any]:
        """Load data from cache."""
        try:
            cache = self._load_cache()
            if cache_key in cache:
                logging.info(f"Cache hit: {cache_key}")
                return cache[cache_key]
            return None
        except Exception as e:
            logging.error(f"Cache load failed: {e}")
            return None

    def _load_cache(self) -> Dict:
        """Load the entire cache file."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}


# Initialize cache manager
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


def collect_data(m_values: List[int], n_values: List[int], num_runs: int = 5) -> List[Tuple]:
    """Collect computational data with caching."""
    data = []
    for m in m_values:
        for n in n_values:
            cache_key = cache_manager.create_key(m, n, "base_computation")
            cached_result = cache_manager.load(cache_key)

            if cached_result is not None:
                data.append(tuple(cached_result))
                continue

            # Compute if not cached
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

                logging.debug(f"Run {run + 1}/{num_runs}: m={m}, n={n}, "
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
            cache_manager.save(cache_key, result)
            data.append(result)

            logging.info(f"m={m}, n={n}: avg_iterations={result[2]:.2f}, "
                         f"avg_operations={result[3]:.2f}, avg_time={result[4]:.4f}s")

    return data


def calculate_loss(data: List[Tuple], model_func: callable, params: np.ndarray) -> float:
    """Calculate sum of squared errors for a model."""
    X = [(d[0], d[1]) for d in data]
    y_true = [d[3] for d in data]
    y_pred = [model_func((m, n), *params) for m, n in X]
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)


def create_plots(data: List[Tuple], model_func: callable, model_name: str,
                 params: np.ndarray, plot_dir: str = "plots"):
    """Create visualization plots in both linear and logarithmic scales."""
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)

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
        plt.title(f"{model_name} - Operations vs. m ({scale.capitalize()} Scale)")
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
        plt.title(f"{model_name} - Operations vs. n ({scale.capitalize()} Scale)")
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)

        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{safe_name}_{scale}_2d.png")
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
            ax.set_title(f"3D Plot - {model_name} ({scale.capitalize()} Scale)")
            plt.colorbar(surf)
            ax.legend()
            ax.grid(True)

            plt.savefig(f"{plot_dir}/{safe_name}_{scale}_3d.png")
            plt.close()


def fit_model(data: List[Tuple], model_func: callable, model_name: str) -> Tuple[Optional[np.ndarray], float]:
    """Fit model to data with caching."""
    cache_key = cache_manager.create_key(
        "_".join(str(d[0]) for d in data[:3]),
        "_".join(str(d[1]) for d in data[:3]),
        model_name
    )

    cached_result = cache_manager.load(cache_key)
    if cached_result is not None:
        return np.array(cached_result[0]), cached_result[1]

    X = np.array([(d[0], d[1]) for d in data]).T
    y = np.array([d[3] for d in data])

    try:
        initial_guess = [1.0] * (model_func.__code__.co_argcount - 1)
        bounds = [(1e-6, None)] * len(initial_guess)

        def objective(params):
            predictions = [model_func((m, n), *params) for m, n in zip(X[0], X[1])]
            return np.sum((np.array(predictions) - np.array(y)) ** 2)

        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

        if result.success:
            params = result.x
            loss = result.fun
            cache_manager.save(cache_key, (params.tolist(), loss))
            create_plots(data, model_func, model_name, params)
            return params, loss
        else:
            logging.error(f"Optimization failed for {model_name}: {result.message}")
            return None, float("inf")

    except Exception as e:
        logging.error(f"Fitting failed for {model_name}: {e}")
        return None, float("inf")


def main():
    # Configuration
    m_values = list(range(200, 2001, 50))
    n_values = list(range(200, 2001, 50))
    num_runs = 5

    # Models to test
    models = [
        (borgwardt_model, "Borgwardt's Model"),
        (smoothed_analysis_model, "Smoothed Analysis Model"),
        (adler_megiddo_model, "Adler-Megiddo Model"),
        (refined_borgwardt_model, "Refined Borgwardt Model"),
        (general_model, "General Model"),
        (general_model_log, "General Model with Log")
    ]

    # Collect data
    logging.info("Starting data collection...")
    data = collect_data(m_values, n_values, num_runs)

    # Fit models and find best
    best_model = None
    best_loss = float("inf")
    best_model_name = ""

    for model_func, model_name in models:
        logging.info(f"Fitting {model_name}...")
        params, loss = fit_model(data, model_func, model_name)

        if loss < best_loss:
            best_loss = loss
            best_model = (model_func, params)
            best_model_name = model_name

    # Save best model
    cache_manager.save("best_model", {
        "name": best_model_name,
        "parameters": best_model[1].tolist() if best_model[1] is not None else None,
        "loss": best_loss
    })

    # Print results
    print(f"\nBest Model: {best_model_name}")
    print(f"Loss (SSE): {best_loss:.4f}")
    if best_model[1] is not None:
        print("Parameters:", best_model[1])


if __name__ == "__main__":
    main()
