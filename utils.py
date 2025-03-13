import os
import re
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt


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


def format_parameter_equation(model_name, params):
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
        if params[0] > 10e-8:  # Only include terms with significant weights
            terms.append(f"{params[0]:.4f} × ({params[1]:.6f} × m³ × n)")

        # Smoothed Analysis term
        if params[2] > 10e-8:
            terms.append(f"{params[2]:.4f} × ({params[3]:.6f} × m × n⁵ × log(n)^{params[4]:.6f})")

        # Adler-Megiddo term
        if params[5] > 10e-8:
            terms.append(f"{params[5]:.4f} × ({params[6]:.6f} × n⁴)")

        # Refined Borgwardt term
        if params[7] > 10e-8:
            terms.append(f"{params[7]:.4f} × ({params[8]:.6f} × m³ × n²)")

        # General Model Mixed term
        if params[9] > 10e-8:
            terms.append(
                f"{params[9]:.4f} × ({params[10]:.6f} × m^{params[11]:.6f} × n^{params[12]:.6f} × log(n)^{params[13]:.6f} + {params[14]:.6f} × m^{params[15]:.6f} × n^{params[16]:.6f})")

        return "Operations = " + " + ".join(terms)
    else:
        return "Unknown model"
