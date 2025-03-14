# --- START OF FILE utils.py ---

import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


def calculate_loss(data: List[Tuple], model_func: callable, params: np.ndarray) -> float:
    """Calculate a sum of squared errors for a model."""
    X = [(d[0], d[1]) for d in data]
    y_true = [d[2] for d in data]  # Corrected index
    y_pred = [model_func((m, n), *params) for m, n in X]
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2) / len(data)


def extract_plot_data(data: List[Tuple], model_func: callable, params: np.ndarray) -> Dict[str, Any]:
    """Extracts and organizes data needed for plotting."""
    df = DataFrame(data, columns=['m', 'n', 'operations'])
    plot_data = {
        'm_vals': sorted(df['m'].unique()),
        'n_vals': sorted(df['n'].unique()),
        'data_by_m': {},
        'data_by_n': {},
        'avg_ops_by_m': {},
        'avg_ops_by_n': {},
        'model_fits_m': {},
        'model_fits_n': {},
        'm_grid': None,  # Placeholder
        'n_grid': None,
        'ops_grid': None,
        'm_data_3d': None,
        'n_data_3d': None,
        'ops_data_3d': None
    }

    # 2D Plots Data
    for n in plot_data['n_vals']:
        subset = df[df['n'] == n]
        plot_data['data_by_m'][n] = [subset[subset['m'] == m]['operations'].values for m in plot_data['m_vals']]
        plot_data['avg_ops_by_m'][n] = subset.groupby('m')['operations'].mean().values
        if params is not None:
            m_range = np.linspace(min(plot_data['m_vals']), max(plot_data['m_vals']), 100)
            plot_data['model_fits_m'][n] = (m_range, model_func((m_range, np.full_like(m_range, n)), *params))

    for m in plot_data['m_vals']:
        subset = df[df['m'] == m]
        plot_data['data_by_n'][m] = [subset[subset['n'] == n]['operations'].values for n in plot_data['n_vals']]
        plot_data['avg_ops_by_n'][m] = subset.groupby('n')['operations'].mean().values
        if params is not None:
            n_range = np.linspace(min(plot_data['n_vals']), max(plot_data['n_vals']), 100)
            plot_data['model_fits_n'][m] = (n_range, model_func((np.full_like(n_range, m), n_range), *params))

    # 3D Plots Data
    if params is not None:
        df_grouped = df.groupby(['m', 'n'])['operations'].mean().reset_index()
        plot_data['m_grid'], plot_data['n_grid'] = np.meshgrid(plot_data['m_vals'], plot_data['n_vals'])
        plot_data['ops_grid'] = model_func((plot_data['m_grid'], plot_data['n_grid']), *params)
        plot_data['m_data_3d'] = df_grouped['m'].values
        plot_data['n_data_3d'] = df_grouped['n'].values
        plot_data['ops_data_3d'] = df_grouped['operations'].values

    return plot_data


def create_plots(plot_data: Dict[str, Any], model_func: callable, model_name: str,
                 params: np.ndarray, plot_dir: str = "plots", sample_id: str = ""):
    """Create visualization plots with box plots and connected average lines."""

    os.makedirs(plot_dir, exist_ok=True)

    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    sample_suffix = f"_{sample_id}" if sample_id else ""

    # Create separate figures for linear and log scales
    for scale in ["linear", "log"]:
        plt.figure(figsize=(30, 20), dpi=200)

        # Plot against m
        plt.subplot(1, 2, 1)
        for n in plot_data['n_vals']:
            # Create the box plot
            bp = plt.boxplot(plot_data['data_by_m'][n], positions=plot_data['m_vals'], widths=0.6,
                             patch_artist=True, showfliers=False)

            # Customize box plot appearance
            for box in bp['boxes']:
                box.set(facecolor='lightblue')
            for median in bp['medians']:
                median.set(color='red', linewidth=2)

            # Plot the average line
            plt.plot(plot_data['m_vals'], plot_data['avg_ops_by_m'][n], 'k-', linewidth=1, alpha=0.7)

            if params is not None:
                m_range, model_fit = plot_data['model_fits_m'][n]
                plt.plot(m_range, model_fit, "--", label=f"Fit n={n}")

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
        for m in plot_data['m_vals']:

            # Create the box plot
            bp = plt.boxplot(plot_data['data_by_n'][m], positions=plot_data['n_vals'], widths=0.6,
                             patch_artist=True, showfliers=False)

            # Customize box plot appearance
            for box in bp['boxes']:
                box.set(facecolor='lightblue')
            for median in bp['medians']:
                median.set(color='red', linewidth=2)

            # Plot the average line
            plt.plot(plot_data['n_vals'], plot_data['avg_ops_by_n'][m], 'k-', linewidth=1, alpha=0.7)

            if params is not None:
                n_range, model_fit = plot_data['model_fits_n'][m]
                plt.plot(n_range, model_fit, "--", label=f"Fit m={m}")

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
            ops_grid = plot_data['ops_grid']
            if scale == "log":
                ops_grid = np.log10(ops_grid)

            fig = plt.figure(figsize=(30, 20), dpi=200)
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(plot_data['m_grid'], plot_data['n_grid'], ops_grid, cmap="viridis", alpha=0.7)

            # Plot actual *averaged* data points
            ops_data = plot_data['ops_data_3d']
            if scale == "log":
                ops_data = np.log10(ops_data)

            ax.scatter(plot_data['m_data_3d'], plot_data['n_data_3d'], ops_data, c="red", marker="o",
                       label="Actual Data (Averaged)")

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
