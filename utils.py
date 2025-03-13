import os
import re
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame


def calculate_loss(data: List[Tuple], model_func: callable, params: np.ndarray) -> float:
    """Calculate a sum of squared errors for a model."""
    X = [(d[0], d[1]) for d in data]
    y_true = [d[3] for d in data]
    y_pred = [model_func((m, n), *params) for m, n in X]
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)


def create_plots(data: List[Tuple], model_func: callable, model_name: str,
                 params: np.ndarray, plot_dir: str = "plots", sample_id: str = ""):
    """Create visualization plots with box plots and connected average lines."""

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    sample_suffix = f"_{sample_id}" if sample_id else ""

    # Convert data to DataFrame
    df = DataFrame(data, columns=['m', 'n', 'problem_data', 'operations'])

    # Create separate figures for linear and log scales
    for scale in ["linear", "log"]:
        plt.figure(figsize=(30, 20), dpi=200)

        # Plot against m
        plt.subplot(1, 2, 1)
        for n in sorted(df['n'].unique()):
            subset = df[df['n'] == n]
            # Create a list of operation counts for each unique 'm'
            data_to_plot = [subset[subset['m'] == m]['operations'].values for m in sorted(subset['m'].unique())]
            m_vals = sorted(subset['m'].unique())

            # Create the box plot
            bp = plt.boxplot(data_to_plot, positions=m_vals, widths=0.6, patch_artist=True, showfliers=False)

            # Customize box plot appearance
            for box in bp['boxes']:
                box.set(facecolor='lightblue')
            for median in bp['medians']:
                median.set(color='red', linewidth=2)

            # Plot the average line
            avg_ops = subset.groupby('m')['operations'].mean().values
            plt.plot(m_vals, avg_ops, 'k-', linewidth=1, alpha=0.7) # solid line

            if params is not None:
                m_range = np.linspace(min(m_vals), max(m_vals), 100)
                plt.plot(m_range, model_func((m_range, np.full_like(m_range, n)), *params),
                         "--", label=f"Fit n={n}")


        plt.xlabel("m")
        plt.ylabel("Operations")
        plt.title(f"{model_name} - Operations vs. m ({scale.capitalize()} Scale){" - " + sample_id if sample_id else ""}")
        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)



        # Plot against n
        plt.subplot(1, 2, 2)
        for m in sorted(df['m'].unique()):
            subset = df[df['m'] == m]

            # Create a list of operation counts for each unique 'n'
            data_to_plot = [subset[subset['n'] == n]['operations'].values for n in sorted(subset['n'].unique())]
            n_vals = sorted(subset['n'].unique())

            # Create the box plot
            bp = plt.boxplot(data_to_plot, positions=n_vals, widths=0.6, patch_artist=True, showfliers=False)


            # Customize box plot appearance
            for box in bp['boxes']:
                box.set(facecolor='lightblue')
            for median in bp['medians']:
                median.set(color='red', linewidth=2)

            # Plot the average line
            avg_ops = subset.groupby('n')['operations'].mean().values
            plt.plot(n_vals, avg_ops, 'k-', linewidth=1, alpha=0.7)

            if params is not None:
                n_range = np.linspace(min(n_vals), max(n_vals), 100)
                plt.plot(n_range,
                         model_func((np.full_like(n_range, m), n_range), *params),
                         "--", label=f"Fit m={m}")

        plt.xlabel("n")
        plt.ylabel("Operations")
        plt.title(f"{model_name} - Operations vs. n ({scale.capitalize()} Scale){" - " + sample_id if sample_id else ""}")

        if scale == "log":
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)


        plt.tight_layout()
        plt.savefig(f"{plot_dir}/{safe_name}{sample_suffix}_{scale}_2d.png")
        plt.close()


    # 3D Plots (both scales) -  Keep original 3D plots, just average data
    if params is not None:
        df_grouped = df.groupby(['m', 'n'])['operations'].mean().reset_index() # For 3D plot
        for scale in ["linear", "log"]:
            m_vals = sorted(df_grouped['m'].unique())
            n_vals = sorted(df_grouped['n'].unique())
            m_grid, n_grid = np.meshgrid(m_vals, n_vals)
            ops_grid = model_func((m_grid, n_grid), *params)

            fig = plt.figure(figsize=(30, 20), dpi=200)
            ax = fig.add_subplot(111, projection="3d")

            if scale == "log":
                ops_grid = np.log10(ops_grid)

            surf = ax.plot_surface(m_grid, n_grid, ops_grid, cmap="viridis", alpha=0.7)

            # Plot actual *averaged* data points
            m_data = df_grouped['m'].values
            n_data = df_grouped['n'].values
            ops_data = df_grouped['operations'].values  # Use the averaged values

            if scale == "log":
                ops_data = np.log10(ops_data)

            ax.scatter(m_data, n_data, ops_data, c="red", marker="o", label="Actual Data (Averaged)")

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
