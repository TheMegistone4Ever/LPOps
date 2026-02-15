import gc
import logging
import math
import os
import pickle
import re
import sys
import time
import warnings
from itertools import combinations, islice
from typing import Any, Dict, List, Tuple, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from tqdm import tqdm

warnings.filterwarnings("ignore")

CACHE_DIR = "cache"
PLOTS_DIR = "plots_combi"
LOG_FILE = "gmdh_execution.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
BATCH_SIZE_INITIAL = 4096

os.makedirs(PLOTS_DIR, exist_ok=True)

if torch.cuda.is_available():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("gmdh")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


log = _configure_logging()


def _f_refined(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return (m ** 3) * (n ** 2)


def _f_smoothed(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return m * (n ** 5) * np.log(np.where(n > 1, n, 1.1))


def _f_poly_mn(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return m * n


def _safe_log(n: np.ndarray) -> np.ndarray:
    return np.where(n > 1, np.log(n), .0)


def _f_general(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return (
            .63 * m ** 2.96 * n ** .02 * _safe_log(n) ** 1.62
            + 4.04 * m ** -4.11 * n ** 2.92
    )


def _f_adler_megiddo(_: np.ndarray, n: np.ndarray) -> np.ndarray:
    return n ** 4


BASIS_FUNCTIONS = {
    "m3n2": _f_refined,
    "mn5lnn": _f_smoothed,
    "poly_mn": _f_poly_mn,
    "general": _f_general,
    "adler_megiddo": _f_adler_megiddo,
}


def generate_full_feature_matrix(
        df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    m = df["m"].values
    n = df["n"].values

    base_data = {name: func(m, n) for name, func in BASIS_FUNCTIONS.items()}
    base_names = list(base_data.keys())

    final_data = dict()
    feature_names = list()

    for name in base_names:
        final_data[name] = base_data[name]
        feature_names.append(name)

    for name in base_names:
        sq_name = f"({name})^2"
        final_data[sq_name] = base_data[name] ** 2
        feature_names.append(sq_name)

    for name_a, name_b in combinations(base_names, 2):
        cross_name = f"{name_a} * {name_b}"
        final_data[cross_name] = base_data[name_a] * base_data[name_b]
        feature_names.append(cross_name)

    return pd.DataFrame(final_data), feature_names


def load_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_DIR):
        return pd.DataFrame()

    all_data: List[List] = list()
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]

    for function_name in tqdm(files, desc="Loading cache", ncols=100):
        try:
            with open(os.path.join(CACHE_DIR, function_name), "rb") as fh:
                data = pickle.load(fh)
                if not data:
                    continue
                if isinstance(data, list) and len(data) > 3:
                    all_data.extend(data)
                elif isinstance(data, (list, tuple)) and len(data) == 3:
                    all_data.append(list(data))
        except (pickle.UnpicklingError, EOFError):
            continue

    if not all_data:
        return pd.DataFrame()

    df = (
        pd.DataFrame(all_data, columns=["m", "n", "ops"])
        .drop_duplicates()
        .astype(float)
    )
    return df[df["ops"] > 0]


class AsyncGPULoader:
    def __init__(self, iterator, device: torch.device):
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self.next_cpu_indices = None
        self._preload()

    def _preload(self):
        try:
            self.next_cpu_indices = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            self.next_cpu_indices = None
            return

        tensor = torch.tensor(self.next_cpu_indices, dtype=torch.long).pin_memory()
        with torch.cuda.stream(self.stream):
            self.next_batch = tensor.to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_gpu = self.next_batch
        batch_cpu = self.next_cpu_indices
        self._preload()
        return batch_cpu, batch_gpu


def _batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


@torch.no_grad()
def solve_batch_torch(
        idx_tensor: torch.Tensor,
        xt_x: torch.Tensor,
        xt_y: torch.Tensor,
        x_test_t: torch.Tensor,
        y_test: torch.Tensor,
        alpha: float,
) -> Tuple[np.ndarray, torch.Tensor]:
    batch_size = idx_tensor.size(0)
    k = idx_tensor.size(1)

    rows = idx_tensor.unsqueeze(2).expand(batch_size, k, k)
    cols = idx_tensor.unsqueeze(1).expand(batch_size, k, k)
    a_batch = xt_x[rows, cols]

    identity = (
        torch.eye(k, device=DEVICE, dtype=DTYPE)
        .unsqueeze(0)
        .expand(batch_size, k, k)
    )
    a_batch.add_(identity, alpha=alpha)

    b_batch = xt_y[idx_tensor]

    weights = None
    solver_used = None

    try:
        cholesky = torch.linalg.cholesky(a_batch)
        weights = torch.cholesky_solve(b_batch, cholesky)
        solver_used = "cholesky"
    except RuntimeError:
        pass

    if weights is None:
        try:
            weights = torch.linalg.solve(a_batch, b_batch)
            solver_used = "solve"
        except RuntimeError:
            pass

    if weights is None:
        try:
            weights = torch.linalg.lstsq(a_batch, b_batch, rcond=1e-4).solution
            solver_used = "lstsq"
        except RuntimeError:
            pass

    if weights is None:
        a_pinv = torch.linalg.pinv(a_batch, rcond=1e-6)
        weights = torch.bmm(a_pinv, b_batch)
        solver_used = "pinv"

    if solver_used == "pinv":
        log.debug(f"Using pseudoinverse for batch (feature combination may be rank-deficient)")

    x_test_sub = x_test_t[idx_tensor].permute(0, 2, 1)
    y_pred = torch.bmm(x_test_sub, weights).squeeze(2)

    diff = y_pred.sub(y_test.unsqueeze(0))
    mse_batch = torch.mean(diff.pow(2), dim=1)

    return mse_batch.cpu().numpy(), weights


def perform_coefficient_clustering(
        coefs: np.ndarray,
        feature_names: List[str],
        x_data: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    log.info("Coefficient clustering initiated")
    t0 = time.perf_counter()

    feature_stds = np.std(x_data, axis=0)
    items = list()
    for w, std, name in zip(coefs, feature_stds, feature_names):
        importance = abs(w) * (std if std > 1e-9 else 1.)
        items.append({"raw_val": w, "name": name, "importance": importance})

    sorted_items = sorted(items, key=lambda x: x["importance"], reverse=True)

    if not sorted_items:
        return np.array(list()), list()

    smallest_imp = sorted_items[-1]["importance"]
    m1 = [sorted_items[0]]

    for i in range(1, len(sorted_items)):
        candidate = sorted_items[i]
        avg_m1 = sum(item["importance"] for item in m1) / len(m1)
        dist_good = avg_m1 - candidate["importance"]
        dist_bad = candidate["importance"] - smallest_imp
        if dist_good < dist_bad:
            m1.append(candidate)
        else:
            break

    final_names = [item["name"] for item in m1]
    final_coefs = np.array([item["raw_val"] for item in m1])

    elapsed = time.perf_counter() - t0
    log.info(
        "Clustering complete: retained %d / %d features (%.4f s): %s",
        len(m1),
        len(sorted_items),
        elapsed,
        ", ".join(final_names),
    )
    return final_coefs, final_names


def _gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def _fit_gaussian(x, y):
    try:
        popt, _ = curve_fit(
            _gaussian,
            x,
            y,
            p0=[max(y), x[np.argmax(y)], 1.],
            maxfev=5000,
        )
        return popt
    except (RuntimeError, ValueError):
        return None


def plot_metrics(
        k_values: List[int],
        combinations_list: List[int],
        speeds: List[float],
        best_mses: List[float],
        elapsed_times: List[float],
) -> None:
    k_arr = np.array(k_values, dtype=float)
    comb_arr = np.array(combinations_list, dtype=float)
    speed_arr = np.array(speeds, dtype=float)
    mse_arr = np.array(best_mses, dtype=float)
    time_arr = np.array(elapsed_times, dtype=float)
    x_smooth = np.linspace(k_arr.min(), k_arr.max(), 200)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(k_arr, mse_arr, "b-o", lw=2, markersize=6, label="Best MSE")
    if len(k_arr) > 2:
        try:
            log_mse = np.log(mse_arr + 1e-30)
            z = np.polyfit(k_arr, log_mse, 2)
            p = np.poly1d(z)
            ax.plot(
                x_smooth,
                np.exp(p(x_smooth)),
                "r--",
                lw=2,
                alpha=.4,
                label="Quadratic fit (log-space)",
            )
        except (Exception,):
            pass
    ax.set_yscale("log")
    ax.set_xlabel("Model complexity (k)")
    ax.set_ylabel("MSE")
    ax.set_title("Best MSE per k")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metric_mse_vs_k.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(k_arr, comb_arr, "ko", alpha=.6, label="Actual combinations")
    try:
        popt = _fit_gaussian(k_arr, comb_arr)
        ax.plot(
            x_smooth,
            _gaussian(x_smooth, *popt),
            "r--",
            lw=2,
            alpha=.4,
            label="Gaussian fit",
        )
    except (Exception,):
        pass
    ax.set_xlabel("k")
    ax.set_ylabel("Number of combinations")
    ax.set_title("Combinations vs k")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metric_combinations_vs_k.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(k_arr, speed_arr, "g.", markersize=8, label="Measured speed (it/s)")
    if len(k_arr) > 3:
        try:
            z = np.polyfit(k_arr, speed_arr, 4)
            p = np.poly1d(z)
            ax.plot(
                x_smooth,
                p(x_smooth),
                "r--",
                lw=2,
                alpha=.4,
                label="Polynomial fit (deg 4)",
            )
        except (Exception,):
            pass
    ax.set_xlabel("k")
    ax.set_ylabel("Speed (iterations / s)")
    ax.set_title("Processing speed vs k")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metric_speed_vs_k.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(k_arr, time_arr, "m-s", lw=2, markersize=6, label="Total time")
    if len(k_arr) > 2:
        try:
            popt = _fit_gaussian(k_arr, time_arr)
            ax.plot(
                x_smooth,
                _gaussian(x_smooth, *popt),
                "r--",
                lw=2,
                alpha=.4,
                label="Gaussian fit",
            )
        except (Exception,):
            pass
    ax.set_xlabel("k")
    ax.set_ylabel("Processing time (s)")
    ax.set_title("Total processing time per k")
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "metric_time_vs_k.png"))
    plt.close(fig)


def plot_loss_history(history: List[float]) -> None:
    history = [h for h in history if h < 1e30]
    if not history:
        return

    accumulated_min = list()
    current_min = float("inf")
    for h in history:
        current_min = min(current_min, h)
        accumulated_min.append(current_min)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    ax.plot(
        history,
        marker=".",
        linestyle="none",
        markersize=2,
        alpha=.3,
        color="gray",
        label="Sample models",
    )
    ax.plot(accumulated_min, "r-", lw=2, label="Best found")
    ax.set_yscale("log")
    ax.set_title("MSE loss dynamics")
    ax.set_xlabel("Search step")
    ax.set_ylabel("MSE")
    ax.grid(True, which="both", alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss_distribution_log.png"))
    plt.close(fig)


def extract_plot_data(
        data: Any,
        model_func: Callable[[Tuple[np.ndarray, np.ndarray], Any], np.ndarray],
        params: List,
) -> Dict[str, Any]:
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()

    plot_data: Dict[str, Any] = {
        "m_vals": sorted(df["m"].unique()),
        "n_vals": sorted(df["n"].unique()),
        "data_by_m": dict(),
        "data_by_n": dict(),
        "avg_ops_by_m": dict(),
        "avg_ops_by_n": dict(),
        "model_fits_m": dict(),
        "model_fits_n": dict(),
        "m_grid": None,
        "n_grid": None,
        "ops_grid": None,
        "m_data_3d": None,
        "n_data_3d": None,
        "ops_data_3d": None,
    }

    m_vals = plot_data["m_vals"]
    n_vals = plot_data["n_vals"]

    for n in n_vals:
        subset = df[df["n"] == n]
        plot_data["data_by_m"][n] = [
            subset[subset["m"] == m]["ops"].values for m in m_vals
        ]
        plot_data["avg_ops_by_m"][n] = subset.groupby("m")["ops"].mean().values  # type: ignore
        if params is not None:
            m_range = np.linspace(min(m_vals), max(m_vals), 100)
            plot_data["model_fits_m"][n] = (
                m_range,
                model_func((m_range, np.full_like(m_range, n)), *params),
            )

    for m in m_vals:
        subset = df[df["m"] == m]
        plot_data["data_by_n"][m] = [
            subset[subset["n"] == n]["ops"].values for n in n_vals
        ]
        plot_data["avg_ops_by_n"][m] = subset.groupby("n")["ops"].mean().values  # type: ignore
        if params is not None:
            n_range = np.linspace(min(n_vals), max(n_vals), 100)
            plot_data["model_fits_n"][m] = (
                n_range,
                model_func((np.full_like(n_range, m), n_range), *params),
            )

    if params is not None:
        df_grouped = df.groupby(["m", "n"])["ops"].mean().reset_index()
        plot_data["m_grid"], plot_data["n_grid"] = np.meshgrid(m_vals, n_vals)
        plot_data["ops_grid"] = model_func(
            (plot_data["m_grid"], plot_data["n_grid"]), *params
        )
        plot_data["m_data_3d"] = df_grouped["m"].values
        plot_data["n_data_3d"] = df_grouped["n"].values
        plot_data["ops_data_3d"] = df_grouped["ops"].values

    return plot_data


def create_plots(
        plot_data: Dict[str, Any],
        model_name: str,
        params: Any,
        plot_dir: str,
) -> None:
    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)

    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(30, 20), dpi=200)

        ax = axes[0]
        for n in plot_data["n_vals"]:
            bp = ax.boxplot(
                plot_data["data_by_m"][n],
                positions=plot_data["m_vals"],
                widths=.6,
                patch_artist=True,
                showfliers=False,
            )
            for box in bp["boxes"]:
                box.set(facecolor="lightblue")
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)
            ax.plot(
                plot_data["m_vals"],
                plot_data["avg_ops_by_m"][n],
                "k-",
                linewidth=1,
                alpha=.7,
            )
            if params is not None:
                m_r, fit = plot_data["model_fits_m"][n]
                ax.plot(m_r, fit, "--", label=f"n={n}")
        ax.set_xlabel("m")
        ax.set_ylabel("Ops")
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=.2)

        ax = axes[1]
        for m in plot_data["m_vals"]:
            bp = ax.boxplot(
                plot_data["data_by_n"][m],
                positions=plot_data["n_vals"],
                widths=.6,
                patch_artist=True,
                showfliers=False,
            )
            for box in bp["boxes"]:
                box.set(facecolor="lightblue")
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)
            ax.plot(
                plot_data["n_vals"],
                plot_data["avg_ops_by_n"][m],
                "k-",
                linewidth=1,
                alpha=.7,
            )
            if params is not None:
                n_r, fit = plot_data["model_fits_n"][m]
                ax.plot(n_r, fit, "--", label=f"m={m}")
        ax.set_xlabel("n")
        ax.set_ylabel("Ops")
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=.2)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_2d.png"))
        plt.close(fig)

    if params is not None:
        for scale in ("linear", "log"):
            ops_grid = plot_data["ops_grid"].copy()
            ops_data = plot_data["ops_data_3d"].copy()
            if scale == "log":
                ops_grid = np.log10(np.where(ops_grid > 1e-9, ops_grid, 1e-9))
                ops_data = np.log10(np.where(ops_data > 1e-9, ops_data, 1e-9))

            fig = plt.figure(figsize=(16, 12), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                plot_data["m_grid"],
                plot_data["n_grid"],
                ops_grid,
                cmap="viridis",
                alpha=.7,
            )
            ax.scatter(
                plot_data["m_data_3d"],
                plot_data["n_data_3d"],
                ops_data,
                c="red",
                marker="o",
            )
            ax.set_xlabel("m")
            ax.set_ylabel("n")
            ax.set_zlabel("Ops")
            fig.colorbar(surf)
            fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_3d.png"))
            plt.close(fig)


def run_gmdh(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, float]:
    t_total_start = time.perf_counter()
    log.info("GMDH initialisation (raw data, no normalisation)")

    x_df, feature_names = generate_full_feature_matrix(df)
    x_raw = x_df.values
    y = df["ops"].values
    n_features = len(feature_names)

    log.info("Feature matrix: %d samples x %d features", x_raw.shape[0], n_features)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(y))
    split = len(y) // 2
    idx_1, idx_2 = indices[:split], indices[split:]

    x_tr1 = torch.tensor(x_raw[idx_1], dtype=DTYPE, device=DEVICE)
    y_tr1 = torch.tensor(y[idx_1], dtype=DTYPE, device=DEVICE).unsqueeze(1)
    x_tr2 = torch.tensor(x_raw[idx_2], dtype=DTYPE, device=DEVICE)
    y_tr2 = torch.tensor(y[idx_2], dtype=DTYPE, device=DEVICE).unsqueeze(1)

    x_te1_t = torch.tensor(x_raw[idx_2].T, dtype=DTYPE, device=DEVICE)
    y_te1 = torch.tensor(y[idx_2], dtype=DTYPE, device=DEVICE)
    x_te2_t = torch.tensor(x_raw[idx_1].T, dtype=DTYPE, device=DEVICE)
    y_te2 = torch.tensor(y[idx_1], dtype=DTYPE, device=DEVICE)

    log.info("Computing Gram matrices")
    xt_x_1 = x_tr1.T @ x_tr1
    xt_y_1 = x_tr1.T @ y_tr1
    xt_x_2 = x_tr2.T @ x_tr2
    xt_y_2 = x_tr2.T @ y_tr2

    del x_tr1, y_tr1, x_tr2, y_tr2
    torch.cuda.empty_cache()
    gc.collect()

    best_combined_mse = float("inf")
    best_indices = None
    mse_history: List[float] = list()

    metrics_k: List[int] = list()
    metrics_combs: List[int] = list()
    metrics_speed: List[float] = list()
    metrics_best_mse: List[float] = list()
    metrics_elapsed: List[float] = list()

    total_combinations = sum(math.comb(n_features, k) for k in range(1, n_features + 1))
    log.info("Total combinations to evaluate: %d", total_combinations)
    for k in range(1, n_features + 1):
        t_k_start = time.perf_counter()
        comb_gen = combinations(range(n_features), k)
        total_combs = math.comb(n_features, k)
        current_batch_size = BATCH_SIZE_INITIAL
        k_best_mse = float("inf")
        processed_count = 0

        log.info("k=%d: %d combinations", k, total_combs)

        with tqdm(total=total_combs, desc=f"k={k}", ncols=100) as pbar:
            async_loader = AsyncGPULoader(
                _batched(comb_gen, current_batch_size), DEVICE
            )

            while True:
                try:
                    try:
                        batch_cpu, batch_gpu = next(async_loader)
                    except StopIteration:
                        break

                    mse1, w1 = solve_batch_torch(
                        batch_gpu, xt_x_1, xt_y_1, x_te1_t, y_te1, .1
                    )
                    mse2, _ = solve_batch_torch(
                        batch_gpu, xt_x_2, xt_y_2, x_te2_t, y_te2, .1
                    )

                    total_mse = mse1 + mse2
                    min_idx = np.argmin(total_mse)
                    min_val = total_mse[min_idx]

                    if min_val < k_best_mse:
                        k_best_mse = float(min_val)

                    if min_val < best_combined_mse:
                        best_combined_mse = float(min_val)
                        best_indices = batch_cpu[min_idx]

                        best_w = w1[min_idx].flatten().cpu().numpy()
                        formula = " + ".join(
                            f"{w:+.4e}*{feature_names[i]}"
                            for w, i in zip(best_w, batch_cpu[min_idx])
                        )
                        pbar.set_postfix({"BestMSE": f"{best_combined_mse:.2e}"})
                        log.debug("New best: %s", formula)

                    mse_history.append(float(min_val))
                    processed_count += len(batch_cpu)
                    pbar.update(len(batch_cpu))

                    del mse1, mse2, total_mse, w1

                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    current_batch_size //= 2
                    if current_batch_size < 1:
                        raise RuntimeError("Batch size reduced to zero")
                    log.warning("OOM: batch size reduced to %d", current_batch_size)
                    raise RuntimeError("OOM requires manual restart")

        elapsed_k = time.perf_counter() - t_k_start
        speed = processed_count / elapsed_k if elapsed_k > 0 else .0

        metrics_k.append(k)
        metrics_combs.append(total_combs)
        metrics_speed.append(speed)
        metrics_best_mse.append(k_best_mse)
        metrics_elapsed.append(elapsed_k)

        log.info(
            "k=%d complete: %.2f it/s, best MSE=%.6e, total time=%.3f s",
            k,
            speed,
            k_best_mse,
            elapsed_k,
        )
        torch.cuda.empty_cache()
        gc.collect()

    selected_features = [feature_names[i] for i in best_indices]
    log.info("Global best structure: %s", selected_features)

    plot_metrics(
        metrics_k, metrics_combs, metrics_speed, metrics_best_mse, metrics_elapsed
    )
    plot_loss_history(mse_history)

    x_sub_full = x_raw[:, best_indices]
    final_model = Ridge(alpha=.1, fit_intercept=True)
    final_model.fit(x_sub_full, y)

    best_raw_names = [feature_names[i] for i in best_indices]
    final_coefs, final_names = perform_coefficient_clustering(
        final_model.coef_, best_raw_names, x_sub_full
    )

    final_indices_full = [feature_names.index(name) for name in final_names]

    model_binary = {
        "features": final_names,
        "coefficients": final_coefs,
        "intercept": final_model.intercept_,
        "mse": best_combined_mse,
    }
    joblib.dump(model_binary, os.path.join(PLOTS_DIR, "final_model.bin"))
    log.info("Model binary saved")

    def _predict_wrapper(coords, *_args):
        m_in, n_in = coords
        temp_df = pd.DataFrame({"m": m_in.flatten(), "n": n_in.flatten()})
        x_full, _ = generate_full_feature_matrix(temp_df)
        x_sub = x_full.values[:, final_indices_full]
        return np.dot(x_sub, final_coefs).reshape(m_in.shape)

    plot_data = extract_plot_data(df, _predict_wrapper, [None])
    create_plots(plot_data, "GMDH_Refined", params=True, plot_dir=PLOTS_DIR)

    t_total = time.perf_counter() - t_total_start
    log.info("Total GMDH execution time: %.3f s", t_total)

    return final_names, final_coefs, best_combined_mse


def main() -> None:
    t0 = time.perf_counter()
    log.info("Execution started")

    df = load_data()
    if df.empty:
        log.error("No data found in cache directory '%s'", CACHE_DIR)
        return

    log.info("Loaded %d data points", len(df))

    try:
        names, coefs, mse = run_gmdh(df)

        terms = [f"{w:+.16e} * [{n}]" for w, n in zip(coefs, names)]
        equation = "y = " + "\n    ".join(terms)

        log.info("Final model:\n%s", equation)
        log.info("Final MSE: %.6e", mse)

        with open(
                os.path.join(PLOTS_DIR, "final_formula.txt"), "w", encoding="utf-8"
        ) as fh:
            fh.write(equation)

        log.info("Formula written to %s/final_formula.txt", PLOTS_DIR)

    except (Exception,):
        log.exception("Critical execution error")

    elapsed = time.perf_counter() - t0
    log.info("Total wall-clock time: %.3f s", elapsed)


if __name__ == "__main__":
    main()
