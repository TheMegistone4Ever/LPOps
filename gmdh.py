import gc
import logging
import math
import os
import pickle
import sys
import time
import warnings
from itertools import combinations, islice
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

IS_BENCHMARKING = True
BENCHMARK_NOISE_PERCENT = .03

CACHE_DIR = "cache"

if IS_BENCHMARKING:
    GMDH_CACHE_DIR = "gmdh_cache_benchmark"
    PLOTS_DIR = "plots_combi_benchmark"
    LOG_FILE = "gmdh_execution_benchmark.log"
else:
    GMDH_CACHE_DIR = "gmdh_cache"
    PLOTS_DIR = "plots_combi"
    LOG_FILE = "gmdh_execution.log"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
BATCH_SIZE = 4096

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GMDH_CACHE_DIR, exist_ok=True)

if torch.cuda.is_available():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("gmdh")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = configure_logging()

if IS_BENCHMARKING:
    def _f_bench_m3(m, n):
        return m ** 3.


    def _f_bench_n2(m, n):
        return n ** 2.


    def _f_bench_lnm(m, n):
        return np.log(np.where(m > 1., m, 1.1))


    BASIS_FUNCTIONS = {
        "m3": _f_bench_m3,
        "n2": _f_bench_n2,
        "lnm": _f_bench_lnm,
    }
else:
    def _f_refined(m, n):
        return (m ** 3) * (n ** 2)


    def _f_smoothed(m, n):
        return m * (n ** 5) * np.log(np.where(n > 1, n, 1.1))


    def _f_poly_mn(m, n):
        return m * n


    def _safe_log(n):
        return np.where(n > 1, np.log(n), 0.)


    def _f_general(m, n):
        return .63 * m ** 2.96 * n ** .02 * _safe_log(n) ** 1.62 + 4.04 * m ** (-4.11) * n ** 2.92


    def _f_adler_megiddo(_, n):
        return n ** 4


    def _f_log_n_log_m(m, n):
        return np.log(np.where(m > 1, m, 1.1)) + np.log(np.where(n > 1, n, 1.1))


    BASIS_FUNCTIONS = {
        "m3n2": _f_refined,
        "mn5lnn": _f_smoothed,
        "poly_mn": _f_poly_mn,
        "general": _f_general,
        "adler_megiddo": _f_adler_megiddo,
        "log_n_log_m": _f_log_n_log_m,
    }


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    m = df["m"].values
    n = df["n"].values
    base = {name: func(m, n) for name, func in BASIS_FUNCTIONS.items()}
    base_names = list(base.keys())
    data = dict()
    names = list()
    for name in base_names:
        data[name] = base[name]
        names.append(name)
    for name in base_names:
        sq = f"({name})^2"
        data[sq] = base[name] ** 2
        names.append(sq)
    for a, b in combinations(base_names, 2):
        cross = f"{a} * {b}"
        data[cross] = base[a] * base[b]
        names.append(cross)
    return pd.DataFrame(data), names


def load_data() -> pd.DataFrame:
    if IS_BENCHMARKING:
        np.random.seed(1810)
        m_vals = np.arange(10, 110, 10).astype(float)
        n_vals = np.arange(10, 110, 10).astype(float)
        data = list()
        for m in m_vals:
            for n in n_vals:
                for _ in range(5):
                    f1 = m ** 3.
                    f2 = n ** 2.

                    y_ideal = 150. + (f1 * f2)

                    noise_std = BENCHMARK_NOISE_PERCENT * y_ideal
                    ops = y_ideal + np.random.normal(0, noise_std)
                    data.append([m, n, ops])
        df = pd.DataFrame(data, columns=["m", "n", "ops"])
        return df[df["ops"] > 0]

    if not os.path.exists(CACHE_DIR):
        return pd.DataFrame()
    all_data = list()
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
    for function_name in tqdm(files, desc="Завантаження кешу", ncols=100):
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
    df = pd.DataFrame(all_data, columns=["m", "n", "ops"]).drop_duplicates().astype(float)
    return df[df["ops"] > 0]


def _batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def _torch_solve(a, b):
    try:
        ch = torch.linalg.cholesky(a)
        return torch.cholesky_solve(b, ch)
    except RuntimeError:
        pass
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError:
        pass
    try:
        return torch.linalg.lstsq(a, b, rcond=1e-4).solution
    except RuntimeError:
        pass
    return torch.bmm(torch.linalg.pinv(a, rcond=1e-6), b)


def _torch_solve_single(a, b):
    try:
        ch = torch.linalg.cholesky(a)
        return torch.cholesky_solve(b, ch)
    except RuntimeError:
        pass
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError:
        pass
    return torch.linalg.lstsq(a, b, rcond=1e-4).solution


@torch.no_grad()
def solve_lsm_batch(idx_tensor, xt_x, xt_y, x_test_t, y_test, alpha):
    bs = idx_tensor.size(0)
    k = idx_tensor.size(1)
    rows = idx_tensor.unsqueeze(2).expand(bs, k, k)
    cols = idx_tensor.unsqueeze(1).expand(bs, k, k)
    a = xt_x[rows, cols]
    eye = torch.eye(k, device=DEVICE, dtype=DTYPE).unsqueeze(0).expand(bs, k, k)
    a.add_(eye, alpha=alpha)
    b = xt_y[idx_tensor]
    w = _torch_solve(a, b)
    x_sub = x_test_t[idx_tensor].permute(0, 2, 1)
    pred = torch.bmm(x_sub, w).squeeze(2)
    diff = pred - y_test.unsqueeze(0)
    sse = torch.sum(diff.pow(2), dim=1)
    return sse.cpu().numpy(), w.cpu()


@torch.no_grad()
def solve_lsm_single(indices, xt_x, xt_y, x_test_t, y_test, alpha):
    idx = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    k = len(indices)
    a = xt_x[idx][:, idx].clone()
    a.add_(torch.eye(k, device=DEVICE, dtype=DTYPE), alpha=alpha)
    b = xt_y[idx]
    w = _torch_solve_single(a, b)
    x_sub = x_test_t[idx].T
    pred = x_sub @ w
    diff = pred.squeeze() - y_test
    sse = torch.sum(diff.pow(2)).item()
    return sse, w.cpu().numpy().flatten()


@torch.no_grad()
def solve_and_cluster(x_data, y_data, feature_names, alpha, normalise: bool = True):
    n_feat = len(feature_names)
    n_samples = x_data.shape[0]

    if normalise:
        stds = np.std(x_data, axis=0)
        stds[stds < 1e-15] = 1e-15
    else:
        stds = np.ones(n_feat, dtype=float)
    x_norm = x_data / stds

    x_t = torch.tensor(x_norm, dtype=DTYPE, device=DEVICE)
    y_t = torch.tensor(y_data, dtype=DTYPE, device=DEVICE).unsqueeze(1)

    ones = torch.ones(n_samples, 1, dtype=DTYPE, device=DEVICE)
    x_aug = torch.cat([x_t, ones], dim=1)

    xt_x = x_aug.T @ x_aug
    xt_y = x_aug.T @ y_t

    eye = torch.eye(n_feat + 1, device=DEVICE, dtype=DTYPE)
    eye[-1, -1] = 0.

    a = xt_x.clone()
    a.add_(eye, alpha=alpha)

    w = _torch_solve_single(a, xt_y)

    w_np = w.cpu().numpy().flatten()

    std_coefs = w_np[:-1]

    raw_coefs = std_coefs / stds

    del x_t, y_t, ones, x_aug, xt_x, xt_y, a, w
    torch.cuda.empty_cache()

    abs_std = np.abs(std_coefs)
    order = np.argsort(-abs_std)

    log.info(f"Коефіцієнти (norm={normalise}, з урахуванням Bias):")
    for rank, i in enumerate(order):
        log.info("\t#%d %s: raw=%.4e, std_coef=%.4e, |std|=%.4e, std(x)=%.4e",
                 rank + 1, feature_names[i], raw_coefs[i], std_coefs[i], abs_std[i], stds[i])

    sorted_abs = abs_std[order]
    sorted_names = [feature_names[i] for i in order]

    if len(sorted_abs) <= 1:
        return raw_coefs, sorted_names, list()

    m1_names = [sorted_names[0]]
    split_pos = 1
    smallest = sorted_abs[-1]

    for i in range(1, len(sorted_abs)):
        avg_m1 = np.mean(sorted_abs[:i])
        dist_to_good = avg_m1 - sorted_abs[i]
        dist_to_bad = sorted_abs[i] - smallest
        if dist_to_good < dist_to_bad:
            m1_names.append(sorted_names[i])
            split_pos = i + 1
        else:
            break

    m2_names = sorted_names[split_pos:]
    return raw_coefs, m1_names, m2_names


def cache_path_k(k):
    return os.path.join(GMDH_CACHE_DIR, f"m2_k_{k:03d}.pkl")


def save_cache_k(k, result):
    with open(cache_path_k(k), "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache_k(k):
    path = cache_path_k(k)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError):
        return None


def cache_path_m1():
    return os.path.join(GMDH_CACHE_DIR, "m1_result.pkl")


def save_cache_m1(data):
    with open(cache_path_m1(), "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache_m1():
    path = cache_path_m1()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError):
        return None


def cache_path_final():
    return os.path.join(GMDH_CACHE_DIR, "final_result.pkl")


def save_cache_final(data):
    with open(cache_path_final(), "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache_final():
    path = cache_path_final()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError):
        return None


def _predict_model(m_arr, n_arr, indices, coefs, intercept):
    temp_df = pd.DataFrame({"m": m_arr.flatten(), "n": n_arr.flatten()})
    x_temp, _ = build_feature_matrix(temp_df)
    return (x_temp.values[:, indices] @ coefs + intercept).reshape(m_arr.shape)


def _plot_scatter_pred(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=.3, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2)
    ax.set_xlabel("Реальні значення")
    ax.set_ylabel("Прогноз")
    ax.set_title(title)
    ax.grid(True, alpha=.3)


def plot_models(df, m1_indices, best_indices, m1_coefs, full_coefs, m1_intercept, full_intercept):
    m_vals = sorted(df["m"].unique())
    n_vals = sorted(df["n"].unique())

    models = [
        ("Залишковий опис (M1 + зсув)", m1_indices, m1_coefs, m1_intercept, "залишковий"),
        ("Надлишковий опис (M1 + M2 + зсув)", best_indices, full_coefs, full_intercept, "надлишковий"),
    ]

    for model_title, m_idx, m_coefs, m_intercept, safe_tag in models:
        for scale in ("linear", "log"):
            scale_label = "лінійний" if scale == "linear" else "логарифмічний"
            fig, axes = plt.subplots(1, 2, figsize=(30, 20), dpi=200)

            ax = axes[0]
            for n_val in n_vals:
                subset = df[df["n"] == n_val]
                box_data = [subset[subset["m"] == mv]["ops"].values for mv in m_vals]
                bp = ax.boxplot(box_data, positions=m_vals, widths=.6, patch_artist=True, showfliers=False)
                for box in bp["boxes"]:
                    box.set(facecolor="lightblue")
                for median in bp["medians"]:
                    median.set(color="red", linewidth=2)
                avg_ops = subset.groupby("m")["ops"].mean()
                ax.plot(avg_ops.index, avg_ops.values, "k-", linewidth=1, alpha=.7)
                m_range = np.linspace(min(m_vals), max(m_vals), 100)
                fit_vals = _predict_model(m_range, np.full_like(m_range, n_val), m_idx, m_coefs, m_intercept)
                ax.plot(m_range, fit_vals, "--", label=f"n={int(n_val)}")
            ax.set_xlabel("m")
            ax.set_ylabel("Операції")
            ax.set_title(f"{model_title} — залежність від m")
            if scale == "log":
                ax.set_yscale("log")
            ax.grid(True, alpha=.2)
            ax.legend(fontsize=7)

            ax = axes[1]
            for m_val in m_vals:
                subset = df[df["m"] == m_val]
                box_data = [subset[subset["n"] == nv]["ops"].values for nv in n_vals]
                bp = ax.boxplot(box_data, positions=n_vals, widths=.6, patch_artist=True, showfliers=False)
                for box in bp["boxes"]:
                    box.set(facecolor="lightblue")
                for median in bp["medians"]:
                    median.set(color="red", linewidth=2)
                avg_ops = subset.groupby("n")["ops"].mean()
                ax.plot(avg_ops.index, avg_ops.values, "k-", linewidth=1, alpha=.7)
                n_range = np.linspace(min(n_vals), max(n_vals), 100)
                fit_vals = _predict_model(np.full_like(n_range, m_val), n_range, m_idx, m_coefs, m_intercept)
                ax.plot(n_range, fit_vals, "--", label=f"m={int(m_val)}")
            ax.set_xlabel("n")
            ax.set_ylabel("Операції")
            ax.set_title(f"{model_title} — залежність від n")
            if scale == "log":
                ax.set_yscale("log")
            ax.grid(True, alpha=.2)
            ax.legend(fontsize=7)

            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, f"{safe_tag}_{scale_label}_2d.png"))
            plt.close(fig)

        for scale in ("linear", "log"):
            scale_label = "лінійний" if scale == "linear" else "логарифмічний"
            m_grid, n_grid = np.meshgrid(
                np.linspace(min(m_vals), max(m_vals), 50),
                np.linspace(min(n_vals), max(n_vals), 50),
            )
            ops_grid = _predict_model(m_grid, n_grid, m_idx, m_coefs, m_intercept)

            df_grouped = df.groupby(["m", "n"])["ops"].mean().reset_index()
            m_data_3d = df_grouped["m"].values
            n_data_3d = df_grouped["n"].values
            ops_data_3d = df_grouped["ops"].values

            if scale == "log":
                ops_grid = np.log10(np.maximum(ops_grid, 1e-9))
                ops_data_3d = np.log10(np.maximum(ops_data_3d, 1e-9))

            fig = plt.figure(figsize=(16, 12), dpi=150)
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(m_grid, n_grid, ops_grid, cmap="viridis", alpha=.7)
            ax.scatter(m_data_3d, n_data_3d, ops_data_3d, c="red", marker="o", s=20)
            ax.set_xlabel("m")
            ax.set_ylabel("n")
            ax.set_zlabel("Операції")
            ax.set_title(model_title)
            fig.colorbar(surf)
            fig.savefig(os.path.join(PLOTS_DIR, f"{safe_tag}_{scale_label}_3d.png"))
            plt.close(fig)

    x_df, _ = build_feature_matrix(df)
    x_raw = x_df.values
    y = df["ops"].values
    pred_m1 = x_raw[:, m1_indices] @ m1_coefs + m1_intercept
    pred_full = x_raw[:, best_indices] @ full_coefs + full_intercept

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=200)
    _plot_scatter_pred(axes[0], y, pred_m1, "M1 + зсув (залишковий)")
    _plot_scatter_pred(axes[1], y, pred_full, "M1 + M2 + зсув (надлишковий)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "порівняння_прогноз_vs_реальність.png"))
    plt.close(fig)


@torch.no_grad()
def _fit_with_intercept(x_full, y_full, ones, indices, alpha):
    idx = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    x_sub = x_full[:, idx]
    x_aug = torch.cat([x_sub, ones], dim=1)
    xt_x = x_aug.T @ x_aug
    xt_x.add_(torch.eye(x_aug.shape[1], device=DEVICE, dtype=DTYPE), alpha=alpha)
    xt_y = x_aug.T @ y_full
    w = _torch_solve_single(xt_x, xt_y)
    w_np = w.cpu().numpy().flatten()
    coefs = w_np[:-1]
    intercept = w_np[-1]
    pred = (x_aug @ w).squeeze()
    diff = pred - y_full.squeeze()
    sse = torch.sum(diff.pow(2)).item()
    return coefs, intercept, sse


def run_gmdh(df: pd.DataFrame):
    t_start = time.perf_counter()
    log.info("Ініціалізація МГУА")

    x_df, feature_names = build_feature_matrix(df)
    x_raw = x_df.values
    y = df["ops"].values
    n_features = len(feature_names)
    alpha = .1

    log.info("Матриця ознак: %d зразків x %d ознак", x_raw.shape[0], n_features)
    log.info("Ознаки: %s", ", ".join(feature_names))
    rng = np.random.RandomState(1810)
    indices = rng.permutation(len(y))
    split = len(y) // 2
    idx_1, idx_2 = indices[:split], indices[split:]

    log.info("Розбиття: половина 1 = %d, половина 2 = %d", len(idx_1), len(idx_2))

    x1 = torch.tensor(x_raw[idx_1], dtype=DTYPE, device=DEVICE)
    y1 = torch.tensor(y[idx_1], dtype=DTYPE, device=DEVICE).unsqueeze(1)
    x2 = torch.tensor(x_raw[idx_2], dtype=DTYPE, device=DEVICE)
    y2 = torch.tensor(y[idx_2], dtype=DTYPE, device=DEVICE).unsqueeze(1)

    xt_x_1 = x1.T @ x1
    xt_y_1 = x1.T @ y1
    xt_x_2 = x2.T @ x2
    xt_y_2 = x2.T @ y2

    x_te1_t = torch.tensor(x_raw[idx_2].T, dtype=DTYPE, device=DEVICE)
    y_te1 = torch.tensor(y[idx_2], dtype=DTYPE, device=DEVICE)
    x_te2_t = torch.tensor(x_raw[idx_1].T, dtype=DTYPE, device=DEVICE)
    y_te2 = torch.tensor(y[idx_1], dtype=DTYPE, device=DEVICE)

    del x1, y1, x2, y2
    torch.cuda.empty_cache()
    gc.collect()

    cached_m1 = load_cache_m1()
    if cached_m1 is not None:
        m1_names = cached_m1["m1_names"]
        m2_names = cached_m1["m2_names"]
        m1_indices_list = cached_m1["m1_indices"]
        m2_indices_list = cached_m1["m2_indices"]
        sse_m1_only = cached_m1["sse_m1_only"]
        log.info("Кластери M1/M2 завантажено з кешу")
    else:
        log.info("Крок 1: стандартизована підгонка + кластеризація (половина 1)")
        raw_coefs_1, m1_1, m2_1 = solve_and_cluster(x_raw[idx_1], y[idx_1], feature_names, alpha)

        log.info("Крок 2: стандартизована підгонка + кластеризація (половина 2)")
        raw_coefs_2, m1_2, m2_2 = solve_and_cluster(x_raw[idx_2], y[idx_2], feature_names, alpha)

        log.info("Половина 1 -> M1_1: %s", m1_1)
        log.info("Половина 1 -> M2_1: %s", m2_1)
        log.info("Половина 2 -> M1_2: %s", m1_2)
        log.info("Половина 2 -> M2_2: %s", m2_2)

        m1_names = sorted(set(m1_1) & set(m1_2))
        all_features = set(feature_names)
        m2_names = sorted(all_features - set(m1_names))

        log.info("Результуючий M1 (перетин): %s", m1_names)
        log.info("Результуючий M2 (решта): %s", m2_names)

        if not m1_names:
            log.warning("M1 порожній! Беремо ознаку з найбільшим середнім стандартизованим коефіцієнтом")
            stds_all = np.std(x_raw, axis=0)
            stds_all[stds_all < 1e-15] = 1e-15
            avg_std_coef = (np.abs(raw_coefs_1) * stds_all + np.abs(raw_coefs_2) * stds_all) / 2
            best_idx = np.argmax(avg_std_coef)
            m1_names = [feature_names[best_idx]]
            m2_names = sorted(all_features - set(m1_names))
            log.info("Скоригований M1: %s", m1_names)

        m1_indices_list = [feature_names.index(n) for n in m1_names]
        m2_indices_list = [feature_names.index(n) for n in m2_names]

        sse_1, _ = solve_lsm_single(m1_indices_list, xt_x_1, xt_y_1, x_te1_t, y_te1, alpha)
        sse_2, _ = solve_lsm_single(m1_indices_list, xt_x_2, xt_y_2, x_te2_t, y_te2, alpha)
        sse_m1_only = sse_1 + sse_2

        save_cache_m1({
            "m1_names": m1_names,
            "m2_names": m2_names,
            "m1_indices": m1_indices_list,
            "m2_indices": m2_indices_list,
            "sse_m1_only": sse_m1_only,
        })

    log.info("M1 (%d ознак): %s", len(m1_indices_list), m1_names)
    log.info("M2 (%d ознак): %s", len(m2_indices_list), m2_names)
    log.info("SSE тільки M1: %.6e", sse_m1_only)

    best_sse = sse_m1_only
    best_m2_combo = list()
    best_m2_combo_names = list()
    n_m2 = len(m2_indices_list)
    total_combos = sum(math.comb(n_m2, k) for k in range(1, n_m2 + 1))

    log.info("Загальна кількість комбінацій M2: %d", total_combos)

    for k in range(1, n_m2 + 1):
        n_combs_k = math.comb(n_m2, k)
        log.info("k=%d з M2: %d комбінацій", k, n_combs_k)

        cached_k = load_cache_k(k)
        if cached_k is not None:
            log.info("k=%d: завантажено з кешу", k)
            if cached_k["best_sse"] < best_sse:
                best_sse = cached_k["best_sse"]
                best_m2_combo = cached_k["best_m2_combo"]
                best_m2_combo_names = cached_k["best_m2_combo_names"]
                log.info("k=%d: краще SSE з кешу = %.6e, M2: %s", k, best_sse, best_m2_combo_names)
            continue

        t_k = time.perf_counter()
        k_best_sse = float("inf")
        k_best_combo = list()
        k_best_names = list()
        processed = 0
        batch_size = BATCH_SIZE

        with tqdm(total=n_combs_k, desc=f"k={k}/{n_m2}", ncols=100) as pbar:
            while True:
                combo_gen = combinations(range(n_m2), k)
                if processed > 0:
                    combo_gen = islice(combo_gen, processed, None)

                oom = False
                for batch_cpu in _batched(combo_gen, batch_size):
                    full_indices = list()
                    for combo in batch_cpu:
                        feat_idx = m1_indices_list + [m2_indices_list[j] for j in combo]
                        full_indices.append(feat_idx)

                    try:
                        idx_tensor = torch.tensor(full_indices, dtype=torch.long, device=DEVICE)

                        sse1, w1 = solve_lsm_batch(idx_tensor, xt_x_1, xt_y_1, x_te1_t, y_te1, alpha)
                        sse2, _ = solve_lsm_batch(idx_tensor, xt_x_2, xt_y_2, x_te2_t, y_te2, alpha)

                        total_sse = sse1 + sse2
                        min_idx = np.argmin(total_sse)
                        min_val = total_sse[min_idx]

                        if min_val < k_best_sse:
                            k_best_sse = float(min_val)
                            k_best_combo = list(batch_cpu[min_idx])
                            k_best_names = [m2_names[j] for j in k_best_combo]

                        if min_val < best_sse:
                            best_sse = float(min_val)
                            best_m2_combo = list(batch_cpu[min_idx])
                            best_m2_combo_names = [m2_names[j] for j in best_m2_combo]
                            pbar.set_postfix({"SSE": f"{best_sse:.2e}"})

                        processed += len(batch_cpu)
                        pbar.update(len(batch_cpu))

                        del idx_tensor, sse1, sse2, w1

                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                        if not is_oom:
                            raise
                        torch.cuda.empty_cache()
                        gc.collect()
                        old_bs = batch_size
                        batch_size = max(1, batch_size // 2)
                        log.warning("OOM при k=%d: батч %d -> %d (оброблено %d/%d)", k, old_bs, batch_size, processed,
                                    n_combs_k)
                        oom = True
                        break

                if not oom:
                    break

        elapsed_k = time.perf_counter() - t_k
        log.info("k=%d завершено за %.2f с, найкраще SSE=%.6e", k, elapsed_k, k_best_sse)

        save_cache_k(k, {
            "k": k,
            "best_sse": k_best_sse,
            "best_m2_combo": k_best_combo,
            "best_m2_combo_names": k_best_names,
            "elapsed": elapsed_k,
            "n_combs": n_combs_k,
        })

        torch.cuda.empty_cache()
        gc.collect()

    log.info("=" * 60)
    log.info("Найкраще SSE (крос-валідація): %.6e", best_sse)
    log.info("M1: %s", m1_names)
    log.info("Найкращі з M2: %s", best_m2_combo_names)

    best_all_names = m1_names + best_m2_combo_names
    best_all_indices = m1_indices_list + [m2_indices_list[j] for j in best_m2_combo]

    log.info("=" * 60)
    log.info("Фінальна підгонка на ВСІХ даних")

    cached_final = load_cache_final()
    if cached_final is not None and set(cached_final.get("best_all_names", list())) == set(best_all_names):
        final_coefs = cached_final["final_coefs"]
        final_intercept = cached_final["final_intercept"]
        final_sse = cached_final["final_sse"]
        m1_coefs = cached_final["m1_coefs"]
        m1_intercept = cached_final["m1_intercept"]
        log.info("Фінальний результат завантажено з кешу")
    else:
        x_full = torch.tensor(x_raw, dtype=DTYPE, device=DEVICE)
        y_full = torch.tensor(y, dtype=DTYPE, device=DEVICE).unsqueeze(1)
        ones = torch.ones(len(y), 1, dtype=DTYPE, device=DEVICE)

        final_coefs, final_intercept, final_sse = _fit_with_intercept(x_full, y_full, ones, best_all_indices, alpha)
        m1_coefs, m1_intercept, _ = _fit_with_intercept(x_full, y_full, ones, m1_indices_list, alpha)

        del x_full, y_full, ones
        torch.cuda.empty_cache()

        save_cache_final({
            "best_all_names": best_all_names,
            "best_all_indices": best_all_indices,
            "final_coefs": final_coefs,
            "final_intercept": final_intercept,
            "final_sse": final_sse,
            "m1_coefs": m1_coefs,
            "m1_intercept": m1_intercept,
            "m1_names": m1_names,
            "m1_indices": m1_indices_list,
        })

    log.info("SSE на всіх даних: %.6e", final_sse)

    log.info("=" * 60)
    log.info("НАЙКРАЩА ФОРМУЛА:")
    log.info("y = M1 + M2 + зсув")
    log.info("")
    log.info("M1 (залишковий опис):")
    for c, n in zip(m1_coefs, m1_names):
        log.info("  %+.10e * %s", c, n)
    log.info("")
    log.info("M2 (додаткові терми):")
    m2_selected_coefs = final_coefs[len(m1_indices_list):]
    for c, n in zip(m2_selected_coefs, best_m2_combo_names):
        log.info("  %+.10e * %s", c, n)
    log.info("")
    log.info("Зсув (bias): %+.10e", final_intercept)
    log.info("=" * 60)

    formula_lines = ["y = M1 + M2 + зсув", "", "M1 (залишковий опис):"]
    for c, n in zip(m1_coefs, m1_names):
        formula_lines.append(f"  {c:+.16e} * {n}")
    formula_lines.append("")
    formula_lines.append("M2 (додаткові терми):")
    for c, n in zip(m2_selected_coefs, best_m2_combo_names):
        formula_lines.append(f"  {c:+.16e} * {n}")
    formula_lines.append("")
    formula_lines.append(f"Зсув (bias): {final_intercept:+.16e}")
    formula_lines.append("")
    formula_lines.append(f"SSE (крос-валідація): {best_sse:.6e}")
    formula_lines.append(f"SSE (всі дані): {final_sse:.6e}")

    with open(os.path.join(PLOTS_DIR, "фінальна_формула.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(formula_lines))

    model_bin = {
        "m1_names": m1_names,
        "m1_indices": m1_indices_list,
        "m1_coefs": m1_coefs,
        "m1_intercept": m1_intercept,
        "m2_names": best_m2_combo_names,
        "m2_indices": [m2_indices_list[j] for j in best_m2_combo],
        "m2_coefs": m2_selected_coefs,
        "all_names": best_all_names,
        "all_indices": best_all_indices,
        "all_coefs": final_coefs,
        "intercept": final_intercept,
        "sse_cv": best_sse,
        "sse_full": final_sse,
        "feature_names": feature_names,
    }
    joblib.dump(model_bin, os.path.join(PLOTS_DIR, "фінальна_модель.bin"))
    log.info("Модель збережено у %s/фінальна_модель.bin", PLOTS_DIR)

    log.info("Побудова графіків...")
    plot_models(
        df,
        m1_indices_list, best_all_indices,
        m1_coefs, final_coefs,
        m1_intercept, final_intercept,
    )
    log.info("Графіки збережено у %s/", PLOTS_DIR)

    t_total = time.perf_counter() - t_start
    log.info("Загальний час МГУА: %.3f с", t_total)

    return best_all_names, final_coefs, final_intercept, final_sse


def main():
    t0 = time.perf_counter()
    log.info("Початок виконання")

    df = load_data()
    if df.empty:
        log.error("Дані не знайдено у директорії '%s'", CACHE_DIR)
        return

    log.info("Завантажено %d точок даних", len(df))

    try:
        run_gmdh(df)
        log.info("Виконання завершено успішно")
    except Exception as e:
        log.exception(f"Критична помилка виконання: {e}")

    elapsed = time.perf_counter() - t0
    log.info("Загальний час: %.3f с", elapsed)


if __name__ == "__main__":
    main()
