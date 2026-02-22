import logging
import math
import os
import pickle
import re
import sys
import time
import warnings
from itertools import combinations
from typing import Any, List, Optional, Tuple, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm

warnings.filterwarnings("ignore")

CACHE_DIR = "cache"
GMDH_CACHE_DIR = "gmdh_cache"
PLOTS_DIR = "plots_combi"
LOG_FILE = "gmdh_execution.log"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GMDH_CACHE_DIR, exist_ok=True)


# ═══════════════════════════ LOGGING ═══════════════════════════

def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("gmdh")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = _configure_logging()


# ═══════════════════════ CACHE HELPERS ═════════════════════════

def _cache_path(name: str) -> str:
    return os.path.join(GMDH_CACHE_DIR, f"{name}.pkl")


def _save_cache(name: str, data: Any) -> None:
    path = _cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.debug("Cache saved: %s", path)


def _load_cache(name: str) -> Optional[Any]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        log.debug("Cache loaded: %s", path)
        return data
    except (pickle.UnpicklingError, EOFError, Exception) as e:
        log.warning("Cache corrupted %s: %s", path, e)
        return None


def _cache_exists(name: str) -> bool:
    return os.path.exists(_cache_path(name))


# ═══════════════════ BASIS FUNCTIONS ═══════════════════════════

def _f_refined(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return (m ** 3) * (n ** 2)


def _f_smoothed(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return m * (n ** 5) * np.log(np.where(n > 1, n, 1.1))


def _f_poly_mn(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return m * n


def _safe_log(n: np.ndarray) -> np.ndarray:
    return np.where(n > 1, np.log(n), 0.0)


def _f_general(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return (
            0.63 * m ** 2.96 * n ** 0.02 * _safe_log(n) ** 1.62
            + 4.04 * m ** (-4.11) * n ** 2.92
    )


def _f_adler_megiddo(_: np.ndarray, n: np.ndarray) -> np.ndarray:
    return n ** 4


def _f_log_n_log_m(m: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.log(np.where(m > 1, m, 1.1)) + np.log(np.where(n > 1, n, 1.1))


BASIS_FUNCTIONS = {
    "m3n2": _f_refined,
    "mn5lnn": _f_smoothed,
    "poly_mn": _f_poly_mn,
    "general": _f_general,
    "adler_megiddo": _f_adler_megiddo,
    "log_n_log_m": _f_log_n_log_m,
}


# ═══════════════ FEATURE MATRIX GENERATION ═════════════════════

def generate_full_feature_matrix(
        df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    m = df["m"].values
    n = df["n"].values

    base_data = {name: func(m, n) for name, func in BASIS_FUNCTIONS.items()}
    base_names = list(base_data.keys())

    final_data = {}
    feature_names = []

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


# ═══════════════════ DATA LOADING ══════════════════════════════

def load_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_DIR):
        return pd.DataFrame()

    all_data: List[List] = []
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]

    for fname in tqdm(files, desc="Loading cache", ncols=100):
        try:
            with open(os.path.join(CACHE_DIR, fname), "rb") as fh:
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


# ═══════════════ COEFFICIENT CLUSTERING ════════════════════════

def perform_coefficient_clustering(
        coefs: np.ndarray,
        feature_names: List[str],
        x_data: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """
    Сортуємо за importance = |coef| * std(feature).
    Кластеризація: M1 (важливі) та M2 (решта).
    Повертає (m1_names, m2_names).
    """
    feature_stds = np.std(x_data, axis=0)
    items = []
    # for w, std, name in zip(coefs, feature_stds, feature_names):
    for w, std, name in tqdm(
            zip(coefs, feature_stds, feature_names),
            desc="Clustering coefficients",
            ncols=100,
    ):
        importance = abs(w) * (std if std > 1e-9 else 1.0)
        items.append({"name": name, "importance": importance})

    sorted_items = sorted(items, key=lambda x: x["importance"], reverse=True)

    if not sorted_items:
        return [], list(feature_names)

    smallest_imp = sorted_items[-1]["importance"]
    m1_items = [sorted_items[0]]

    # for i in range(1, len(sorted_items)):
    for i in tqdm(
            range(1, len(sorted_items)),
            desc="Determining M1/M2 split",
            ncols=100,
    ):
        candidate = sorted_items[i]
        avg_m1 = sum(it["importance"] for it in m1_items) / len(m1_items)
        dist_good = avg_m1 - candidate["importance"]
        dist_bad = candidate["importance"] - smallest_imp
        if dist_good < dist_bad:
            m1_items.append(candidate)
        else:
            break

    m1_names = [it["name"] for it in m1_items]
    m2_names = [it["name"] for it in sorted_items if it["name"] not in m1_names]

    return m1_names, m2_names


# ═══════════════════ LSM / SSE ═════════════════════════════════

def fit_lsm(
        x_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.1
) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, y_train)
    return model


def compute_sse(model: Ridge, x_test: np.ndarray, y_test: np.ndarray) -> float:
    y_pred = model.predict(x_test)
    return float(np.sum((y_test - y_pred) ** 2))


# ═══════════════════ SCIENTIFIC PLOTS ══════════════════════════

def plot_model_vs_data(
        df: pd.DataFrame,
        predict_func: Callable,
        plot_dir: str,
        model_name: str = "GMDH",
        predict_m1_only: Callable = None,
        predict_m1_m2_all: Callable = None,
) -> None:
    """
    Наукові графіки:
    - 2D slices з boxplot + три моделі (M1+bias, M1+M2+bias, найкраща)
    - Графік залишків
    - 3D поверхні для всіх трьох моделей
    """
    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    m_vals = sorted(df["m"].unique())
    n_vals = sorted(df["n"].unique())

    y_actual = df["ops"].values
    y_predicted = predict_func(df["m"].values, df["n"].values)

    # ═══════════════════════════════════════════════════════════
    # 2D slices: boxplot + model curves
    # ═══════════════════════════════════════════════════════════
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(30, 20), dpi=200)

        # ── Left: ops vs m, fixed n ──
        ax = axes[0]
        for n_fixed in n_vals:
            subset = df[df["n"] == n_fixed]
            box_data = [
                subset[subset["m"] == m]["ops"].values for m in m_vals
            ]
            bp = ax.boxplot(
                box_data,
                positions=m_vals,
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            for box in bp["boxes"]:
                box.set(facecolor="lightblue")
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)

            # Average line
            avg_ops = subset.groupby("m")["ops"].mean()
            ax.plot(
                avg_ops.index, avg_ops.values,
                "k-", linewidth=1, alpha=0.7,
            )

            m_range = np.linspace(min(m_vals), max(m_vals), 100)

            # M1 + bias (залишковий)
            if predict_m1_only is not None:
                y_m1 = predict_m1_only(m_range, np.full_like(m_range, n_fixed))
                ax.plot(
                    m_range, y_m1, ":",
                    color="orange", linewidth=2, alpha=0.7,
                    label=f"M1+bias (n={n_fixed})" if n_fixed == n_vals[0] else "",
                )

            # M1 + ALL M2 + bias (надлишковий)
            if predict_m1_m2_all is not None:
                y_all = predict_m1_m2_all(m_range, np.full_like(m_range, n_fixed))
                ax.plot(
                    m_range, y_all, "-.",
                    color="purple", linewidth=2, alpha=0.7,
                    label=f"M1+M2+bias (n={n_fixed})" if n_fixed == n_vals[0] else "",
                )

            # Best formula (M1 + selected M2 + bias)
            y_best = predict_func(m_range, np.full_like(m_range, n_fixed))
            ax.plot(
                m_range, y_best, "--",
                color="green", linewidth=2.5,
                label=f"Найкраща (n={n_fixed})" if n_fixed == n_vals[0] else "",
            )

        ax.set_xlabel("m (кількість обмежень)", fontsize=12)
        ax.set_ylabel("Операції", fontsize=12)
        ax.set_title("Залежність операцій від m", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)

        # Legend with model descriptions
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="orange", ls=":", lw=2, label="M1 + bias (залишковий)"),
            Line2D([0], [0], color="purple", ls="-.", lw=2, label="M1 + M2(all) + bias (надлишковий)"),
            Line2D([0], [0], color="green", ls="--", lw=2.5, label="Найкраща формула (МГУА)"),
            Line2D([0], [0], color="black", ls="-", lw=1, label="Середнє значення даних"),
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

        # ── Right: ops vs n, fixed m ──
        ax = axes[1]
        for m_fixed in m_vals:
            subset = df[df["m"] == m_fixed]
            box_data = [
                subset[subset["n"] == n]["ops"].values for n in n_vals
            ]
            bp = ax.boxplot(
                box_data,
                positions=n_vals,
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            for box in bp["boxes"]:
                box.set(facecolor="lightblue")
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)

            avg_ops = subset.groupby("n")["ops"].mean()
            ax.plot(
                avg_ops.index, avg_ops.values,
                "k-", linewidth=1, alpha=0.7,
            )

            n_range = np.linspace(min(n_vals), max(n_vals), 100)

            # M1 + bias
            if predict_m1_only is not None:
                y_m1 = predict_m1_only(np.full_like(n_range, m_fixed), n_range)
                ax.plot(
                    n_range, y_m1, ":",
                    color="orange", linewidth=2, alpha=0.7,
                )

            # M1 + ALL M2 + bias
            if predict_m1_m2_all is not None:
                y_all = predict_m1_m2_all(np.full_like(n_range, m_fixed), n_range)
                ax.plot(
                    n_range, y_all, "-.",
                    color="purple", linewidth=2, alpha=0.7,
                )

            # Best formula
            y_best = predict_func(np.full_like(n_range, m_fixed), n_range)
            ax.plot(
                n_range, y_best, "--",
                color="green", linewidth=2.5,
            )

        ax.set_xlabel("n (кількість змінних)", fontsize=12)
        ax.set_ylabel("Операції", fontsize=12)
        ax.set_title("Залежність операцій від n", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

        fig.suptitle(
            f"Модель {safe_name}: дані vs апроксимація ({scale})", fontsize=14
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_2d.png"))
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Residuals (для найкращої формули)
    # ═══════════════════════════════════════════════════════════
    residuals = y_actual - y_predicted

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

    ax = axes[0]
    ax.scatter(y_predicted, residuals, s=8, alpha=0.4, c="steelblue")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Передбачені значення", fontsize=12)
    ax.set_ylabel("Залишки (actual − predicted)", fontsize=12)
    ax.set_title("Залишки vs передбачені значення", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(y_actual, y_predicted, s=8, alpha=0.4, c="darkgreen")
    lim_min = min(y_actual.min(), y_predicted.min())
    lim_max = max(y_actual.max(), y_predicted.max())
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "r--", lw=1.5, label="Ідеал (y=x)",
    )
    ax.set_xlabel("Реальні значення", fontsize=12)
    ax.set_ylabel("Передбачені значення", fontsize=12)
    ax.set_title("Реальні vs передбачені", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Модель {safe_name}: аналіз залишків", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{safe_name}_residuals.png"))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # 3D surfaces: all three models
    # ═══════════════════════════════════════════════════════════
    df_grouped = df.groupby(["m", "n"])["ops"].mean().reset_index()

    for scale in ("linear", "log"):
        m_grid, n_grid = np.meshgrid(
            np.linspace(min(m_vals), max(m_vals), 50),
            np.linspace(min(n_vals), max(n_vals), 50),
        )

        # Compute surfaces
        ops_best = predict_func(
            m_grid.ravel(), n_grid.ravel()
        ).reshape(m_grid.shape)

        ops_m1 = None
        if predict_m1_only is not None:
            ops_m1 = predict_m1_only(
                m_grid.ravel(), n_grid.ravel()
            ).reshape(m_grid.shape)

        ops_all = None
        if predict_m1_m2_all is not None:
            ops_all = predict_m1_m2_all(
                m_grid.ravel(), n_grid.ravel()
            ).reshape(m_grid.shape)

        ops_data = df_grouped["ops"].values.copy()

        if scale == "log":
            ops_best = np.log10(np.clip(ops_best, 1e-9, None))
            if ops_m1 is not None:
                ops_m1 = np.log10(np.clip(ops_m1, 1e-9, None))
            if ops_all is not None:
                ops_all = np.log10(np.clip(ops_all, 1e-9, None))
            ops_data = np.log10(np.clip(ops_data, 1e-9, None))

        fig = plt.figure(figsize=(20, 14), dpi=150)

        # --- Subplot 1: M1 + bias (залишковий) ---
        if ops_m1 is not None:
            ax1 = fig.add_subplot(131, projection="3d")
            ax1.plot_surface(
                m_grid, n_grid, ops_m1,
                cmap="Oranges", alpha=0.7,
            )
            ax1.scatter(
                df_grouped["m"].values,
                df_grouped["n"].values,
                ops_data,
                c="red", marker="o", s=15,
            )
            ax1.set_xlabel("m", fontsize=10)
            ax1.set_ylabel("n", fontsize=10)
            z_label = "log₁₀(Ops)" if scale == "log" else "Ops"
            ax1.set_zlabel(z_label, fontsize=10)
            ax1.set_title("M1 + bias\n(залишковий)", fontsize=11)

        # --- Subplot 2: M1 + ALL M2 + bias (надлишковий) ---
        if ops_all is not None:
            ax2 = fig.add_subplot(132, projection="3d")
            ax2.plot_surface(
                m_grid, n_grid, ops_all,
                cmap="Purples", alpha=0.7,
            )
            ax2.scatter(
                df_grouped["m"].values,
                df_grouped["n"].values,
                ops_data,
                c="red", marker="o", s=15,
            )
            ax2.set_xlabel("m", fontsize=10)
            ax2.set_ylabel("n", fontsize=10)
            ax2.set_zlabel(z_label, fontsize=10)
            ax2.set_title("M1 + M2(all) + bias\n(надлишковий)", fontsize=11)

        # --- Subplot 3: Best formula ---
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.plot_surface(
            m_grid, n_grid, ops_best,
            cmap="viridis", alpha=0.7,
        )
        ax3.scatter(
            df_grouped["m"].values,
            df_grouped["n"].values,
            ops_data,
            c="red", marker="o", s=15,
        )
        ax3.set_xlabel("m", fontsize=10)
        ax3.set_ylabel("n", fontsize=10)
        ax3.set_zlabel(z_label, fontsize=10)
        ax3.set_title("Найкраща формула\n(МГУА)", fontsize=11)

        fig.suptitle(
            f"Порівняння моделей ({scale}): залишковий → надлишковий → оптимальний",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_3d.png"))
        plt.close(fig)


# ═══════════════════ MAIN GMDH ALGORITHM ═══════════════════════

def run_gmdh(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, float, float]:
    """
    Алгоритм МГУА:

    1. Дані → дві половини
    2. Для кожної половини: повний опис → кластеризація → M1_i, M2_i
    3. M1 = M1_1 ∩ M1_2;  M2 = все інше
    4. Часткові описи: M1 + комбінації M2
    5. SSE = SSE_1 + SSE_2 (cross-validation)
    6. Найкраща формула = argmin SSE
    7. Фінальна модель тренується НА ВСІХ ДАНИХ, SSE рахується на всіх даних

    Повертає (feature_names, coefficients, intercept, sse_full).
    """
    t_total_start = time.perf_counter()
    log.info("=" * 70)
    log.info("GMDH ALGORITHM START")
    log.info("=" * 70)

    # ── Step 0: Feature matrix ──
    x_df, feature_names = generate_full_feature_matrix(df)
    x_raw = x_df.values
    y = df["ops"].values
    n_features = len(feature_names)
    n_samples = len(y)

    log.info(
        "Feature matrix: %d samples × %d features", n_samples, n_features
    )
    log.info("Features: %s", ", ".join(feature_names))

    # ── Step 1: Split data ──
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    split = n_samples // 2
    idx_1, idx_2 = indices[:split], indices[split:]

    x_half1, y_half1 = x_raw[idx_1], y[idx_1]
    x_half2, y_half2 = x_raw[idx_2], y[idx_2]

    log.info("Split: half1=%d, half2=%d samples", len(idx_1), len(idx_2))

    # ══════════════════════════════════════════════════════════════
    # Step 2a: Clustering on HALF-1 (cached)
    # ══════════════════════════════════════════════════════════════
    cache_cluster_1 = _load_cache("step2a_clustering_half1")

    if cache_cluster_1 is not None:
        m1_1_names = cache_cluster_1["m1_names"]
        m2_1_names = cache_cluster_1["m2_names"]
        coefs_1 = cache_cluster_1["coefs"]
        log.info("Step 2a: LOADED FROM CACHE")
    else:
        log.info("─" * 50)
        log.info("Step 2a: Full description on HALF-1, clustering")

        model_full_1 = fit_lsm(x_half1, y_half1)
        coefs_1 = model_full_1.coef_

        log.info("  Full model 1 coefficients:")
        for name, c in zip(feature_names, coefs_1):
            log.info("    %-25s : %+.6e", name, c)

        m1_1_names, m2_1_names = perform_coefficient_clustering(
            coefs_1, feature_names, x_half1
        )

        _save_cache("step2a_clustering_half1", {
            "m1_names": m1_1_names,
            "m2_names": m2_1_names,
            "coefs": coefs_1,
        })

    log.info("  M1_1 (important, half1): %s", m1_1_names)
    log.info("  M2_1 (rest, half1):      %s", m2_1_names)

    # ══════════════════════════════════════════════════════════════
    # Step 2b: Clustering on HALF-2 (cached)
    # ══════════════════════════════════════════════════════════════
    cache_cluster_2 = _load_cache("step2b_clustering_half2")

    if cache_cluster_2 is not None:
        m1_2_names = cache_cluster_2["m1_names"]
        m2_2_names = cache_cluster_2["m2_names"]
        coefs_2 = cache_cluster_2["coefs"]
        log.info("Step 2b: LOADED FROM CACHE")
    else:
        log.info("─" * 50)
        log.info("Step 2b: Full description on HALF-2, clustering")

        model_full_2 = fit_lsm(x_half2, y_half2)
        coefs_2 = model_full_2.coef_

        log.info("  Full model 2 coefficients:")
        for name, c in zip(feature_names, coefs_2):
            log.info("    %-25s : %+.6e", name, c)

        m1_2_names, m2_2_names = perform_coefficient_clustering(
            coefs_2, feature_names, x_half2
        )

        _save_cache("step2b_clustering_half2", {
            "m1_names": m1_2_names,
            "m2_names": m2_2_names,
            "coefs": coefs_2,
        })

    log.info("  M1_2 (important, half2): %s", m1_2_names)
    log.info("  M2_2 (rest, half2):      %s", m2_2_names)

    # ══════════════════════════════════════════════════════════════
    # Step 3: Intersection / Union (cached)
    # ══════════════════════════════════════════════════════════════
    cache_intersection = _load_cache("step3_intersection")

    if cache_intersection is not None:
        m1_final = cache_intersection["m1_final"]
        m2_final = cache_intersection["m2_final"]
        log.info("Step 3: LOADED FROM CACHE")
    else:
        log.info("─" * 50)
        log.info("Step 3: M1 = intersection, M2 = everything else")

        m1_final = sorted(set(m1_1_names) & set(m1_2_names))
        m2_final = sorted(set(feature_names) - set(m1_final))

        _save_cache("step3_intersection", {
            "m1_final": m1_final,
            "m2_final": m2_final,
        })

    log.info("  M1 (intersection): %d features: %s", len(m1_final), m1_final)
    log.info("  M2 (everything else): %d features: %s", len(m2_final), m2_final)

    if not m1_final:
        log.warning("  M1 is empty! All features go to M2.")

    # ── Feature indices ──
    m1_indices = [feature_names.index(name) for name in m1_final]
    m2_indices = [feature_names.index(name) for name in m2_final]

    # ══════════════════════════════════════════════════════════════
    # Step 4–5: Evaluate partial descriptions (each combo cached)
    # ══════════════════════════════════════════════════════════════
    total_m2_combos = sum(
        math.comb(len(m2_final), r) for r in range(0, len(m2_final) + 1)
    )
    log.info("─" * 50)
    log.info(
        "Step 4-5: Evaluating %d partial descriptions "
        "(M1[%d] + combos of M2[%d])",
        total_m2_combos, len(m1_final), len(m2_final),
    )

    # Try to load completed combo results
    cache_all_combos = _load_cache("step5_all_combo_results")

    if cache_all_combos is not None:
        sse_values_list = cache_all_combos["sse_values"]
        combo_labels_list = cache_all_combos["combo_labels"]
        combo_indices_list = cache_all_combos["combo_indices"]
        best_combo_idx = cache_all_combos["best_combo_idx"]
        best_sse_cv = cache_all_combos["best_sse_cv"]
        best_feature_indices = cache_all_combos["best_feature_indices"]

        log.info(
            "Step 4-5: LOADED FROM CACHE (%d combos, best SSE=%.6e)",
            len(sse_values_list), best_sse_cv,
        )
    else:
        # Check for partially completed combos
        cache_partial = _load_cache("step5_partial_progress")
        if cache_partial is not None:
            sse_values_list = cache_partial["sse_values"]
            combo_labels_list = cache_partial["combo_labels"]
            combo_indices_list = cache_partial["combo_indices"]
            start_combo = cache_partial["next_combo"]
            log.info(
                "Step 4-5: Resuming from combo %d / %d",
                start_combo, total_m2_combos,
            )
        else:
            sse_values_list: List[float] = []
            combo_labels_list: List[str] = []
            combo_indices_list: List[List[int]] = []
            start_combo = 0

        alpha = 0.1
        combo_count = 0
        save_interval = max(1, total_m2_combos // 20)  # save ~20 times

        for r in tqdm(
                range(0, len(m2_final) + 1),
                desc="M2 subset sizes",
                ncols=100, position=0,
        ):
            # for m2_subset in combinations(range(len(m2_final)), r):'
            for m2_subset in tqdm(
                    combinations(range(len(m2_final)), r),
                    desc=f"Combos of size {r}",
                    ncols=100,
                    total=len(list(combinations(range(len(m2_final)), r))), position=1,
            ):
                if combo_count < start_combo:
                    combo_count += 1
                    continue

                # Feature indices for this partial description
                m2_sub_indices = [m2_indices[j] for j in m2_subset]
                candidate_indices = m1_indices + m2_sub_indices

                m2_names_in_combo = [m2_final[j] for j in m2_subset]
                if m2_names_in_combo:
                    label = "M1+" + "+".join(m2_names_in_combo)
                else:
                    label = "M1 only"

                if not candidate_indices:
                    sse_values_list.append(float("inf"))
                    combo_labels_list.append("(empty)")
                    combo_indices_list.append([])
                    combo_count += 1
                    continue

                # Check individual combo cache
                combo_cache_name = f"combo_{combo_count:06d}"
                cached_combo = _load_cache(combo_cache_name)

                if cached_combo is not None:
                    sse_total = cached_combo["sse"]
                else:
                    x1_sub = x_half1[:, candidate_indices]
                    x2_sub = x_half2[:, candidate_indices]

                    # xn_sub is dataset that

                    # SSE_1: train half1, test half2
                    model_1 = fit_lsm(x1_sub, y_half1, alpha=alpha)
                    sse_1 = compute_sse(model_1, x2_sub, y_half2)

                    # SSE_2: train half2, test half1
                    model_2 = fit_lsm(x2_sub, y_half2, alpha=alpha)
                    sse_2 = compute_sse(model_2, x1_sub, y_half1)

                    sse_total = sse_1 + sse_2

                    _save_cache(combo_cache_name, {
                        "combo_idx": combo_count,
                        "label": label,
                        "candidate_indices": candidate_indices,
                        "sse_1": sse_1,
                        "sse_2": sse_2,
                        "sse": sse_total,
                    })

                sse_values_list.append(sse_total)
                combo_labels_list.append(label)
                combo_indices_list.append(candidate_indices)

                combo_count += 1

                # Periodic partial save
                if combo_count % save_interval == 0:
                    _save_cache("step5_partial_progress", {
                        "sse_values": sse_values_list,
                        "combo_labels": combo_labels_list,
                        "combo_indices": combo_indices_list,
                        "next_combo": combo_count,
                    })
                    log.debug(
                        "  Partial progress saved at combo %d / %d",
                        combo_count, total_m2_combos,
                    )

        # Find best
        sse_arr = np.array(sse_values_list)
        best_combo_idx = int(np.argmin(sse_arr))
        best_sse_cv = float(sse_arr[best_combo_idx])
        best_feature_indices = combo_indices_list[best_combo_idx]

        # Save completed results
        _save_cache("step5_all_combo_results", {
            "sse_values": sse_values_list,
            "combo_labels": combo_labels_list,
            "combo_indices": combo_indices_list,
            "best_combo_idx": best_combo_idx,
            "best_sse_cv": best_sse_cv,
            "best_feature_indices": best_feature_indices,
        })

        log.info("Step 4-5 complete: %d combos evaluated", combo_count)

    log.info("Best CV SSE = %.6e", best_sse_cv)
    log.info("Best combination: %s", combo_labels_list[best_combo_idx])

    # ══════════════════════════════════════════════════════════════
    # Step 6: Final model on ALL data (cached)
    # ══════════════════════════════════════════════════════════════
    cache_final = _load_cache("step6_final_model")

    if cache_final is not None:
        best_feature_names = cache_final["feature_names"]
        final_coefs = cache_final["coefficients"]
        final_intercept = cache_final["intercept"]
        sse_full = cache_final["sse_full"]
        best_feature_indices = cache_final["feature_indices"]
        m1_in_best = cache_final["m1_in_best"]
        m2_in_best = cache_final["m2_in_best"]
        log.info("Step 6: LOADED FROM CACHE (SSE_full=%.6e)", sse_full)
    else:
        log.info("─" * 50)
        log.info("Step 6: Training final model on ALL %d samples", n_samples)

        best_feature_names = [feature_names[i] for i in best_feature_indices]
        x_best_full = x_raw[:, best_feature_indices]

        final_model = fit_lsm(x_best_full, y, alpha=0.1)
        final_coefs = final_model.coef_
        final_intercept = float(final_model.intercept_)

        # SSE on ALL data (single metric)
        y_pred_full = final_model.predict(x_best_full)
        sse_full = float(np.sum((y - y_pred_full) ** 2))

        m1_in_best = [n for n in m1_final if n in best_feature_names]
        m2_in_best = [n for n in best_feature_names if n not in m1_final]

        _save_cache("step6_final_model", {
            "feature_names": best_feature_names,
            "feature_indices": best_feature_indices,
            "coefficients": final_coefs,
            "intercept": final_intercept,
            "sse_full": sse_full,
            "m1_in_best": m1_in_best,
            "m2_in_best": m2_in_best,
            "best_sse_cv": best_sse_cv,
        })

    # ── Log formula ──
    log.info("=" * 70)
    log.info("FINAL FORMULA")
    log.info("=" * 70)
    log.info("")
    log.info("Best formula = M1 + M2 + bias")
    log.info("")
    log.info("M1 (core features from intersection):")
    for name in m1_in_best:
        idx_in_best = best_feature_names.index(name)
        log.info("  %+.10e * %s", final_coefs[idx_in_best], name)
    log.info("")
    log.info("M2 (selected from combinations):")
    if m2_in_best:
        for name in m2_in_best:
            idx_in_best = best_feature_names.index(name)
            log.info("  %+.10e * %s", final_coefs[idx_in_best], name)
    else:
        log.info("  (none)")
    log.info("")
    log.info("Bias (intercept): %+.10e", final_intercept)
    log.info("")
    log.info("SSE (full data): %.6e", sse_full)

    # ── Save model binary ──
    model_binary = {
        "features": best_feature_names,
        "m1_features": m1_in_best,
        "m2_features": m2_in_best,
        "coefficients": final_coefs,
        "intercept": final_intercept,
        "sse_full": sse_full,
        "sse_cv": best_sse_cv,
        "feature_indices": best_feature_indices,
    }
    bin_path = os.path.join(PLOTS_DIR, "final_model.bin")
    joblib.dump(model_binary, bin_path)
    log.info("Model binary saved to %s", bin_path)

    # ── Plots ──
    log.info("─" * 50)
    log.info("Generating scientific plots")

    # --- Predict function: BEST formula (M1 + selected M2 + bias) ---
    def predict_best(m_in, n_in):
        temp_df = pd.DataFrame({
            "m": np.asarray(m_in).flatten(),
            "n": np.asarray(n_in).flatten(),
        })
        x_full_tmp, _ = generate_full_feature_matrix(temp_df)
        x_sub_tmp = x_full_tmp.values[:, best_feature_indices]
        model_tmp = Ridge(alpha=0.1, fit_intercept=True)
        model_tmp.coef_ = final_coefs
        model_tmp.intercept_ = final_intercept
        return model_tmp.predict(x_sub_tmp)

    # --- Predict function: M1 only + bias (залишковий опис) ---
    if m1_indices:
        x_m1_full = x_raw[:, m1_indices]
        model_m1_only = fit_lsm(x_m1_full, y, alpha=0.1)

        def predict_m1_only(m_in, n_in):
            temp_df = pd.DataFrame({
                "m": np.asarray(m_in).flatten(),
                "n": np.asarray(n_in).flatten(),
            })
            x_full_tmp, _ = generate_full_feature_matrix(temp_df)
            x_sub_tmp = x_full_tmp.values[:, m1_indices]
            return model_m1_only.predict(x_sub_tmp)
    else:
        predict_m1_only = None

    # --- Predict function: M1 + ALL M2 + bias (надлишковий опис) ---
    all_indices = m1_indices + m2_indices
    x_all_full = x_raw[:, all_indices]
    model_all = fit_lsm(x_all_full, y, alpha=0.1)

    def predict_m1_m2_all(m_in, n_in):
        temp_df = pd.DataFrame({
            "m": np.asarray(m_in).flatten(),
            "n": np.asarray(n_in).flatten(),
        })
        x_full_tmp, _ = generate_full_feature_matrix(temp_df)
        x_sub_tmp = x_full_tmp.values[:, all_indices]
        return model_all.predict(x_sub_tmp)

    plot_model_vs_data(
        df, predict_best, plot_dir=PLOTS_DIR, model_name="GMDH",
        predict_m1_only=predict_m1_only,
        predict_m1_m2_all=predict_m1_m2_all,
    )
    plot_sse_combinations(
        sse_values_list, combo_labels_list, best_combo_idx, PLOTS_DIR
    )

    t_total = time.perf_counter() - t_total_start
    log.info("=" * 70)
    log.info("GMDH COMPLETE. Total time: %.3f s", t_total)
    log.info("=" * 70)

    return best_feature_names, final_coefs, final_intercept, sse_full


# ═══════════════════════════ MAIN ══════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    log.info("Execution started")

    df = load_data()
    if df.empty:
        log.error('No data found in cache directory "%s"', CACHE_DIR)
        return

    log.info("Loaded %d data points", len(df))

    try:
        names, coefs, intercept, sse_full = run_gmdh(df)

        # Write formula file
        lines = [
            "Best formula = M1 + M2 + bias",
            "",
            "Full formula:",
        ]
        for w, n in zip(coefs, names):
            lines.append(f"  {w:+.16e} * [{n}]")
        lines.append(f"  {intercept:+.16e}  (bias)")
        lines.append("")
        lines.append(f"SSE (full data): {sse_full:.6e}")

        equation = "\n".join(lines)
        log.info("Final model:\n%s", equation)

        formula_path = os.path.join(PLOTS_DIR, "final_formula.txt")
        with open(formula_path, "w", encoding="utf-8") as fh:
            fh.write(equation)
        log.info("Formula written to %s", formula_path)

    except Exception:
        log.exception("Critical execution error")

    elapsed = time.perf_counter() - t0
    log.info("Total wall-clock time: %.3f s", elapsed)


if __name__ == "__main__":
    main()
