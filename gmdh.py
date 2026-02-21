import logging
import math
import os
import pickle
import re
import sys
import time
import warnings
from itertools import combinations
from typing import Any, Dict, List, Tuple, Callable

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


# ─────────────────────────── logging ───────────────────────────

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


# ─────────────────────── basis functions ───────────────────────

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


# ─────────────────── feature matrix generation ─────────────────

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


# ─────────────────────── data loading ──────────────────────────

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


# ─────────────────── coefficient clustering ────────────────────

def perform_coefficient_clustering(
        coefs: np.ndarray,
        feature_names: List[str],
        x_data: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """
    Сортуємо за зростанням модулів importance = |coef| * std(feature).
    Кластеризація: M1 (важливі, ближче до найбільшого) та M2 (решта).

    Повертає (m1_names, m2_names).
    """
    feature_stds = np.std(x_data, axis=0)
    items = []
    for w, std, name in zip(coefs, feature_stds, feature_names):
        importance = abs(w) * (std if std > 1e-9 else 1.0)
        items.append({"raw_val": w, "name": name, "importance": importance})

    # Сортуємо за зростанням модулів (importance)
    sorted_items = sorted(items, key=lambda x: x["importance"], reverse=True)

    if not sorted_items:
        return [], list(feature_names)

    smallest_imp = sorted_items[-1]["importance"]
    m1_items = [sorted_items[0]]

    for i in range(1, len(sorted_items)):
        candidate = sorted_items[i]
        avg_m1 = sum(item["importance"] for item in m1_items) / len(m1_items)
        dist_good = avg_m1 - candidate["importance"]
        dist_bad = candidate["importance"] - smallest_imp
        if dist_good < dist_bad:
            m1_items.append(candidate)
        else:
            break

    m1_names = [item["name"] for item in m1_items]
    m2_names = [item["name"] for item in sorted_items if item["name"] not in m1_names]

    return m1_names, m2_names


# ────────────────────── LSM fit + SSE ──────────────────────────

def fit_lsm(x_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.1):
    """Fit Ridge regression, return model."""
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, y_train)
    return model


def compute_sse(model, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Compute Sum of Squared Errors on test set."""
    y_pred = model.predict(x_test)
    residuals = y_test - y_pred
    return float(np.sum(residuals ** 2))


# ──────────────────── scientific plots ─────────────────────────

def plot_model_vs_data(
        df: pd.DataFrame,
        predict_func: Callable,
        plot_dir: str,
        model_name: str = "GMDH",
) -> None:
    """
    Наукові графіки: реальні дані vs модель.
    - 2D зрізи по фіксованих n (ops vs m) та по фіксованих m (ops vs n)
    - Графік залишків
    - 3D поверхня
    """
    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    m_vals = sorted(df["m"].unique())
    n_vals = sorted(df["n"].unique())

    y_actual = df["ops"].values
    y_predicted = predict_func(df["m"].values, df["n"].values)
    residuals = y_actual - y_predicted

    # ── 2D slices: fixed n, ops vs m ──
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=200)

        ax = axes[0]
        for n_fixed in n_vals:
            subset = df[df["n"] == n_fixed].sort_values("m")
            if len(subset) < 2:
                continue
            ax.scatter(subset["m"], subset["ops"], s=15, alpha=0.6, label=f"data n={n_fixed}")
            m_range = np.linspace(subset["m"].min(), subset["m"].max(), 200)
            y_model = predict_func(m_range, np.full_like(m_range, n_fixed))
            ax.plot(m_range, y_model, "--", lw=1.5, alpha=0.8)
        ax.set_xlabel("m (кількість обмежень)", fontsize=12)
        ax.set_ylabel("Операції", fontsize=12)
        ax.set_title("Залежність кількості операцій від m", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        ax = axes[1]
        for m_fixed in m_vals:
            subset = df[df["m"] == m_fixed].sort_values("n")
            if len(subset) < 2:
                continue
            ax.scatter(subset["n"], subset["ops"], s=15, alpha=0.6, label=f"data m={m_fixed}")
            n_range = np.linspace(subset["n"].min(), subset["n"].max(), 200)
            y_model = predict_func(np.full_like(n_range, m_fixed), n_range)
            ax.plot(n_range, y_model, "--", lw=1.5, alpha=0.8)
        ax.set_xlabel("n (кількість змінних)", fontsize=12)
        ax.set_ylabel("Операції", fontsize=12)
        ax.set_title("Залежність кількості операцій від n", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        fig.suptitle(f"Модель {safe_name}: дані vs апроксимація ({scale})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_slices_{scale}.png"))
        plt.close(fig)

    # ── Residuals plot ──
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
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="Ідеал (y=x)")
    ax.set_xlabel("Реальні значення", fontsize=12)
    ax.set_ylabel("Передбачені значення", fontsize=12)
    ax.set_title("Реальні vs передбачені", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Модель {safe_name}: аналіз залишків", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{safe_name}_residuals.png"))
    plt.close(fig)

    # ── 3D surface ──
    for scale in ("linear", "log"):
        df_grouped = df.groupby(["m", "n"])["ops"].mean().reset_index()
        m_grid, n_grid = np.meshgrid(
            np.linspace(min(m_vals), max(m_vals), 50),
            np.linspace(min(n_vals), max(n_vals), 50),
        )
        ops_grid = predict_func(m_grid.ravel(), n_grid.ravel()).reshape(m_grid.shape)

        ops_plot = ops_grid.copy()
        ops_data = df_grouped["ops"].values.copy()
        if scale == "log":
            ops_plot = np.log10(np.clip(ops_plot, 1e-9, None))
            ops_data = np.log10(np.clip(ops_data, 1e-9, None))

        fig = plt.figure(figsize=(14, 10), dpi=150)
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(m_grid, n_grid, ops_plot, cmap="viridis", alpha=0.7)
        ax.scatter(
            df_grouped["m"].values,
            df_grouped["n"].values,
            ops_data,
            c="red", marker="o", s=20, label="Дані",
        )
        ax.set_xlabel("m (обмеження)", fontsize=11)
        ax.set_ylabel("n (змінні)", fontsize=11)
        z_label = "log₁₀(Операції)" if scale == "log" else "Операції"
        ax.set_zlabel(z_label, fontsize=11)
        ax.set_title(f"Модель {safe_name}: 3D поверхня ({scale})", fontsize=13)
        fig.colorbar(surf, shrink=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_3d.png"))
        plt.close(fig)


def plot_sse_combinations(
        sse_values: List[float],
        combo_labels: List[str],
        best_idx: int,
        plot_dir: str,
) -> None:
    """Графік SSE для всіх комбінацій M2."""
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)

    x_pos = np.arange(len(sse_values))
    colors = ["red" if i == best_idx else "steelblue" for i in range(len(sse_values))]
    ax.bar(x_pos, sse_values, color=colors, alpha=0.7)
    ax.set_xlabel("Комбінація M2 компонент", fontsize=12)
    ax.set_ylabel("SSE (сума квадратів помилок)", fontsize=12)
    ax.set_title("SSE для часткових описів (M1 + комбінації M2)", fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Якщо комбінацій не дуже багато — показуємо мітки
    if len(combo_labels) <= 30:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(combo_labels, rotation=90, fontsize=7)
    else:
        ax.set_xticks([best_idx])
        ax.set_xticklabels([combo_labels[best_idx]], rotation=45, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "sse_combinations.png"))
    plt.close(fig)


# ──────────────────── cache utilities ──────────────────────────

def _gmdh_result_cache_path() -> str:
    return os.path.join(GMDH_CACHE_DIR, "gmdh_result.pkl")


def _save_result_cache(result: Dict[str, Any]) -> None:
    with open(_gmdh_result_cache_path(), "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_result_cache() -> Dict[str, Any] | None:
    path = _gmdh_result_cache_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except (pickle.UnpicklingError, EOFError, Exception):
        return None


# ──────────────────── MAIN GMDH ALGORITHM ──────────────────────

def run_gmdh(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, float, float]:
    """
    Переписаний алгоритм МГУА (комбінаторний GMDH):

    1. Дані → дві половини.
    2. Для кожної половини:
       - Повний надлишковий опис (Ridge на всіх фічах)
       - Сортування коефіцієнтів за модулем importance
       - Кластеризація → M1_i (важливі), M2_i (решта)
    3. Результуючі кластери:
       - M1 = M1_1 ∩ M1_2 (перетин — тільки ті фічі, що важливі в обох)
       - M2 = все інше (об'єднання M2_1 ∪ M2_2 ∪ ті, що не потрапили в перетин)
    4. Часткові описи: M1 + всі можливі комбінації M2 (від 0 до |M2| елементів)
    5. Для кожного часткового опису:
       - SSE_1: тренуємо на 1-й половині (LSM), SSE на 2-й
       - SSE_2: тренуємо на 2-й половині (LSM), SSE на 1-й
       - SSE = SSE_1 + SSE_2
    6. Найкраща формула = argmin SSE

    Повертає (feature_names, coefficients, intercept, best_sse).
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

    log.info("Feature matrix: %d samples × %d features", n_samples, n_features)
    log.info("Features: %s", ", ".join(feature_names))

    # ── Step 1: Split data ──
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    split = n_samples // 2
    idx_1, idx_2 = indices[:split], indices[split:]

    x_half1, y_half1 = x_raw[idx_1], y[idx_1]
    x_half2, y_half2 = x_raw[idx_2], y[idx_2]

    log.info("Split: half1=%d samples, half2=%d samples", len(idx_1), len(idx_2))

    # ── Step 2a: Full redundant description on half1 (check on half2) ──
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
    log.info("  M1_1 (important, half1): %s", m1_1_names)
    log.info("  M2_1 (rest, half1):      %s", m2_1_names)

    # ── Step 2b: Full redundant description on half2 (check on half1) ──
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
    log.info("  M1_2 (important, half2): %s", m1_2_names)
    log.info("  M2_2 (rest, half2):      %s", m2_2_names)

    # ── Step 3: Intersection / Union ──
    log.info("─" * 50)
    log.info("Step 3: Computing final M1 (intersection) and M2 (everything else)")

    m1_final = sorted(set(m1_1_names) & set(m1_2_names))
    # M2 = all features NOT in M1
    all_feature_set = set(feature_names)
    m2_final = sorted(all_feature_set - set(m1_final))

    log.info("  M1 (intersection): %d features: %s", len(m1_final), m1_final)
    log.info("  M2 (everything else): %d features: %s", len(m2_final), m2_final)

    if not m1_final:
        log.warning("  M1 is empty! All features will be in M2.")

    # ── Step 4: Get feature indices ──
    m1_indices = [feature_names.index(name) for name in m1_final]
    m2_indices = [feature_names.index(name) for name in m2_final]

    # ── Step 5: Enumerate partial descriptions ──
    # Partial description = M1 + subset of M2
    # Subsets of M2: from 0 elements to all elements
    log.info("─" * 50)

    total_m2_combos = sum(math.comb(len(m2_final), r) for r in range(0, len(m2_final) + 1))
    log.info(
        "Step 5: Evaluating partial descriptions: M1(%d) + combinations of M2(%d) = %d candidates",
        len(m1_final), len(m2_final), total_m2_combos,
    )

    best_sse = float("inf")
    best_feature_indices = None
    best_m2_combo = None
    best_combo_label = None

    sse_values_list: List[float] = []
    combo_labels_list: List[str] = []
    best_combo_idx = 0

    alpha = 0.1  # Ridge regularisation

    # Pre-compute M1 part (constant across all combos)
    combo_count = 0

    for r in tqdm(
            range(0, len(m2_final) + 1),
            desc="M2 subset sizes",
            ncols=100,
    ):
        # for m2_subset in combinations(range(len(m2_final)), r):#
        for m2_subset in tqdm(
                combinations(range(len(m2_final)), r),
                desc=f"Combos of size {r}",
                ncols=100,
                leave=False,
        ):
            # Feature indices for this partial description
            m2_sub_indices = [m2_indices[j] for j in m2_subset]
            candidate_indices = m1_indices + m2_sub_indices

            if not candidate_indices:
                # Skip empty model
                sse_values_list.append(float("inf"))
                combo_labels_list.append("(empty)")
                combo_count += 1
                continue

            # Extract sub-matrices
            x1_sub = x_half1[:, candidate_indices]
            x2_sub = x_half2[:, candidate_indices]

            # SSE_1: train on half1, test on half2
            model_1 = fit_lsm(x1_sub, y_half1, alpha=alpha)
            sse_1 = compute_sse(model_1, x2_sub, y_half2)

            # SSE_2: train on half2, test on half1
            model_2 = fit_lsm(x2_sub, y_half2, alpha=alpha)
            sse_2 = compute_sse(model_2, x1_sub, y_half1)

            sse_total = sse_1 + sse_2

            # Label for this combination
            m2_names_in_combo = [m2_final[j] for j in m2_subset]
            if m2_names_in_combo:
                label = "M1+" + "+".join(m2_names_in_combo)
            else:
                label = "M1 only"

            sse_values_list.append(sse_total)
            combo_labels_list.append(label)

            if sse_total < best_sse:
                best_sse = sse_total
                best_feature_indices = candidate_indices
                best_m2_combo = m2_names_in_combo
                best_combo_label = label
                best_combo_idx = combo_count

                log.debug(
                    "  New best SSE=%.6e with %d features: %s",
                    best_sse, len(candidate_indices), label,
                )

            combo_count += 1

    log.info("─" * 50)
    log.info("Step 5 complete: evaluated %d partial descriptions", combo_count)
    log.info("Best SSE = %.6e", best_sse)
    log.info("Best combination: %s", best_combo_label)

    # ── Step 6: Final model — retrain on full data ──
    log.info("─" * 50)
    log.info("Step 6: Training final model on full dataset")

    best_feature_names = [feature_names[i] for i in best_feature_indices]
    x_best_full = x_raw[:, best_feature_indices]

    final_model = fit_lsm(x_best_full, y, alpha=alpha)
    final_coefs = final_model.coef_
    final_intercept = final_model.intercept_

    # ── Format output ──
    m1_in_best = [n for n in m1_final if n in best_feature_names]
    m2_in_best = [n for n in best_feature_names if n not in m1_final]

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

    # SSE on full data
    y_pred_full = final_model.predict(x_best_full)
    sse_full = float(np.sum((y - y_pred_full) ** 2))
    mse_full = float(np.mean((y - y_pred_full) ** 2))
    log.info("Full data SSE: %.6e", sse_full)
    log.info("Full data MSE: %.6e", mse_full)

    # ── Save model ──
    model_binary = {
        "features": best_feature_names,
        "m1_features": m1_in_best,
        "m2_features": m2_in_best,
        "coefficients": final_coefs,
        "intercept": final_intercept,
        "best_sse_cv": best_sse,
        "mse_full": mse_full,
        "feature_indices": best_feature_indices,
    }
    bin_path = os.path.join(PLOTS_DIR, "final_model.bin")
    joblib.dump(model_binary, bin_path)
    log.info("Model binary saved to %s", bin_path)

    _save_result_cache(model_binary)
    log.info("Result cached to %s", GMDH_CACHE_DIR)

    # ── Plots ──
    log.info("─" * 50)
    log.info("Generating plots")

    def predict_func(m_in, n_in):
        temp_df = pd.DataFrame({
            "m": np.asarray(m_in).flatten(),
            "n": np.asarray(n_in).flatten(),
        })
        x_full_tmp, _ = generate_full_feature_matrix(temp_df)
        x_sub_tmp = x_full_tmp.values[:, best_feature_indices]
        return final_model.predict(x_sub_tmp)

    plot_model_vs_data(df, predict_func, PLOTS_DIR, model_name="GMDH")
    plot_sse_combinations(sse_values_list, combo_labels_list, best_combo_idx, PLOTS_DIR)

    t_total = time.perf_counter() - t_total_start
    log.info("=" * 70)
    log.info("GMDH COMPLETE. Total time: %.3f s", t_total)
    log.info("=" * 70)

    return best_feature_names, final_coefs, final_intercept, best_sse


# ──────────────────────────── main ─────────────────────────────

def main() -> None:
    t0 = time.perf_counter()
    log.info("Execution started")

    df = load_data()
    if df.empty:
        log.error('No data found in cache directory "%s"', CACHE_DIR)
        return

    log.info("Loaded %d data points", len(df))

    try:
        names, coefs, intercept, best_sse = run_gmdh(df)

        # Write formula file
        lines = ["Best formula = M1 + M2 + bias", ""]
        lines.append("Full formula:")
        terms = []
        for w, n in zip(coefs, names):
            terms.append(f"  {w:+.16e} * [{n}]")
        lines.extend(terms)
        lines.append(f"  {intercept:+.16e}  (bias)")
        lines.append("")
        lines.append(f"Cross-validated SSE: {best_sse:.6e}")

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
