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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Шрифт для коректного відображення кирилиці на графіках
matplotlib.rcParams["font.family"] = "DejaVu Sans"

CACHE_DIR = "cache"
GMDH_CACHE_DIR = "gmdh_cache"
PLOTS_DIR = "plots_combi"
LOG_FILE = "gmdh_execution.log"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GMDH_CACHE_DIR, exist_ok=True)


# ═══════════════════════════ ЛОГУВАННЯ ═══════════════════════════

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


# ═══════════════════════ ДОПОМІЖНІ ФУНКЦІЇ КЕШУ ═════════════════════════

def _cache_path(name: str) -> str:
    return os.path.join(GMDH_CACHE_DIR, f"{name}.pkl")


def _save_cache(name: str, data: Any) -> None:
    path = _cache_path(name)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.debug("Кеш збережено: %s", path)


def _load_cache(name: str) -> Optional[Any]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        log.debug("Кеш завантажено: %s", path)
        return data
    except (pickle.UnpicklingError, EOFError, Exception) as e:
        log.warning("Кеш пошкоджено %s: %s", path, e)
        return None


def _cache_exists(name: str) -> bool:
    return os.path.exists(_cache_path(name))


# ═══════════════════ БАЗИСНІ ФУНКЦІЇ ═══════════════════════════

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


# ═══════════════ ГЕНЕРАЦІЯ МАТРИЦІ ОЗНАК ═════════════════════

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


# ═══════════════════ ЗАВАНТАЖЕННЯ ДАНИХ ══════════════════════════════

def load_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_DIR):
        return pd.DataFrame()

    all_data: List[List] = []
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]

    for fname in tqdm(files, desc="Завантаження кешу", ncols=100):
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


# ═══════════════ КЛАСТЕРИЗАЦІЯ КОЕФІЦІЄНТІВ ════════════════════════

def perform_coefficient_clustering(
        coefs: np.ndarray,
        feature_names: List[str],
        x_data: np.ndarray,
) -> Tuple[List[str], List[str]]:
    """
    Сортуємо за важливістю = |коеф| * стд(ознака).
    Кластеризація: M1 (важливі) та M2 (решта).
    Повертає (m1_імена, m2_імена).
    """
    feature_stds = np.std(x_data, axis=0)
    items = []
    for w, std, name in tqdm(
            zip(coefs, feature_stds, feature_names),
            desc="Кластеризація коефіцієнтів",
            ncols=100,
    ):
        importance = abs(w) * (std if std > 1e-9 else 1.0)
        items.append({"name": name, "importance": importance})

    sorted_items = sorted(items, key=lambda x: x["importance"], reverse=True)

    if not sorted_items:
        return [], list(feature_names)

    smallest_imp = sorted_items[-1]["importance"]
    m1_items = [sorted_items[0]]

    for i in tqdm(
            range(1, len(sorted_items)),
            desc="Визначення розбиття M1/M2",
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


# ═══════════════════ МНК / ЗСК ═════════════════════════════════

def fit_lsm(
        x_train: np.ndarray, y_train: np.ndarray, alpha: float = 0.1
) -> Ridge:
    """Підгонка моделі методом найменших квадратів (Ridge-регресія)."""
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, y_train)
    return model


def compute_sse(model: Ridge, x_test: np.ndarray, y_test: np.ndarray) -> float:
    """Обчислення залишкової суми квадратів (ЗСК / SSE)."""
    y_pred = model.predict(x_test)
    return float(np.sum((y_test - y_pred) ** 2))


# ═══════════════════ НАУКОВІ ГРАФІКИ ══════════════════════════

def plot_model_vs_data(
        df: pd.DataFrame,
        predict_func: Callable,
        plot_dir: str,
        model_name: str = "МГУА",
        predict_m1_only: Callable = None,
        predict_m1_m2_all: Callable = None,
) -> None:
    """
    Наукові графіки:
    - 2D зрізи з boxplot + три моделі (M1+зміщення, M1+M2+зміщення, найкраща)
    - Графік залишків
    - 3D поверхні для всіх трьох моделей
    """
    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    m_vals = sorted(df["m"].unique())
    n_vals = sorted(df["n"].unique())

    y_actual = df["ops"].values
    y_predicted = predict_func(df["m"].values, df["n"].values)

    # ═══════════════════════════════════════════════════════════
    # 2D зрізи: boxplot + криві моделей
    # ═══════════════════════════════════════════════════════════
    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(30, 20), dpi=200)

        # ── Ліворуч: операції vs m, фіксоване n ──
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

            # Лінія середніх значень
            avg_ops = subset.groupby("m")["ops"].mean()
            ax.plot(
                avg_ops.index, avg_ops.values,
                "k-", linewidth=1, alpha=0.7,
            )

            m_range = np.linspace(min(m_vals), max(m_vals), 100)

            # M1 + зміщення (залишковий опис)
            if predict_m1_only is not None:
                y_m1 = predict_m1_only(m_range, np.full_like(m_range, n_fixed))
                ax.plot(
                    m_range, y_m1, ":",
                    color="orange", linewidth=2, alpha=0.7,
                )

            # M1 + усі M2 + зміщення (надлишковий опис)
            if predict_m1_m2_all is not None:
                y_all = predict_m1_m2_all(m_range, np.full_like(m_range, n_fixed))
                ax.plot(
                    m_range, y_all, "-.",
                    color="purple", linewidth=2, alpha=0.7,
                )

            # Найкраща формула (M1 + відібрані M2 + зміщення)
            y_best = predict_func(m_range, np.full_like(m_range, n_fixed))
            ax.plot(
                m_range, y_best, "--",
                color="green", linewidth=2.5,
            )

        ax.set_xlabel("m (кількість обмежень)", fontsize=12)
        ax.set_ylabel("Кількість операцій", fontsize=12)
        ax.set_title("Залежність кількості операцій від m", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)

        # Легенда з описами моделей
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="orange", ls=":", lw=2,
                   label="M1 + зміщення (залишковий опис)"),
            Line2D([0], [0], color="purple", ls="-.", lw=2,
                   label="M1 + M2(усі) + зміщення (надлишковий опис)"),
            Line2D([0], [0], color="green", ls="--", lw=2.5,
                   label="Найкраща формула (МГУА)"),
            Line2D([0], [0], color="black", ls="-", lw=1,
                   label="Середнє значення даних"),
        ]
        ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

        # ── Праворуч: операції vs n, фіксоване m ──
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

            # M1 + зміщення
            if predict_m1_only is not None:
                y_m1 = predict_m1_only(np.full_like(n_range, m_fixed), n_range)
                ax.plot(
                    n_range, y_m1, ":",
                    color="orange", linewidth=2, alpha=0.7,
                )

            # M1 + усі M2 + зміщення
            if predict_m1_m2_all is not None:
                y_all = predict_m1_m2_all(np.full_like(n_range, m_fixed), n_range)
                ax.plot(
                    n_range, y_all, "-.",
                    color="purple", linewidth=2, alpha=0.7,
                )

            # Найкраща формула
            y_best = predict_func(np.full_like(n_range, m_fixed), n_range)
            ax.plot(
                n_range, y_best, "--",
                color="green", linewidth=2.5,
            )

        ax.set_xlabel("n (кількість змінних)", fontsize=12)
        ax.set_ylabel("Кількість операцій", fontsize=12)
        ax.set_title("Залежність кількості операцій від n", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

        scale_name = "лінійний" if scale == "linear" else "логарифмічний"
        fig.suptitle(
            f"Модель {safe_name}: дані та апроксимація ({scale_name} масштаб)",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_2d.png"))
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # Залишки (для найкращої формули)
    # ═══════════════════════════════════════════════════════════
    residuals = y_actual - y_predicted

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=200)

    ax = axes[0]
    ax.scatter(y_predicted, residuals, s=8, alpha=0.4, c="steelblue")
    ax.axhline(0, color="red", lw=1.5, ls="--")
    ax.set_xlabel("Передбачені значення", fontsize=12)
    ax.set_ylabel("Залишки (факт − прогноз)", fontsize=12)
    ax.set_title("Залишки відносно передбачених значень", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(y_actual, y_predicted, s=8, alpha=0.4, c="darkgreen")
    lim_min = min(y_actual.min(), y_predicted.min())
    lim_max = max(y_actual.max(), y_predicted.max())
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "r--", lw=1.5, label="Ідеал (y = x)",
    )
    ax.set_xlabel("Фактичні значення", fontsize=12)
    ax.set_ylabel("Передбачені значення", fontsize=12)
    ax.set_title("Фактичні та передбачені значення", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Модель {safe_name}: аналіз залишків", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{safe_name}_залишки.png"))
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    # 3D поверхні: усі три моделі
    # ═══════════════════════════════════════════════════════════
    df_grouped = df.groupby(["m", "n"])["ops"].mean().reset_index()

    for scale in ("linear", "log"):
        m_grid, n_grid = np.meshgrid(
            np.linspace(min(m_vals), max(m_vals), 50),
            np.linspace(min(n_vals), max(n_vals), 50),
        )

        # Обчислення поверхонь
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

        z_label = "log₁₀(Операції)" if scale == "log" else "Кількість операцій"
        scale_name = "лінійний" if scale == "linear" else "логарифмічний"

        fig = plt.figure(figsize=(20, 14), dpi=150)

        # --- Підграфік 1: M1 + зміщення (залишковий) ---
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
            ax1.set_xlabel("m (обмеження)", fontsize=10)
            ax1.set_ylabel("n (змінні)", fontsize=10)
            ax1.set_zlabel(z_label, fontsize=10)
            ax1.set_title("M1 + зміщення\n(залишковий опис)", fontsize=11)

        # --- Підграфік 2: M1 + усі M2 + зміщення (надлишковий) ---
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
            ax2.set_xlabel("m (обмеження)", fontsize=10)
            ax2.set_ylabel("n (змінні)", fontsize=10)
            ax2.set_zlabel(z_label, fontsize=10)
            ax2.set_title("M1 + M2(усі) + зміщення\n(надлишковий опис)", fontsize=11)

        # --- Підграфік 3: Найкраща формула ---
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
        ax3.set_xlabel("m (обмеження)", fontsize=10)
        ax3.set_ylabel("n (змінні)", fontsize=10)
        ax3.set_zlabel(z_label, fontsize=10)
        ax3.set_title("Найкраща формула\n(результат МГУА)", fontsize=11)

        fig.suptitle(
            f"Порівняння моделей ({scale_name} масштаб): "
            f"залишковий → надлишковий → оптимальний",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_3d.png"))
        plt.close(fig)


# ═══════════════════ ОСНОВНИЙ АЛГОРИТМ МГУА ═══════════════════════

def run_gmdh(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, float, float]:
    """
    Алгоритм МГУА (метод групового урахування аргументів):

    1. Дані → дві половини
    2. Для кожної половини: повний опис → кластеризація → M1_i, M2_i
    3. M1 = M1_1 ∩ M1_2;  M2 = все інше
    4. Часткові описи: M1 + комбінації M2
    5. ЗСК = ЗСК_1 + ЗСК_2 (перехресна перевірка)
    6. Найкраща формула = argmin ЗСК
    7. Фінальна модель тренується НА ВСІХ ДАНИХ, ЗСК рахується на всіх даних

    Повертає (імена_ознак, коефіцієнти, зміщення, зск_повне).
    """
    t_total_start = time.perf_counter()
    log.info("=" * 70)
    log.info("ПОЧАТОК АЛГОРИТМУ МГУА")
    log.info("=" * 70)

    # ── Крок 0: Матриця ознак ──
    x_df, feature_names = generate_full_feature_matrix(df)
    x_raw = x_df.values
    y = df["ops"].values
    n_features = len(feature_names)
    n_samples = len(y)

    log.info(
        "Матриця ознак: %d спостережень × %d ознак", n_samples, n_features
    )
    log.info("Ознаки: %s", ", ".join(feature_names))

    # ── Крок 1: Розбиття даних ──
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    split = n_samples // 2
    idx_1, idx_2 = indices[:split], indices[split:]

    x_half1, y_half1 = x_raw[idx_1], y[idx_1]
    x_half2, y_half2 = x_raw[idx_2], y[idx_2]

    log.info(
        "Розбиття: половина 1 = %d спостережень, половина 2 = %d спостережень",
        len(idx_1), len(idx_2),
    )

    # ══════════════════════════════════════════════════════════════
    # Крок 2а: Кластеризація на ПОЛОВИНІ-1 (з кешем)
    # ══════════════════════════════════════════════════════════════
    cache_cluster_1 = _load_cache("крок2а_кластеризація_половина1")

    if cache_cluster_1 is not None:
        m1_1_names = cache_cluster_1["m1_names"]
        m2_1_names = cache_cluster_1["m2_names"]
        coefs_1 = cache_cluster_1["coefs"]
        log.info("Крок 2а: ЗАВАНТАЖЕНО З КЕШУ")
    else:
        log.info("─" * 50)
        log.info("Крок 2а: Повний опис на ПОЛОВИНІ-1, кластеризація")

        model_full_1 = fit_lsm(x_half1, y_half1)
        coefs_1 = model_full_1.coef_

        log.info("  Коефіцієнти повної моделі 1:")
        for name, c in zip(feature_names, coefs_1):
            log.info("    %-25s : %+.6e", name, c)

        m1_1_names, m2_1_names = perform_coefficient_clustering(
            coefs_1, feature_names, x_half1
        )

        _save_cache("крок2а_кластеризація_половина1", {
            "m1_names": m1_1_names,
            "m2_names": m2_1_names,
            "coefs": coefs_1,
        })

    log.info("  M1_1 (важливі, половина 1): %s", m1_1_names)
    log.info("  M2_1 (решта, половина 1):    %s", m2_1_names)

    # ══════════════════════════════════════════════════════════════
    # Крок 2б: Кластеризація на ПОЛОВИНІ-2 (з кешем)
    # ══════════════════════════════════════════════════════════════
    cache_cluster_2 = _load_cache("крок2б_кластеризація_половина2")

    if cache_cluster_2 is not None:
        m1_2_names = cache_cluster_2["m1_names"]
        m2_2_names = cache_cluster_2["m2_names"]
        coefs_2 = cache_cluster_2["coefs"]
        log.info("Крок 2б: ЗАВАНТАЖЕНО З КЕШУ")
    else:
        log.info("─" * 50)
        log.info("Крок 2б: Повний опис на ПОЛОВИНІ-2, кластеризація")

        model_full_2 = fit_lsm(x_half2, y_half2)
        coefs_2 = model_full_2.coef_

        log.info("  Коефіцієнти повної моделі 2:")
        for name, c in zip(feature_names, coefs_2):
            log.info("    %-25s : %+.6e", name, c)

        m1_2_names, m2_2_names = perform_coefficient_clustering(
            coefs_2, feature_names, x_half2
        )

        _save_cache("крок2б_кластеризація_половина2", {
            "m1_names": m1_2_names,
            "m2_names": m2_2_names,
            "coefs": coefs_2,
        })

    log.info("  M1_2 (важливі, половина 2): %s", m1_2_names)
    log.info("  M2_2 (решта, половина 2):    %s", m2_2_names)

    # ══════════════════════════════════════════════════════════════
    # Крок 3: Перетин / Об'єднання (з кешем)
    # ══════════════════════════════════════════════════════════════
    cache_intersection = _load_cache("крок3_перетин")

    if cache_intersection is not None:
        m1_final = cache_intersection["m1_final"]
        m2_final = cache_intersection["m2_final"]
        log.info("Крок 3: ЗАВАНТАЖЕНО З КЕШУ")
    else:
        log.info("─" * 50)
        log.info("Крок 3: M1 = перетин, M2 = все інше")

        m1_final = sorted(set(m1_1_names) & set(m1_2_names))
        m2_final = sorted(set(feature_names) - set(m1_final))

        _save_cache("крок3_перетин", {
            "m1_final": m1_final,
            "m2_final": m2_final,
        })

    log.info("  M1 (перетин): %d ознак: %s", len(m1_final), m1_final)
    log.info("  M2 (все інше): %d ознак: %s", len(m2_final), m2_final)

    if not m1_final:
        log.warning("  M1 порожній! Усі ознаки потрапляють до M2.")

    # ── Індекси ознак ──
    m1_indices = [feature_names.index(name) for name in m1_final]
    m2_indices = [feature_names.index(name) for name in m2_final]

    # ══════════════════════════════════════════════════════════════
    # Кроки 4–5: Оцінка часткових описів (кожна комбінація кешується)
    # ══════════════════════════════════════════════════════════════
    total_m2_combos = sum(
        math.comb(len(m2_final), r) for r in range(0, len(m2_final) + 1)
    )
    log.info("─" * 50)
    log.info(
        "Кроки 4-5: Оцінка %d часткових описів "
        "(M1[%d] + комбінації M2[%d])",
        total_m2_combos, len(m1_final), len(m2_final),
    )

    # Спроба завантажити завершені результати
    cache_all_combos = _load_cache("крок5_усі_комбінації")

    if cache_all_combos is not None:
        sse_values_list = cache_all_combos["sse_values"]
        combo_labels_list = cache_all_combos["combo_labels"]
        combo_indices_list = cache_all_combos["combo_indices"]
        best_combo_idx = cache_all_combos["best_combo_idx"]
        best_sse_cv = cache_all_combos["best_sse_cv"]
        best_feature_indices = cache_all_combos["best_feature_indices"]

        log.info(
            "Кроки 4-5: ЗАВАНТАЖЕНО З КЕШУ (%d комбінацій, найкраща ЗСК=%.6e)",
            len(sse_values_list), best_sse_cv,
        )
    else:
        # Перевірка частково завершених комбінацій
        cache_partial = _load_cache("крок5_проміжний_прогрес")
        if cache_partial is not None:
            sse_values_list = cache_partial["sse_values"]
            combo_labels_list = cache_partial["combo_labels"]
            combo_indices_list = cache_partial["combo_indices"]
            start_combo = cache_partial["next_combo"]
            log.info(
                "Кроки 4-5: Продовження з комбінації %d / %d",
                start_combo, total_m2_combos,
            )
        else:
            sse_values_list: List[float] = []
            combo_labels_list: List[str] = []
            combo_indices_list: List[List[int]] = []
            start_combo = 0

        alpha = 0.1
        combo_count = 0
        save_interval = max(1, total_m2_combos // 20)

        for r in tqdm(
                range(0, len(m2_final) + 1),
                desc="Розміри підмножин M2",
                ncols=100, position=0,
        ):
            for m2_subset in tqdm(
                    combinations(range(len(m2_final)), r),
                    desc=f"Комбінації розміру {r}",
                    ncols=100,
                    total=math.comb(len(m2_final), r),
                    position=1,
                    leave=False,
            ):
                if combo_count < start_combo:
                    combo_count += 1
                    continue

                # Індекси ознак для цього часткового опису
                m2_sub_indices = [m2_indices[j] for j in m2_subset]
                candidate_indices = m1_indices + m2_sub_indices

                m2_names_in_combo = [m2_final[j] for j in m2_subset]
                if m2_names_in_combo:
                    label = "M1+" + "+".join(m2_names_in_combo)
                else:
                    label = "тільки M1"

                if not candidate_indices:
                    sse_values_list.append(float("inf"))
                    combo_labels_list.append("(порожній)")
                    combo_indices_list.append([])
                    combo_count += 1
                    continue

                # Перевірка кешу окремої комбінації
                combo_cache_name = f"комбо_{combo_count:06d}"
                cached_combo = _load_cache(combo_cache_name)

                if cached_combo is not None:
                    sse_total = cached_combo["sse"]
                else:
                    x1_sub = x_half1[:, candidate_indices]
                    x2_sub = x_half2[:, candidate_indices]

                    # ЗСК_1: тренуємо на половині 1, перевіряємо на половині 2
                    model_1 = fit_lsm(x1_sub, y_half1, alpha=alpha)
                    sse_1 = compute_sse(model_1, x2_sub, y_half2)

                    # ЗСК_2: тренуємо на половині 2, перевіряємо на половині 1
                    model_2 = fit_lsm(x2_sub, y_half2, alpha=alpha)
                    sse_2 = compute_sse(model_2, x1_sub, y_half1)

                    sse_total = sse_1 + sse_2

                    _save_cache(combo_cache_name, {
                        "номер_комбінації": combo_count,
                        "мітка": label,
                        "індекси_ознак": candidate_indices,
                        "зск_1": sse_1,
                        "зск_2": sse_2,
                        "зск": sse_total,
                    })

                sse_values_list.append(sse_total)
                combo_labels_list.append(label)
                combo_indices_list.append(candidate_indices)

                combo_count += 1

                # Періодичне збереження проміжного стану
                if combo_count % save_interval == 0:
                    _save_cache("крок5_проміжний_прогрес", {
                        "sse_values": sse_values_list,
                        "combo_labels": combo_labels_list,
                        "combo_indices": combo_indices_list,
                        "next_combo": combo_count,
                    })
                    log.debug(
                        "  Проміжний стан збережено на комбінації %d / %d",
                        combo_count, total_m2_combos,
                    )

        # Знаходимо найкращу
        sse_arr = np.array(sse_values_list)
        best_combo_idx = int(np.argmin(sse_arr))
        best_sse_cv = float(sse_arr[best_combo_idx])
        best_feature_indices = combo_indices_list[best_combo_idx]

        # Зберігаємо завершені результати
        _save_cache("крок5_усі_комбінації", {
            "sse_values": sse_values_list,
            "combo_labels": combo_labels_list,
            "combo_indices": combo_indices_list,
            "best_combo_idx": best_combo_idx,
            "best_sse_cv": best_sse_cv,
            "best_feature_indices": best_feature_indices,
        })

        log.info("Кроки 4-5 завершено: оцінено %d комбінацій", combo_count)

    log.info("Найкраща ЗСК (перехресна перевірка) = %.6e", best_sse_cv)
    log.info("Найкраща комбінація: %s", combo_labels_list[best_combo_idx])

    # ══════════════════════════════════════════════════════════════
    # Крок 6: Фінальна модель на ВСІХ даних (з кешем)
    # ══════════════════════════════════════════════════════════════
    cache_final = _load_cache("крок6_фінальна_модель")

    if cache_final is not None:
        best_feature_names = cache_final["імена_ознак"]
        final_coefs = cache_final["коефіцієнти"]
        final_intercept = cache_final["зміщення"]
        sse_full = cache_final["зск_повне"]
        best_feature_indices = cache_final["індекси_ознак"]
        m1_in_best = cache_final["m1_у_найкращій"]
        m2_in_best = cache_final["m2_у_найкращій"]
        log.info("Крок 6: ЗАВАНТАЖЕНО З КЕШУ (ЗСК_повне=%.6e)", sse_full)
    else:
        log.info("─" * 50)
        log.info(
            "Крок 6: Тренування фінальної моделі на ВСІХ %d спостереженнях",
            n_samples,
        )

        best_feature_names = [feature_names[i] for i in best_feature_indices]
        x_best_full = x_raw[:, best_feature_indices]

        final_model = fit_lsm(x_best_full, y, alpha=0.1)
        final_coefs = final_model.coef_
        final_intercept = float(final_model.intercept_)

        # ЗСК на ВСІХ даних (єдина метрика)
        y_pred_full = final_model.predict(x_best_full)
        sse_full = float(np.sum((y - y_pred_full) ** 2))

        m1_in_best = [n for n in m1_final if n in best_feature_names]
        m2_in_best = [n for n in best_feature_names if n not in m1_final]

        _save_cache("крок6_фінальна_модель", {
            "імена_ознак": best_feature_names,
            "індекси_ознак": best_feature_indices,
            "коефіцієнти": final_coefs,
            "зміщення": final_intercept,
            "зск_повне": sse_full,
            "m1_у_найкращій": m1_in_best,
            "m2_у_найкращій": m2_in_best,
            "зск_перехресна": best_sse_cv,
        })

    # ── Виведення формули ──
    log.info("=" * 70)
    log.info("ФІНАЛЬНА ФОРМУЛА")
    log.info("=" * 70)
    log.info("")
    log.info("Найкраща формула = M1 + M2 + зміщення")
    log.info("")
    log.info("M1 (ядро — ознаки з перетину):")
    for name in m1_in_best:
        idx_in_best = best_feature_names.index(name)
        log.info("  %+.10e × %s", final_coefs[idx_in_best], name)
    log.info("")
    log.info("M2 (відібрані з комбінацій):")
    if m2_in_best:
        for name in m2_in_best:
            idx_in_best = best_feature_names.index(name)
            log.info("  %+.10e × %s", final_coefs[idx_in_best], name)
    else:
        log.info("  (немає)")
    log.info("")
    log.info("Зміщення (вільний член): %+.10e", final_intercept)
    log.info("")
    log.info("ЗСК (повні дані): %.6e", sse_full)

    # ── Збереження бінарного файлу моделі ──
    model_binary = {
        "ознаки": best_feature_names,
        "ознаки_m1": m1_in_best,
        "ознаки_m2": m2_in_best,
        "коефіцієнти": final_coefs,
        "зміщення": final_intercept,
        "зск_повне": sse_full,
        "зск_перехресна": best_sse_cv,
        "індекси_ознак": best_feature_indices,
    }
    bin_path = os.path.join(PLOTS_DIR, "фінальна_модель.bin")
    joblib.dump(model_binary, bin_path)
    log.info("Бінарний файл моделі збережено: %s", bin_path)

    # ── Графіки ──
    log.info("─" * 50)
    log.info("Генерація наукових графіків")

    # --- Функція передбачення: НАЙКРАЩА формула ---
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

    # --- Функція передбачення: тільки M1 + зміщення (залишковий опис) ---
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

    # --- Функція передбачення: M1 + усі M2 + зміщення (надлишковий опис) ---
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
        df, predict_best, plot_dir=PLOTS_DIR, model_name="МГУА",
        predict_m1_only=predict_m1_only,
        predict_m1_m2_all=predict_m1_m2_all,
    )

    t_total = time.perf_counter() - t_total_start
    log.info("=" * 70)
    log.info("МГУА ЗАВЕРШЕНО. Загальний час: %.3f с", t_total)
    log.info("=" * 70)

    return best_feature_names, final_coefs, final_intercept, sse_full


# ═══════════════════════════ ГОЛОВНА ФУНКЦІЯ ══════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    log.info("Початок виконання")

    df = load_data()
    if df.empty:
        log.error('Дані не знайдено в директорії кешу "%s"', CACHE_DIR)
        return

    log.info("Завантажено %d спостережень", len(df))

    try:
        names, coefs, intercept, sse_full = run_gmdh(df)

        # Запис формули у файл
        lines = [
            "Найкраща формула = M1 + M2 + зміщення",
            "",
            "Повна формула:",
        ]
        for w, n in zip(coefs, names):
            lines.append(f"  {w:+.16e} × [{n}]")
        lines.append(f"  {intercept:+.16e}  (зміщення / вільний член)")
        lines.append("")
        lines.append(f"ЗСК (повні дані): {sse_full:.6e}")

        equation = "\n".join(lines)
        log.info("Фінальна модель:\n%s", equation)

        formula_path = os.path.join(PLOTS_DIR, "фінальна_формула.txt")
        with open(formula_path, "w", encoding="utf-8") as fh:
            fh.write(equation)
        log.info("Формулу записано до %s", formula_path)

    except Exception:
        log.exception("Критична помилка виконання")

    elapsed = time.perf_counter() - t0
    log.info("Загальний час виконання: %.3f с", elapsed)


if __name__ == "__main__":
    main()
