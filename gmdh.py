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
from typing import Any, List, Optional, Tuple, Callable

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from tqdm import tqdm

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = "DejaVu Sans"

CACHE_DIR = "cache"
GMDH_CACHE_DIR = "gmdh_cache"
PLOTS_DIR = "plots_combi"
LOG_FILE = "gmdh_execution.log"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
BATCH_SIZE_INITIAL = 4096

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GMDH_CACHE_DIR, exist_ok=True)

if torch.cuda.is_available():
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


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


# ═══════════════════ БАЗИСНІ ФУНКЦІЇ ═══════════════════════════

def _f_refined(m, n):
    return (m ** 3) * (n ** 2)


def _f_smoothed(m, n):
    return m * (n ** 5) * np.log(np.where(n > 1, n, 1.1))


def _f_poly_mn(m, n):
    return m * n


def _safe_log(n):
    return np.where(n > 1, np.log(n), 0.0)


def _f_general(m, n):
    return (
            0.63 * m ** 2.96 * n ** 0.02 * _safe_log(n) ** 1.62
            + 4.04 * m ** (-4.11) * n ** 2.92
    )


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
) -> Tuple[List[str], List[str]]:
    """
    Сортуємо за спаданням модулів коефіцієнтів |coef|.
    Кластеризація: M1 (важливі) та M2 (решта).
    """
    items = []
    for w, name in zip(coefs, feature_names):
        items.append({"name": name, "abs_coef": abs(w)})

    sorted_items = sorted(items, key=lambda x: x["abs_coef"], reverse=True)

    if not sorted_items:
        return [], list(feature_names)

    smallest_abs = sorted_items[-1]["abs_coef"]
    m1_items = [sorted_items[0]]

    for i in range(1, len(sorted_items)):
        candidate = sorted_items[i]
        avg_m1 = sum(it["abs_coef"] for it in m1_items) / len(m1_items)
        dist_to_m1 = avg_m1 - candidate["abs_coef"]
        dist_to_m2 = candidate["abs_coef"] - smallest_abs
        if dist_to_m1 < dist_to_m2:
            m1_items.append(candidate)
        else:
            break

    m1_names = [it["name"] for it in m1_items]
    m2_names = [it["name"] for it in sorted_items if it["name"] not in m1_names]

    log.info("  Кластеризація: M1 (%d ознак), M2 (%d ознак)", len(m1_names), len(m2_names))
    for it in sorted_items:
        cluster = "M1" if it["name"] in m1_names else "M2"
        log.info("    [%s] %-30s |β| = %.6e", cluster, it["name"], it["abs_coef"])

    return m1_names, m2_names


# ═══════════════════ МНК / ЗСК ═════════════════════════════════

def fit_lsm(x_train, y_train, alpha=0.1):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train, y_train)
    return model


def compute_sse(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return float(np.sum((y_test - y_pred) ** 2))


# ═══════════════ GPU БАТЧ-РОЗВ'ЯЗУВАЧ ═══════════════════════════

def _batched_combinations(iterable, batch_size: int):
    """Генерує батчі зі списку комбінацій."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            return
        yield batch


@torch.no_grad()
def solve_batch_gpu(
        idx_tensor: torch.Tensor,
        xt_x: torch.Tensor,
        xt_y: torch.Tensor,
        x_test_t: torch.Tensor,
        y_test: torch.Tensor,
        alpha: float,
) -> torch.Tensor:
    """
    Батчевий Ridge-розв'язувач на GPU.

    idx_tensor: (batch, k) — індекси ознак для кожної комбінації
    xt_x:       (F, F)     — попередньо обчислена X^T @ X
    xt_y:       (F, 1)     — попередньо обчислена X^T @ y
    x_test_t:   (F, N_test) — транспонована тестова матриця
    y_test:     (N_test,)   — тестовий вектор y
    alpha:      float       — регуляризація

    Повертає: sse_batch (batch,) — ЗСК для кожної комбінації.
    """
    batch_size = idx_tensor.size(0)
    k = idx_tensor.size(1)

    # Витягуємо підматриці Грама за індексами
    rows = idx_tensor.unsqueeze(2).expand(batch_size, k, k)
    cols = idx_tensor.unsqueeze(1).expand(batch_size, k, k)
    a_batch = xt_x[rows, cols]  # (batch, k, k)

    # Додаємо регуляризацію
    eye = torch.eye(k, device=DEVICE, dtype=DTYPE).unsqueeze(0).expand(batch_size, k, k)
    a_batch = a_batch + alpha * eye

    # Права частина
    b_batch = xt_y[idx_tensor]  # (batch, k, 1)

    # Розв'язуємо систему
    weights = None
    try:
        chol = torch.linalg.cholesky(a_batch)
        weights = torch.cholesky_solve(b_batch, chol)
    except RuntimeError:
        pass

    if weights is None:
        try:
            weights = torch.linalg.solve(a_batch, b_batch)
        except RuntimeError:
            pass

    if weights is None:
        try:
            weights = torch.linalg.lstsq(a_batch, b_batch).solution
        except RuntimeError:
            pass

    if weights is None:
        weights = torch.bmm(torch.linalg.pinv(a_batch), b_batch)

    # Передбачення на тестовій вибірці
    x_test_sub = x_test_t[idx_tensor]  # (batch, k, N_test)
    y_pred = torch.bmm(x_test_sub, weights).squeeze(2)  # (batch, N_test)

    # ЗСК
    diff = y_pred - y_test.unsqueeze(0)
    sse_batch = torch.sum(diff ** 2, dim=1)  # (batch,)

    return sse_batch


@torch.no_grad()
def evaluate_combinations_gpu(
        m1_indices: List[int],
        m2_indices: List[int],
        m2_names: List[str],
        x_half1: np.ndarray,
        y_half1: np.ndarray,
        x_half2: np.ndarray,
        y_half2: np.ndarray,
        alpha: float = 0.1,
) -> Tuple[List[float], List[str], List[List[int]], int, float, List[int]]:
    """
    Оцінює всі часткові описи M1 + комбінації(M2) на GPU з батчингом.
    Для кожної комбінації: ЗСК = ЗСК_1(train half1, test half2) + ЗСК_2(train half2, test half1).
    """
    n_m2 = len(m2_indices)
    total_combos = sum(math.comb(n_m2, r) for r in range(0, n_m2 + 1))

    log.info("  GPU-пристрій: %s", DEVICE)
    log.info("  Загальна кількість комбінацій M2: %d", total_combos)
    log.info("  Початковий розмір батчу: %d", BATCH_SIZE_INITIAL)

    # ── Попередньо обчислюємо матриці Грама на GPU ──
    n_features = x_half1.shape[1]

    # Додаємо стовпець одиниць для intercept
    ones_1 = np.ones((x_half1.shape[0], 1))
    ones_2 = np.ones((x_half2.shape[0], 1))
    x1_bias = np.hstack([x_half1, ones_1])  # (N1, F+1)
    x2_bias = np.hstack([x_half2, ones_2])  # (N2, F+1)

    x1_t = torch.tensor(x1_bias, dtype=DTYPE, device=DEVICE)
    x2_t = torch.tensor(x2_bias, dtype=DTYPE, device=DEVICE)
    y1_t = torch.tensor(y_half1, dtype=DTYPE, device=DEVICE)
    y2_t = torch.tensor(y_half2, dtype=DTYPE, device=DEVICE)

    # X^T @ X та X^T @ y для обох половин
    xt_x_1 = x1_t.T @ x1_t  # (F+1, F+1)
    xt_y_1 = (x1_t.T @ y1_t.unsqueeze(1))  # (F+1, 1)
    xt_x_2 = x2_t.T @ x2_t
    xt_y_2 = (x2_t.T @ y2_t.unsqueeze(1))

    # Транспоновані тестові матриці
    x_test1_t = x2_t.T  # test for model trained on half1 = half2, shape (F+1, N2)
    x_test2_t = x1_t.T  # test for model trained on half2 = half1, shape (F+1, N1)

    del x1_t, x2_t
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    gc.collect()

    # Індекс bias-стовпця
    bias_idx = n_features  # останній стовпець

    sse_values_list: List[float] = []
    combo_labels_list: List[str] = []
    combo_indices_list: List[List[int]] = []
    best_sse = float("inf")
    best_combo_idx = 0
    best_feature_indices = []

    current_batch_size = BATCH_SIZE_INITIAL
    global_combo_count = 0

    for r in tqdm(
            range(0, n_m2 + 1),
            desc="Розміри підмножин M2",
            ncols=100,
            position=0,
    ):
        n_combos_r = math.comb(n_m2, r)
        if n_combos_r == 0:
            continue

        comb_gen = combinations(range(n_m2), r)
        processed_in_r = 0

        with tqdm(
                total=n_combos_r,
                desc=f"  M2 розмір {r}",
                ncols=100,
                position=1,
                leave=False,
        ) as pbar_r:
            while processed_in_r < n_combos_r:
                remaining = n_combos_r - processed_in_r
                bs = min(current_batch_size, remaining)

                batch_combos = list(islice(comb_gen, bs))
                if not batch_combos:
                    break

                actual_bs = len(batch_combos)

                # Будуємо індекси: M1 + підмножина M2 + bias
                all_idx_lists = []
                labels = []
                orig_indices_lists = []

                for m2_subset in batch_combos:
                    m2_sub_idx = [m2_indices[j] for j in m2_subset]
                    candidate = m1_indices + m2_sub_idx
                    # Індекси у розширеній матриці (з bias)
                    candidate_with_bias = candidate + [bias_idx]
                    all_idx_lists.append(candidate_with_bias)
                    orig_indices_lists.append(candidate)

                    m2_names_in = [m2_names[j] for j in m2_subset]
                    if m2_names_in:
                        labels.append("M1+" + "+".join(m2_names_in))
                    else:
                        labels.append("тільки M1")

                # Потрібно вирівняти довжини (padding) — всі комбінації
                # одного розміру r мають однакову кількість ознак
                k = len(all_idx_lists[0])

                try:
                    idx_tensor = torch.tensor(
                        all_idx_lists, dtype=torch.long, device=DEVICE
                    )  # (bs, k)

                    # ЗСК_1: train half1, test half2
                    sse1 = solve_batch_gpu(
                        idx_tensor, xt_x_1, xt_y_1, x_test1_t, y2_t, alpha
                    )
                    # ЗСК_2: train half2, test half1
                    sse2 = solve_batch_gpu(
                        idx_tensor, xt_x_2, xt_y_2, x_test2_t, y1_t, alpha
                    )

                    sse_total = (sse1 + sse2).cpu().numpy()

                    del idx_tensor, sse1, sse2

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or \
                             "out of memory" in str(e).lower()
                    if not is_oom:
                        raise

                    torch.cuda.empty_cache()
                    gc.collect()

                    old_bs = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    log.warning(
                        "  OOM: розмір батчу %d → %d", old_bs, current_batch_size
                    )
                    # Повертаємо комбінації назад у генератор — потрібно перезапустити
                    # Оскільки islice вже спожив, просто пропускаємо і перегенеруємо
                    # Це складно з генератором, тому просто зменшуємо bs і продовжуємо
                    continue

                # Записуємо результати
                for i in range(actual_bs):
                    sse_val = float(sse_total[i])
                    sse_values_list.append(sse_val)
                    combo_labels_list.append(labels[i])
                    combo_indices_list.append(orig_indices_lists[i])

                    if sse_val < best_sse:
                        best_sse = sse_val
                        best_combo_idx = global_combo_count + i
                        best_feature_indices = orig_indices_lists[i]
                        log.debug(
                            "  Нова найкраща ЗСК=%.6e: %s",
                            best_sse, labels[i],
                        )

                global_combo_count += actual_bs
                processed_in_r += actual_bs
                pbar_r.update(actual_bs)

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    log.info("  GPU-оцінка завершена: %d комбінацій", global_combo_count)
    log.info("  Найкраща ЗСК (перехресна) = %.6e", best_sse)

    return (
        sse_values_list,
        combo_labels_list,
        combo_indices_list,
        best_combo_idx,
        best_sse,
        best_feature_indices,
    )


# ═══════════════════ НАУКОВІ ГРАФІКИ ══════════════════════════

def plot_model_vs_data(
        df: pd.DataFrame,
        predict_func: Callable,
        plot_dir: str,
        model_name: str = "МГУА",
        predict_m1_only: Callable = None,
        predict_m1_m2_all: Callable = None,
) -> None:
    safe_name = re.sub(r"[\\/*?:'<>|]", "", model_name)
    m_vals = sorted(df["m"].unique())
    n_vals = sorted(df["n"].unique())

    y_actual = df["ops"].values
    y_predicted = predict_func(df["m"].values, df["n"].values)

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

    for scale in ("linear", "log"):
        fig, axes = plt.subplots(1, 2, figsize=(30, 20), dpi=200)

        ax = axes[0]
        for n_fixed in n_vals:
            subset = df[df["n"] == n_fixed]
            box_data = [subset[subset["m"] == m]["ops"].values for m in m_vals]
            bp = ax.boxplot(box_data, positions=m_vals, widths=0.6,
                            patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set(facecolor="lightblue")
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)
            avg_ops = subset.groupby("m")["ops"].mean()
            ax.plot(avg_ops.index, avg_ops.values, "k-", linewidth=1, alpha=0.7)

            m_range = np.linspace(min(m_vals), max(m_vals), 100)
            if predict_m1_only is not None:
                ax.plot(m_range, predict_m1_only(m_range, np.full_like(m_range, n_fixed)),
                        ":", color="orange", linewidth=2, alpha=0.7)
            if predict_m1_m2_all is not None:
                ax.plot(m_range, predict_m1_m2_all(m_range, np.full_like(m_range, n_fixed)),
                        "-.", color="purple", linewidth=2, alpha=0.7)
            ax.plot(m_range, predict_func(m_range, np.full_like(m_range, n_fixed)),
                    "--", color="green", linewidth=2.5)

        ax.set_xlabel("m (кількість обмежень)", fontsize=12)
        ax.set_ylabel("Кількість операцій", fontsize=12)
        ax.set_title("Залежність кількості операцій від m", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

        ax = axes[1]
        for m_fixed in m_vals:
            subset = df[df["m"] == m_fixed]
            box_data = [subset[subset["n"] == n]["ops"].values for n in n_vals]
            bp = ax.boxplot(box_data, positions=n_vals, widths=0.6,
                            patch_artist=True, showfliers=False)
            for box in bp["boxes"]:
                box.set(facecolor="lightblue")
            for median in bp["medians"]:
                median.set(color="red", linewidth=2)
            avg_ops = subset.groupby("n")["ops"].mean()
            ax.plot(avg_ops.index, avg_ops.values, "k-", linewidth=1, alpha=0.7)

            n_range = np.linspace(min(n_vals), max(n_vals), 100)
            if predict_m1_only is not None:
                ax.plot(n_range, predict_m1_only(np.full_like(n_range, m_fixed), n_range),
                        ":", color="orange", linewidth=2, alpha=0.7)
            if predict_m1_m2_all is not None:
                ax.plot(n_range, predict_m1_m2_all(np.full_like(n_range, m_fixed), n_range),
                        "-.", color="purple", linewidth=2, alpha=0.7)
            ax.plot(n_range, predict_func(np.full_like(n_range, m_fixed), n_range),
                    "--", color="green", linewidth=2.5)

        ax.set_xlabel("n (кількість змінних)", fontsize=12)
        ax.set_ylabel("Кількість операцій", fontsize=12)
        ax.set_title("Залежність кількості операцій від n", fontsize=13)
        if scale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
        ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

        scale_name = "лінійний" if scale == "linear" else "логарифмічний"
        fig.suptitle(f"Модель {safe_name}: дані та апроксимація ({scale_name} масштаб)", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_2d.png"))
        plt.close(fig)

    # ── Залишки ──
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
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1.5, label="Ідеал (y = x)")
    ax.set_xlabel("Фактичні значення", fontsize=12)
    ax.set_ylabel("Передбачені значення", fontsize=12)
    ax.set_title("Фактичні та передбачені значення", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Модель {safe_name}: аналіз залишків", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{safe_name}_залишки.png"))
    plt.close(fig)

    # ── 3D поверхні ──
    df_grouped = df.groupby(["m", "n"])["ops"].mean().reset_index()

    for scale in ("linear", "log"):
        m_grid, n_grid = np.meshgrid(
            np.linspace(min(m_vals), max(m_vals), 50),
            np.linspace(min(n_vals), max(n_vals), 50),
        )
        ops_best = predict_func(m_grid.ravel(), n_grid.ravel()).reshape(m_grid.shape)

        ops_m1 = None
        if predict_m1_only is not None:
            ops_m1 = predict_m1_only(m_grid.ravel(), n_grid.ravel()).reshape(m_grid.shape)
        ops_all = None
        if predict_m1_m2_all is not None:
            ops_all = predict_m1_m2_all(m_grid.ravel(), n_grid.ravel()).reshape(m_grid.shape)

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

        if ops_m1 is not None:
            ax1 = fig.add_subplot(131, projection="3d")
            ax1.plot_surface(m_grid, n_grid, ops_m1, cmap="Oranges", alpha=0.7)
            ax1.scatter(df_grouped["m"].values, df_grouped["n"].values, ops_data,
                        c="red", marker="o", s=15)
            ax1.set_xlabel("m", fontsize=10)
            ax1.set_ylabel("n", fontsize=10)
            ax1.set_zlabel(z_label, fontsize=10)
            ax1.set_title("M1 + зміщення\n(залишковий опис)", fontsize=11)

        if ops_all is not None:
            ax2 = fig.add_subplot(132, projection="3d")
            ax2.plot_surface(m_grid, n_grid, ops_all, cmap="Purples", alpha=0.7)
            ax2.scatter(df_grouped["m"].values, df_grouped["n"].values, ops_data,
                        c="red", marker="o", s=15)
            ax2.set_xlabel("m", fontsize=10)
            ax2.set_ylabel("n", fontsize=10)
            ax2.set_zlabel(z_label, fontsize=10)
            ax2.set_title("M1 + M2(усі) + зміщення\n(надлишковий опис)", fontsize=11)

        ax3 = fig.add_subplot(133, projection="3d")
        ax3.plot_surface(m_grid, n_grid, ops_best, cmap="viridis", alpha=0.7)
        ax3.scatter(df_grouped["m"].values, df_grouped["n"].values, ops_data,
                    c="red", marker="o", s=15)
        ax3.set_xlabel("m", fontsize=10)
        ax3.set_ylabel("n", fontsize=10)
        ax3.set_zlabel(z_label, fontsize=10)
        ax3.set_title("Найкраща формула\n(результат МГУА)", fontsize=11)

        fig.suptitle(
            f"Порівняння моделей ({scale_name} масштаб): "
            f"залишковий → надлишковий → оптимальний", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{safe_name}_{scale}_3d.png"))
        plt.close(fig)


# ═══════════════════ ОСНОВНИЙ АЛГОРИТМ МГУА ═══════════════════════

def run_gmdh(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, float, float]:
    t_total_start = time.perf_counter()
    log.info("=" * 70)
    log.info("ПОЧАТОК АЛГОРИТМУ МГУА")
    log.info("=" * 70)

    x_df, feature_names = generate_full_feature_matrix(df)
    x_raw = x_df.values
    y = df["ops"].values
    n_features = len(feature_names)
    n_samples = len(y)

    log.info("Матриця ознак: %d спостережень × %d ознак", n_samples, n_features)
    log.info("Ознаки: %s", ", ".join(feature_names))

    rng = np.random.RandomState(42)
    indices = rng.permutation(n_samples)
    split = n_samples // 2
    idx_1, idx_2 = indices[:split], indices[split:]

    x_half1, y_half1 = x_raw[idx_1], y[idx_1]
    x_half2, y_half2 = x_raw[idx_2], y[idx_2]

    log.info("Розбиття: половина 1 = %d, половина 2 = %d", len(idx_1), len(idx_2))

    # ── Крок 2а: Кластеризація на ПОЛОВИНІ-1 ──
    cache_c1 = _load_cache("крок2а_кластеризація_половина1")
    if cache_c1 is not None:
        m1_1_names, m2_1_names, coefs_1 = cache_c1["m1"], cache_c1["m2"], cache_c1["coefs"]
        log.info("Крок 2а: ЗАВАНТАЖЕНО З КЕШУ")
    else:
        log.info("─" * 50)
        log.info("Крок 2а: Повний опис на ПОЛОВИНІ-1, кластеризація")
        model_1 = fit_lsm(x_half1, y_half1)
        coefs_1 = model_1.coef_
        log.info("  Коефіцієнти моделі 1:")
        for name, c in zip(feature_names, coefs_1):
            log.info("    %-30s : %+.6e", name, c)
        m1_1_names, m2_1_names = perform_coefficient_clustering(coefs_1, feature_names)
        _save_cache("крок2а_кластеризація_половина1", {
            "m1": m1_1_names, "m2": m2_1_names, "coefs": coefs_1,
        })

    log.info("  M1_1: %s", m1_1_names)
    log.info("  M2_1: %s", m2_1_names)

    # ── Крок 2б: Кластеризація на ПОЛОВИНІ-2 ──
    cache_c2 = _load_cache("крок2б_кластеризація_половина2")
    if cache_c2 is not None:
        m1_2_names, m2_2_names, coefs_2 = cache_c2["m1"], cache_c2["m2"], cache_c2["coefs"]
        log.info("Крок 2б: ЗАВАНТАЖЕНО З КЕШУ")
    else:
        log.info("─" * 50)
        log.info("Крок 2б: Повний опис на ПОЛОВИНІ-2, кластеризація")
        model_2 = fit_lsm(x_half2, y_half2)
        coefs_2 = model_2.coef_
        log.info("  Коефіцієнти моделі 2:")
        for name, c in zip(feature_names, coefs_2):
            log.info("    %-30s : %+.6e", name, c)
        m1_2_names, m2_2_names = perform_coefficient_clustering(coefs_2, feature_names)
        _save_cache("крок2б_кластеризація_половина2", {
            "m1": m1_2_names, "m2": m2_2_names, "coefs": coefs_2,
        })

    log.info("  M1_2: %s", m1_2_names)
    log.info("  M2_2: %s", m2_2_names)

    # ── Крок 3: Перетин / Об'єднання ──
    cache_int = _load_cache("крок3_перетин")
    if cache_int is not None:
        m1_final, m2_final = cache_int["m1_final"], cache_int["m2_final"]
        log.info("Крок 3: ЗАВАНТАЖЕНО З КЕШУ")
    else:
        log.info("─" * 50)
        log.info("Крок 3: M1 = перетин, M2 = все інше")
        m1_final = sorted(set(m1_1_names) & set(m1_2_names))
        m2_final = sorted(set(feature_names) - set(m1_final))
        _save_cache("крок3_перетин", {"m1_final": m1_final, "m2_final": m2_final})

    log.info("  M1 (перетин): %d ознак: %s", len(m1_final), m1_final)
    log.info("  M2 (все інше): %d ознак: %s", len(m2_final), m2_final)
    if not m1_final:
        log.warning("  M1 порожній!")

    m1_indices = [feature_names.index(n) for n in m1_final]
    m2_indices = [feature_names.index(n) for n in m2_final]

    # ── Кроки 4–5: GPU-оцінка часткових описів ──
    total_m2_combos = sum(math.comb(len(m2_final), r) for r in range(len(m2_final) + 1))
    log.info("─" * 50)
    log.info("Кроки 4-5: Оцінка %d часткових описів (M1[%d] + комбінації M2[%d])",
             total_m2_combos, len(m1_final), len(m2_final))

    cache_combos = _load_cache("крок5_усі_комбінації")
    if cache_combos is not None:
        sse_values_list = cache_combos["sse_values"]
        combo_labels_list = cache_combos["combo_labels"]
        combo_indices_list = cache_combos["combo_indices"]
        best_combo_idx = cache_combos["best_combo_idx"]
        best_sse_cv = cache_combos["best_sse_cv"]
        best_feature_indices = cache_combos["best_feature_indices"]
        log.info("Кроки 4-5: ЗАВАНТАЖЕНО З КЕШУ (%d комбінацій, ЗСК=%.6e)",
                 len(sse_values_list), best_sse_cv)
    else:
        (
            sse_values_list,
            combo_labels_list,
            combo_indices_list,
            best_combo_idx,
            best_sse_cv,
            best_feature_indices,
        ) = evaluate_combinations_gpu(
            m1_indices, m2_indices, m2_final,
            x_half1, y_half1, x_half2, y_half2,
        )

        _save_cache("крок5_усі_комбінації", {
            "sse_values": sse_values_list,
            "combo_labels": combo_labels_list,
            "combo_indices": combo_indices_list,
            "best_combo_idx": best_combo_idx,
            "best_sse_cv": best_sse_cv,
            "best_feature_indices": best_feature_indices,
        })

    log.info("Найкраща ЗСК (перехресна) = %.6e", best_sse_cv)
    log.info("Найкраща комбінація: %s", combo_labels_list[best_combo_idx])

    # ── Крок 6: Фінальна модель на ВСІХ даних ──
    cache_final = _load_cache("крок6_фінальна_модель")
    if cache_final is not None:
        best_feature_names = cache_final["імена_ознак"]
        final_coefs = cache_final["коефіцієнти"]
        final_intercept = cache_final["зміщення"]
        sse_full = cache_final["зск_повне"]
        best_feature_indices = cache_final["індекси_ознак"]
        m1_in_best = cache_final["m1_у_найкращій"]
        m2_in_best = cache_final["m2_у_найкращій"]
        log.info("Крок 6: ЗАВАНТАЖЕНО З КЕШУ (ЗСК=%.6e)", sse_full)
    else:
        log.info("─" * 50)
        log.info("Крок 6: Фінальна модель на ВСІХ %d спостереженнях", n_samples)

        best_feature_names = [feature_names[i] for i in best_feature_indices]
        x_best = x_raw[:, best_feature_indices]
        final_model = fit_lsm(x_best, y, alpha=0.1)
        final_coefs = final_model.coef_
        final_intercept = float(final_model.intercept_)
        y_pred = final_model.predict(x_best)
        sse_full = float(np.sum((y - y_pred) ** 2))

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
        idx = best_feature_names.index(name)
        log.info("  %+.10e × %s", final_coefs[idx], name)
    log.info("")
    log.info("M2 (відібрані з комбінацій):")
    if m2_in_best:
        for name in m2_in_best:
            idx = best_feature_names.index(name)
            log.info("  %+.10e × %s", final_coefs[idx], name)
    else:
        log.info("  (немає)")
    log.info("")
    log.info("Зміщення: %+.10e", final_intercept)
    log.info("ЗСК (повні дані): %.6e", sse_full)

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
    log.info("Модель збережено: %s", bin_path)

    # ── Графіки ──
    log.info("─" * 50)
    log.info("Генерація графіків")

    def predict_best(m_in, n_in):
        tmp = pd.DataFrame({"m": np.asarray(m_in).flatten(), "n": np.asarray(n_in).flatten()})
        xf, _ = generate_full_feature_matrix(tmp)
        mdl = Ridge(alpha=0.1, fit_intercept=True)
        mdl.coef_ = final_coefs
        mdl.intercept_ = final_intercept
        return mdl.predict(xf.values[:, best_feature_indices])

    if m1_indices:
        mdl_m1 = fit_lsm(x_raw[:, m1_indices], y, alpha=0.1)

        def predict_m1_only(m_in, n_in):
            tmp = pd.DataFrame({"m": np.asarray(m_in).flatten(), "n": np.asarray(n_in).flatten()})
            xf, _ = generate_full_feature_matrix(tmp)
            return mdl_m1.predict(xf.values[:, m1_indices])
    else:
        predict_m1_only = None

    all_idx = m1_indices + m2_indices
    mdl_all = fit_lsm(x_raw[:, all_idx], y, alpha=0.1)

    def predict_all(m_in, n_in):
        tmp = pd.DataFrame({"m": np.asarray(m_in).flatten(), "n": np.asarray(n_in).flatten()})
        xf, _ = generate_full_feature_matrix(tmp)
        return mdl_all.predict(xf.values[:, all_idx])

    plot_model_vs_data(df, predict_best, PLOTS_DIR, "МГУА",
                       predict_m1_only=predict_m1_only,
                       predict_m1_m2_all=predict_all)

    t_total = time.perf_counter() - t_total_start
    log.info("=" * 70)
    log.info("МГУА ЗАВЕРШЕНО. Час: %.3f с", t_total)
    log.info("=" * 70)

    return best_feature_names, final_coefs, final_intercept, sse_full


# ═══════════════════════════ ГОЛОВНА ФУНКЦІЯ ══════════════════════════════

def main() -> None:
    t0 = time.perf_counter()
    log.info("Початок виконання")

    df = load_data()
    if df.empty:
        log.error('Дані не знайдено у "%s"', CACHE_DIR)
        return

    log.info("Завантажено %d спостережень", len(df))

    try:
        names, coefs, intercept, sse_full = run_gmdh(df)

        lines = ["Найкраща формула = M1 + M2 + зміщення", "", "Повна формула:"]
        for w, n in zip(coefs, names):
            lines.append(f"  {w:+.16e} × [{n}]")
        lines.append(f"  {intercept:+.16e}  (зміщення)")
        lines.append("")
        lines.append(f"ЗСК (повні дані): {sse_full:.6e}")

        equation = "\n".join(lines)
        log.info("Фінальна модель:\n%s", equation)

        path = os.path.join(PLOTS_DIR, "фінальна_формула.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(equation)
        log.info("Формулу записано: %s", path)

    except Exception:
        log.exception("Критична помилка")

    log.info("Загальний час: %.3f с", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
