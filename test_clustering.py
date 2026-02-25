import logging
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd


def setup_logger():
    logger = logging.getLogger("gmdh_check")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(sh)
    return logger


log = setup_logger()

FINAL_MODEL = {
    "M1": {"general": 1.0045105641951464e+00},
    "M2": {
        "(poly_mn)^2": 1.4658495770918936e-03,
        "poly_mn * adler_megiddo": -1.6650481510065285e-10,
        "(general)^2": 5.6043683082312146e-13,
        "(adler_megiddo)^2": 8.9888833075594756e-18,
        "log_n_log_m": -2.4846699797334588e+08,
        "general * log_n_log_m": -7.9314261089856578e-02
    },
    "intercept": 2.8891117719259033e+09
}


def _f_refined(m, n): return (m ** 3) * (n ** 2)


def _f_smoothed(m, n): return m * (n ** 5) * np.log(np.where(n > 1, n, 1.1))


def _f_poly_mn(m, n): return m * n


def _safe_log(n): return np.where(n > 1, np.log(n), 0.0)


def _f_general(m, n): return 0.63 * m ** 2.96 * n ** 0.02 * _safe_log(n) ** 1.62 + 4.04 * m ** (-4.11) * n ** 2.92


def _f_adler_megiddo(_, n): return n ** 4


def _f_log_n_log_m(m, n): return np.log(np.where(m > 1, m, 1.1)) + np.log(np.where(n > 1, n, 1.1))


BASIS_FUNCTIONS = {
    "m3n2": _f_refined, "mn5lnn": _f_smoothed, "poly_mn": _f_poly_mn,
    "general": _f_general, "adler_megiddo": _f_adler_megiddo, "log_n_log_m": _f_log_n_log_m,
}


def build_full_features(df: pd.DataFrame):
    m, n = df["m"].values, df["n"].values
    base = {name: func(m, n) for name, func in BASIS_FUNCTIONS.items()}
    data = dict(base)
    for name in base: data[f"({name})^2"] = base[name] ** 2
    for a, b in combinations(base, 2): data[f"{a} * {b}"] = base[a] * base[b]
    return pd.DataFrame(data), list(data.keys())


def load_data():
    all_data = []
    if not os.path.exists("cache"): return pd.DataFrame()
    for f in os.listdir("cache"):
        if not f.endswith(".pkl"): continue
        try:
            with open(os.path.join("cache", f), "rb") as fh:
                data = pickle.load(fh)
                if isinstance(data, list) and len(data) > 3:
                    all_data.extend(data)
                elif isinstance(data, (list, tuple)) and len(data) == 3:
                    all_data.append(list(data))
        except:
            pass
    df = pd.DataFrame(all_data, columns=["m", "n", "ops"]).drop_duplicates().astype(float)
    return df[df["ops"] > 0]


def cluster_half(x_data, y_data, feature_names, alpha=0.1):
    stds = np.std(x_data, axis=0)
    stds[stds < 1e-15] = 1e-15
    x_norm = (x_data - np.mean(x_data, axis=0)) / stds

    A = x_norm.T @ x_norm + np.eye(len(feature_names)) * alpha
    std_coefs = np.linalg.solve(A, x_norm.T @ y_data)

    abs_std = np.abs(std_coefs)
    order = np.argsort(-abs_std)
    sorted_abs = abs_std[order]
    sorted_names = [feature_names[i] for i in order]

    m1_names = [sorted_names[0]]
    smallest = sorted_abs[-1]

    for i in range(1, len(sorted_abs)):
        avg_m1 = np.mean(sorted_abs[:i])
        if (avg_m1 - sorted_abs[i]) < (sorted_abs[i] - smallest):
            m1_names.append(sorted_names[i])
        else:
            break

    return m1_names


def main():
    log.info("Завантаження даних...")
    df = load_data()
    if df.empty:
        log.error("Дані відсутні.")
        return

    x_df, feature_names = build_full_features(df)
    X_raw, y = x_df.values, df["ops"].values

    log.info(f"Згенеровано початкових ознак: {len(feature_names)}")

    rng = np.random.RandomState(1810)
    indices = rng.permutation(len(y))
    idx_1, idx_2 = indices[:len(y) // 2], indices[len(y) // 2:]

    m1_1 = cluster_half(X_raw[idx_1], y[idx_1], feature_names)
    m1_2 = cluster_half(X_raw[idx_2], y[idx_2], feature_names)
    m1_final = sorted(set(m1_1) & set(m1_2))

    log.info(f"M1 з першої половини: {m1_1}")
    log.info(f"M1 з другої половини: {m1_2}")
    log.info(f"Перетин (Фінальний M1): {m1_final}")

    log.info("Формування підсумкової моделі:")
    log.info("y = M1 + M2 + Зсув")

    log.info("[M1] Залишковий опис:")
    for feat, coef in FINAL_MODEL["M1"].items():
        log.info(f"  {coef:+.4g} * {feat}")

    log.info("[M2] Додаткові терми:")
    for feat, coef in FINAL_MODEL["M2"].items():
        log.info(f"  {coef:+.4g} * {feat}")

    log.info(f"Зсув (Intercept): {FINAL_MODEL['intercept']:+.4g}")


if __name__ == "__main__":
    main()
