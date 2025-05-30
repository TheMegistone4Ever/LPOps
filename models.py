from typing import Tuple, Callable, List

from numpy import ndarray, log, where, sum as np_sum


def borgwardt_model(X: Tuple[ndarray, ndarray], a: float) -> ndarray:
    """Borgwardt's Model: O(a * m^2 * n^(1/(m-1))) -> O(a * m^3 * n) when m scales with n."""

    m, n = X
    return a * m ** 3 * n


def smoothed_analysis_model(X: Tuple[ndarray, ndarray], a: float, b: float) -> ndarray:
    """Smoothed Analysis Model: O(mn^5 * log^b(n))."""

    m, n = X
    return a * m * n ** 5 * log(n) ** b


def adler_megiddo_model(X: Tuple[ndarray, ndarray], a: float) -> ndarray:
    """Adler-Megiddo Model: O(a * n^4)."""

    m, n = X
    return a * n ** 4


def refined_borgwardt_model(X: Tuple[ndarray, ndarray], a: float) -> ndarray:
    """Refined bound: O(a * m^3 * n^2)."""

    m, n = X
    return a * m ** 3 * n ** 2


def general_model(X: Tuple[ndarray, ndarray], a: float, b: float, c: float) -> ndarray:
    """General model O(a * m^b * n^c)."""

    m, n = X
    return a * m ** b * n ** c


def general_model_log(X: Tuple[ndarray, ndarray], a: float, b: float, c: float, d: float) -> ndarray:
    """General model with log terms: O(a * m^b * n^c * log(n)^d)."""

    m, n = X
    log_n = where(n > 1, log(n), 0)
    return a * (m ** b) * (n ** c) * log_n ** d


def general_model_mixed(X: Tuple[ndarray, ndarray], a: float, b: float, c: float, d: float, e: float,
                        f: float, g: float) -> ndarray:
    """General model with mixed terms: O(a * m^b * n^c * log(n)^d + e * m^f * n^g)."""

    m, n = X
    log_n = where(n > 1, log(n), 0)
    return a * (m ** b) * (n ** c) * (log_n ** d) + e * (m ** f) * (n ** g)


def weighted_ensemble_model(X: Tuple[ndarray, ndarray],
                            params_tuple: List[Tuple[Callable[..., ndarray], float, List[float]]]) -> ndarray:
    """Weighted ensemble model: O(sum_i(w_i * f_i(X, *params)))."""

    return np_sum([weight * model(X, *params) for model, weight, params in params_tuple], axis=0)
