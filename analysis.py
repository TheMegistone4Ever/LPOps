import logging
from time import time, ctime
from typing import List, Tuple

from numpy import ndarray, array, sum as np_sum
from numpy.random.mtrand import Sequence
from scipy.optimize import minimize

from cache_manager import CacheManager
from generator import generate_lp_problem
from models import (borgwardt_model, smoothed_analysis_model, adler_megiddo_model, refined_borgwardt_model,
                    general_model, general_model_log, general_model_mixed, weighted_ensemble_model)
from simplex import simplex_method
from utils import create_plots, calculate_loss, extract_plot_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analysis.log", mode="a"),
        logging.StreamHandler()
    ]
)

cache_manager = CacheManager()


def format_parameter_equation(model_name: str, params: ndarray) -> str:
    """Format model parameters as a readable equation string."""

    if model_name == "borgwardt_model":
        return f"{params[0]:.6f} * m^3 * n"
    elif model_name == "smoothed_analysis_model":
        return f"{params[0]:.6f} * m * n^5 * log(n)^{params[1]:.6f}"
    elif model_name == "adler_megiddo_model":
        return f"{params[0]:.6f} * n^4"
    elif model_name == "refined_borgwardt_model":
        return f"{params[0]:.6f} * m^3 * n^2"
    elif model_name == "general_model":
        return f"{params[0]:.6f} * m^{params[1]:.6f} * n^{params[2]:.6f}"
    elif model_name == "general_model_log":
        return f"{params[0]:.6f} * m^{params[1]:.6f} * n^{params[2]:.6f} * log(n)^{params[3]:.6f}"
    elif model_name == "general_model_mixed":
        return (f"{params[0]:.6f} * m^{params[1]:.6f} * n^{params[2]:.6f} * log(n)^{params[3]:.6f} + "
                f"{params[4]:.6f} * m^{params[5]:.6f} * n^{params[6]:.6f}")
    else:
        return f"Unknown model: {model_name} with params {params}"


def generate_sample(sample_name: str, m_range: Sequence[int], n_range: Sequence[int], num_runs: int,
                    factor: float = 1.0) -> List[Tuple]:
    """
    Generate a sample of LP problems and record their simplex performance,
    with caching for the simplex method results.  Saves only the data needed
    for plotting and analysis.

    Args:
        sample_name: Name of the sample for caching
        m_range: List of constraint counts to test
        n_range: List of variable counts to test
        num_runs: Number of problems to generate for each m, n pair
        factor: Scaling factor for problem generation

    Returns:
        List of tuples (m, n, operation_count)
    """

    sample_key = f"sample_{sample_name}_f{factor}_{len(m_range)}_{len(n_range)}_{num_runs}"
    cached_data = cache_manager.load(sample_key)

    if cached_data:
        logging.info(f"Loaded sample {sample_name} from cache")
        return cached_data

    logging.info(f"Generating sample {sample_name} with factor {factor}")
    sample_data = list()

    for m in m_range:
        for n in n_range:
            logging.info(f"Generating {num_runs} LP problems with m={m}, n={n}")
            for problem_index in range(num_runs):
                c, A, b, _ = generate_lp_problem(m, n, factor=factor)

                problem_key = f"simplex_{sample_name}_f{factor}_{m}_{n}_{problem_index}"
                cached_solution = cache_manager.load(problem_key)

                if cached_solution:
                    logging.info(f"Loaded simplex solution for m={m}, n={n}, problem {problem_index} from cache")
                    flops = cached_solution
                else:
                    logging.info(f"Solving simplex for m={m}, n={n}, problem {problem_index}")
                    cache_manager.save(problem_key, flops := simplex_method(c, A, b)[3])

                sample_data.append((m, n, flops))
                logging.info(f"Problem solved, {flops} operations")

    cache_manager.save(sample_key, sample_data)
    return sample_data


def fit_model(model_func, initial_params, sample_data, model_name, sample_name):
    """
    Fit a model to the sample data.

    Args:
        model_func: The model function to fit
        initial_params: Initial parameter values
        sample_data: The sample data to fit to
        model_name: Name of the model
        sample_name: Name of the sample

    Returns:
        Tuple of (optimal parameters, formula string)
    """

    cache_key = f"fit_{model_name}_{sample_name}_{len(sample_data)}"
    cached_result = cache_manager.load(cache_key)

    if cached_result:
        logging.info(f"Loaded fit for {model_name} on {sample_name} from cache")
        return cached_result["params"], cached_result["formula"]

    logging.info(f"Fitting {model_name} to sample {sample_name}")

    def loss_func(params):
        return calculate_loss(sample_data, model_func, params)

    result = minimize(loss_func, initial_params, method="Nelder-Mead")
    optimal_params = result.x

    formula = format_parameter_equation(model_func.__name__, optimal_params)

    plot_cache_key = f"plot_{model_name}_{sample_name}_{len(sample_data)}"
    if not cache_manager.load(plot_cache_key):
        logging.info(f"Creating and caching plots for {model_name} on {sample_name}")
        plot_data = extract_plot_data(sample_data, model_func, optimal_params)
        create_plots(plot_data, model_name, optimal_params, sample_id=sample_name)
        cache_manager.save(plot_cache_key, plot_data)
    else:
        logging.info(f"Plots for {model_name} on {sample_name} already cached")

    result_data = {"params": optimal_params.tolist(), "formula": formula}
    cache_manager.save(cache_key, result_data)

    return optimal_params, formula


def average_parameters(params1, params2):
    """
    Average two sets of parameters.

    Args:
        params1: First set of parameters
        params2: Second set of parameters

    Returns:
        Averaged parameters
    """

    return (array(params1) + array(params2)) / 2


def evaluate_model(model_func, params, sample_data, model_name, sample_name):
    """
    Evaluate a model on sample data and cache the result.

    Args:
        model_func: The model function to evaluate.
        params: Model parameters.
        sample_data: The sample data to evaluate on.
        model_name: Name of the model.
        sample_name: Name of the sample.

    Returns:
        Mean squared error.
    """

    params_str = "_".join(f"{p:.4f}" for p in params)
    cache_key = f"evaluate_{model_name}_{sample_name}_{params_str}"
    cached_result = cache_manager.load(cache_key)

    if cached_result:
        logging.info(f"Loaded evaluation for {model_name} on {sample_name} from cache")
        return cached_result["loss"]

    logging.info(f"Evaluating {model_name} on {sample_name}")
    loss = calculate_loss(sample_data, model_func, params)

    cache_manager.save(cache_key, {"loss": loss})
    return loss


def create_weighted_model(models_with_params, sample_data, model_name, sample_name):
    """
    Create a weighted ensemble model.

    Args:
        models_with_params: List of (model_func, initial_weight, params)
        sample_data: Sample data to fit to
        model_name: Name of the model
        sample_name: Name of the sample

    Returns:
        Tuple of (optimal weights and params, formula string)
    """

    cache_key = f"fit_weighted_{model_name}_{sample_name}_{len(sample_data)}"
    cached_result = cache_manager.load(cache_key)

    if cached_result:
        logging.info(f"Loaded weighted model {model_name} on {sample_name} from cache")
        weighted_params = [(models_with_params[i][0], w, array(p))
                           for i, (w, p) in enumerate(cached_result["params"])]
        return weighted_params, cached_result["formula"]

    logging.info(f"Creating weighted ensemble model for {sample_name}")

    initial_weights = array([weight for _, weight, _ in models_with_params])
    initial_weights = initial_weights / np_sum(initial_weights)

    def loss_func(weights):

        normalized_weights = weights / np_sum(weights)

        weighted_params_eval = [(model_func, normalized_weights[i], params)
                                for i, (model_func, _, params) in enumerate(models_with_params)]

        return calculate_loss(sample_data, lambda X, *_: weighted_ensemble_model(X, weighted_params_eval), array([1.0]))

    bounds = [(0.001, None) for _ in range(len(initial_weights))]

    result = minimize(loss_func, initial_weights, method="L-BFGS-B", bounds=bounds)
    optimal_weights = result.x / np_sum(result.x)

    weighted_params = [(model_func, optimal_weights[i], params)
                       for i, (model_func, _, params) in enumerate(models_with_params)]

    formula = "Weighted Model: "
    for i, (model_func, weight, params) in enumerate(weighted_params):
        if i > 0:
            formula += " + "
        param_formula = format_parameter_equation(model_func.__name__, params)
        formula += f"{weight:.4f} * ({param_formula})"

    plot_cache_key = f"plot_weighted_{model_name}_{sample_name}"
    if not cache_manager.load(plot_cache_key):
        logging.info(f"Creating and caching plots for weighted {model_name} on {sample_name}")
        plot_data = extract_plot_data(sample_data, lambda X, *_: weighted_ensemble_model(X, weighted_params),
                                      array([1.0]))
        create_plots(plot_data,
                     model_name,
                     array([1.0]),
                     sample_id=sample_name)
        cache_manager.save(plot_cache_key, plot_data)
    else:
        logging.info(f"Plots for weighted {model_name} on {sample_name} already cached")

    result_data = {
        "params": [(w, p.tolist()) for (_, w, p) in weighted_params],
        "formula": formula,
    }
    cache_manager.save(cache_key, result_data)

    return weighted_params, formula


def analyze_and_compare_models():
    """Main analysis function that orchestrates the entire process."""
    logging.info("Starting LP operations complexity analysis")
    main_cache_key = "main_results"
    cached_main_results = cache_manager.load(main_cache_key)

    if cached_main_results:
        logging.info("Loaded main results from cache")
        return cached_main_results

    m_range = list(range(200, 2001, 50))
    n_range = list(range(200, 2001, 50))
    num_runs = 5

    logging.info("Generating sample W1")
    sample_W1 = generate_sample("W1", m_range, n_range, num_runs, factor=1.0)

    logging.info("Generating sample W2")
    sample_W2 = generate_sample("W2", m_range, n_range, num_runs, factor=2.0)

    models = [
        ("Borgwardt", borgwardt_model, [1.0]),
        ("Smoothed Analysis", smoothed_analysis_model, [1.0, 1.0]),
        ("Adler-Megiddo", adler_megiddo_model, [1.0]),
        ("Refined Borgwardt", refined_borgwardt_model, [1.0]),
        ("General", general_model, [1.0, 1.0, 1.0]),
        ("General Log", general_model_log, [1.0, 1.0, 1.0, 1.0]),
        ("General Mixed", general_model_mixed, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ]

    results = dict()

    logging.info("Fitting models on sample W1")
    best_model_name_W1 = None
    best_model_score_W1 = float("inf")

    for model_name, model_func, initial_params in models:
        params_W1, formula_W1 = fit_model(model_func, initial_params, sample_W1, model_name, "W1")
        score_W1 = evaluate_model(model_func, params_W1, sample_W1, model_name, "W1")

        logging.info(f"Model {model_name} on W1: {formula_W1}, Score: {score_W1:.2f}")

        if score_W1 < best_model_score_W1:
            best_model_score_W1 = score_W1
            best_model_name_W1 = model_name

        params_W2, formula_W2 = fit_model(model_func, params_W1, sample_W2, model_name,
                                          "W2")
        score_W2 = evaluate_model(model_func, params_W2, sample_W2, model_name, "W2")

        logging.info(f"Model {model_name} on W2: {formula_W2}, Score: {score_W2:.2f}")

        avg_params = average_parameters(params_W1, params_W2)
        avg_formula = format_parameter_equation(model_func.__name__, avg_params)
        avg_score_W1 = evaluate_model(model_func, avg_params, sample_W1, model_name, "W1_avg")
        avg_score_W2 = evaluate_model(model_func, avg_params, sample_W2, model_name, "W2_avg")

        logging.info(f"Averaged {model_name}: {avg_formula}")
        logging.info(f"Averaged {model_name} Score on W1: {avg_score_W1:.2f}")
        logging.info(f"Averaged {model_name} Score on W2: {avg_score_W2:.2f}")

        results[model_name] = {
            "params_W1": params_W1.tolist(),
            "formula_W1": formula_W1,
            "score_W1": score_W1,
            "params_W2": params_W2.tolist(),
            "formula_W2": formula_W2,
            "score_W2": score_W2,
            "avg_params": avg_params.tolist(),
            "avg_formula": avg_formula,
            "avg_score_W1": avg_score_W1,
            "avg_score_W2": avg_score_W2,
        }

    logging.info(f"Best single model on W1: {best_model_name_W1}")

    logging.info("Generating samples L1 and L2 for weighted model analysis")
    sample_L1 = generate_sample("L1", m_range, n_range, num_runs, factor=1.5)
    sample_L2 = generate_sample("L2", m_range, n_range, num_runs, factor=2.5)

    models_with_params = list()
    for model_name, model_func, _ in models:
        avg_params = array(results[model_name]["avg_params"])
        initial_weight = 1.0 / max(1e-6, results[model_name]["avg_score_W1"])
        models_with_params.append((model_func, initial_weight, avg_params))

    weighted_params_L1, weighted_formula_L1 = create_weighted_model(
        models_with_params, sample_L1, "Weighted Ensemble", "L1")

    weighted_score_L1 = evaluate_model(lambda X, *_: weighted_ensemble_model(X, weighted_params_L1),
                                       array([1.0]), sample_L1, "Weighted Ensemble", "L1")

    logging.info(f"Weighted model on L1: {weighted_formula_L1}")
    logging.info(f"Weighted model score on L1: {weighted_score_L1:.2f}")

    logging.info("Testing weighted model on L2")

    models_with_params_L2 = list()
    for (model_func, weight, params), (_, _, _) in zip(weighted_params_L1,
                                                       models_with_params):
        models_with_params_L2.append((model_func, weight, params))

    weighted_params_L2, weighted_formula_L2 = create_weighted_model(
        models_with_params_L2, sample_L2, "Weighted Ensemble", "L2")

    weighted_score_L2 = evaluate_model(lambda X, *_: weighted_ensemble_model(X, weighted_params_L2),
                                       array([1.0]), sample_L2, "Weighted Ensemble", "L2")

    logging.info(f"Weighted model on L2: {weighted_formula_L2}")
    logging.info(f"Weighted model score on L2: {weighted_score_L2:.2f}")

    avg_weighted_params = list()
    for i in range(len(weighted_params_L1)):
        model_L1, weight_L1, params_L1 = weighted_params_L1[i]
        model_L2, weight_L2, params_L2 = weighted_params_L2[i]
        avg_weight = (weight_L1 + weight_L2) / 2
        avg_weighted_params.append((model_L1, avg_weight, params_L1))

    avg_weighted_formula = "Averaged Weighted Model: "
    for i, (model_func, weight, params) in enumerate(avg_weighted_params):
        if i > 0:
            avg_weighted_formula += " + "
        param_formula = format_parameter_equation(model_func.__name__, params)
        avg_weighted_formula += f"{weight:.4f} * ({param_formula})"

    avg_weighted_score_L1 = evaluate_model(lambda X, *_: weighted_ensemble_model(X, avg_weighted_params),
                                           array([1.0]), sample_L1, "Weighted Ensemble", "L1_avg")
    avg_weighted_score_L2 = evaluate_model(lambda X, *_: weighted_ensemble_model(X, avg_weighted_params),
                                           array([1.0]), sample_L2, "Weighted Ensemble", "L2_avg")

    logging.info(f"Averaged weighted model: {avg_weighted_formula}")
    logging.info(f"Averaged weighted model score on L1: {avg_weighted_score_L1:.2f}")
    logging.info(f"Averaged weighted model score on L2: {avg_weighted_score_L2:.2f}")

    results["Weighted Ensemble"] = {
        "params_L1": [(w, p.tolist()) for (_, w, p) in weighted_params_L1],
        "formula_L1": weighted_formula_L1,
        "score_L1": weighted_score_L1,
        "params_L2": [(w, p.tolist()) for (_, w, p) in weighted_params_L2],
        "formula_L2": weighted_formula_L2,
        "score_L2": weighted_score_L2,
        "avg_params": [(w, p.tolist()) for (_, w, p) in avg_weighted_params],
        "avg_formula": avg_weighted_formula,
        "avg_score_L1": avg_weighted_score_L1,
        "avg_score_L2": avg_weighted_score_L2
    }

    logging.info("=" * 80)
    logging.info("SUMMARY OF RESULTS")
    logging.info("=" * 80)

    logging.info("\nBest Parameters of Each Model:")
    logging.info("-" * 80)
    for model_name in results:
        if model_name != "Weighted Ensemble":
            logging.info(f"\n{model_name}:")
            logging.info(
                f"  By sample W1: {results[model_name]["formula_W1"]} (Score: {results[model_name]["score_W1"]:.2f})")
            logging.info(
                f"  By sample W2: {results[model_name]["formula_W2"]} (Score: {results[model_name]["score_W2"]:.2f})")
            logging.info(f"  Averaged: {results[model_name]["avg_formula"]}")
            logging.info(f"    Score on W1: {results[model_name]["avg_score_W1"]:.2f}")
            logging.info(f"    Score on W2: {results[model_name]["avg_score_W2"]:.2f}")

    logging.info("\nBest Single Model:")
    logging.info("-" * 80)

    best_W1 = min([(model_name, results[model_name]["score_W1"])
                   for model_name in results if model_name != "Weighted Ensemble"],
                  key=lambda x: x[1])

    best_W2 = min([(model_name, results[model_name]["score_W2"])
                   for model_name in results if model_name != "Weighted Ensemble"],
                  key=lambda x: x[1])

    best_avg = min([(model_name, (results[model_name]["avg_score_W1"] + results[model_name]["avg_score_W2"]) / 2)
                    for model_name in results if model_name != "Weighted Ensemble"],
                   key=lambda x: x[1])

    logging.info(f"  Best on W1: {best_W1[0]} (Score: {best_W1[1]:.2f})")
    logging.info(f"    Formula: {results[best_W1[0]]["formula_W1"]}")
    logging.info(f"\n  Best on W2: {best_W2[0]} (Score: {best_W2[1]:.2f})")
    logging.info(f"    Formula: {results[best_W2[0]]["formula_W2"]}")
    logging.info(f"\n  Best on average: {best_avg[0]} (Avg Score: {best_avg[1]:.2f})")
    logging.info(f"    Formula: {results[best_avg[0]]["avg_formula"]}")

    logging.info("\nWeighted Ensemble Model:")
    logging.info("-" * 80)
    logging.info(f"  By sample L1: \n    {results["Weighted Ensemble"]["formula_L1"]}")
    logging.info(f"    Score on L1: {results["Weighted Ensemble"]["score_L1"]:.2f}")
    logging.info(f"\n  By sample L2: \n    {results["Weighted Ensemble"]["formula_L2"]}")
    logging.info(f"    Score on L2: {results["Weighted Ensemble"]["score_L2"]:.2f}")
    logging.info(f"\n  Averaged: \n    {results["Weighted Ensemble"]["avg_formula"]}")
    logging.info(f"    Score on L1: {results["Weighted Ensemble"]["avg_score_L1"]:.2f}")
    logging.info(f"    Score on L2: {results["Weighted Ensemble"]["avg_score_L2"]:.2f}")

    logging.info("\nOverall best model:")
    logging.info("-" * 80)
    best_overall = min([(model_name, (results[model_name]["avg_score_W1"] + results[model_name]["avg_score_W2"]) / 2)
                        for model_name in results if model_name != "Weighted Ensemble"],
                       key=lambda x: x[1])
    best_overall = min(best_overall, ("Weighted Ensemble", (
            results["Weighted Ensemble"]["avg_score_L1"] + results["Weighted Ensemble"]["avg_score_L2"]) / 2),
                       key=lambda x: x[1])

    logging.info(f"  Best overall: {best_overall[0]} (Avg Score: {best_overall[1]:.2f})")

    cache_manager.save(main_cache_key, results)
    return results


if __name__ == "__main__":
    start_time = time()
    logging.info(f"Starting analysis at {ctime(start_time)}!")
    analysis_results = analyze_and_compare_models()
    end_time = time()
    logging.info(f"Analysis completed at {ctime(end_time)}!")
    logging.info(f"Analysis completed in {end_time - start_time:.2f} seconds")
