IS_BENCHMARKING = True
BENCHMARK_NOISE_PERCENT = .01

IS_VARIANCE_FIX = False
CHECK_VAR = False
CACHE_DIR = "cache"

# Synthetic data generation options for CHECK_VAR mode.
# Supported: "normal", "laplace", "uniform", "normal_uniform".
SYNTH_X_DISTRIBUTION = "uniform"
SYNTH_ERROR_DISTRIBUTION = "laplace"


def get_gmdh_paths(is_benchmarking: bool, is_variance_fix: bool):
    gmdh_cache_dir, plots_dir, log_file = "gmdh_cache", "plots_combi", "gmdh_execution"
    for gmdh_path in gmdh_cache_dir, plots_dir, log_file:
        if is_benchmarking:
            gmdh_path += "_benchmark"
        if is_variance_fix:
            gmdh_path += "_var_fixed"
    return gmdh_cache_dir, plots_dir, f"{log_file}.log"
