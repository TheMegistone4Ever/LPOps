from multiprocessing import Pool

from numpy.random import seed as np_seed

from gmdh import load_data, run_gmdh

TRUE_FEATURES = {"x1", "x2", "x3", "x4", "x5", }


def run_with_seed(seed: int):
    np_seed(seed)

    best_names, coefs, intercept, sse, m1, m2 = run_gmdh(load_data())
    print(f"SEED {seed} -> best_names: {best_names}, m1: {m1}, m2: {m2}")

    good = all(f in TRUE_FEATURES for f in best_names)

    return seed, best_names, good


def worker(seed: int):
    try:
        seed, features, good = run_with_seed(seed)

        print(f"SEED {seed} -> {features}")
        if good:
            print(f"FOUND GOOD SEED: {seed}")

        return seed, features if good else None

    except Exception as e:
        print(f"SEED {seed} ERROR: {e}")
        return None


def search_seeds(max_seed: int):
    seeds = list(range(max_seed))

    good_results = list()

    with Pool(10) as pool:
        for result in pool.imap_unordered(worker, seeds):
            if result is not None and result[1] is not None:
                good_results.append(result)

    print("\nSeed: name of features found")

    for seed, features in sorted(good_results):
        print(f"{seed}: {features}")


if __name__ == "__main__":
    search_seeds(10000)
