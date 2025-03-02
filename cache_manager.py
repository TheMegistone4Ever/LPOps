import json
import logging
import os
from typing import Any, Optional


def create_key(m: int, n: int, model_type: str) -> str:
    """Create a unique key for identifying computations."""

    return f"{m}_{n}_{model_type}"


class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "computation_cache.json")
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def save(self, key: str, data: Any) -> None:
        """Save data to file."""

        try:
            # Load existing data
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
            else:
                cache = {}

            # Update with new data
            cache[key] = data

            # Save back to file
            with open(self.cache_file, "w") as f:
                json.dump(cache, f)
            logging.info(f"Saved to cache: {key}")
        except Exception as e:
            logging.error(f"Cache save failed: {e}")

    def load(self, key: str) -> Optional[Any]:
        """Load data from a file."""

        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)
                    if key in cache:
                        logging.info(f"Cache hit: {key}")
                        return cache[key]
            return None
        except Exception as e:
            logging.error(f"Cache load failed: {e}")
            return None
