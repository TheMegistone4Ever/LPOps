import json
import logging
import os
import tempfile
from typing import Any, Optional

from numpy import ndarray, array, number


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

        os.makedirs(self.cache_dir, exist_ok=True)

    def save(self, key: str, data: Any) -> None:
        """Save data to file, converting ndarrays to lists."""

        try:
            # Load existing data (handling potential corruption)
            cache = {}
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r") as f:
                        cache = json.load(f)
                except json.JSONDecodeError as e:
                    logging.warning(f"Cache file corrupted, resetting: {e}")
                    cache = {}

            # Convert ndarrays to lists *before* saving
            serialized_data = self._serialize_data(data)
            cache[key] = serialized_data

            # Atomic write using tempfile
            with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.cache_dir, delete=False
            ) as temp_file:
                json.dump(cache, temp_file, indent=4)
                temp_file_name = temp_file.name

            os.replace(temp_file_name, self.cache_file)
            logging.info(f"Saved to cache: {key}")

        except Exception as e:
            logging.exception(f"Cache save failed: {e}")
            if 'temp_file_name' in locals() and os.path.exists(temp_file_name):
                try:
                    os.remove(temp_file_name)  # Cleanup temp file on failure
                except OSError as remove_error:
                    logging.error(f"Failed to remove temporary file: {remove_error}")

    def load(self, key: str) -> Optional[Any]:
        """Load data from file, converting lists back to ndarrays."""

        try:
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
                serialized_data = cache.get(key)
                if serialized_data:
                    # Convert lists back to ndarrays *after* loading
                    return self._deserialize_data(serialized_data)
                return None
        except FileNotFoundError:
            logging.info(f"Cache file not found: {self.cache_file}")
            return None
        except json.JSONDecodeError as e:
            logging.warning(f"Cache file corrupted, ignoring: {e}")
            return None
        except Exception as e:
            logging.exception(f"Cache load failed: {e}")
            return None

    def _serialize_data(self, data: Any) -> Any:
        """Recursively convert ndarrays to lists."""

        if isinstance(data, ndarray):
            return data.tolist()  # Convert ndarray to list
        elif isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._serialize_data(item) for item in data)
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        else:
            return data

    def _deserialize_data(self, data: Any) -> Any:
        """Recursively convert lists back to ndarrays."""

        if isinstance(data, list):
            # Helper function to recursively check if all elements are numeric
            def all_numeric(items):
                if isinstance(items, list):
                    return all(all_numeric(item) for item in items)
                return isinstance(items, (int, float, number))

            if all_numeric(data):
                try:
                    return array(data)
                except (ValueError, TypeError):
                    # If conversion fails, it was probably just a list
                    return [self._deserialize_data(item) for item in data]
            else:
                return [self._deserialize_data(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._deserialize_data(item) for item in data)
        elif isinstance(data, dict):
            return {key: self._deserialize_data(value) for key, value in data.items()}
        else:
            return data
