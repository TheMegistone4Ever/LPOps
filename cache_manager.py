import logging
from os import path, makedirs, replace, remove, listdir
from pickle import load, PickleError, dump
from tempfile import NamedTemporaryFile
from typing import Any, Optional


def create_key(m: int, n: int, model_type: str) -> str:
    """Create a unique key for identifying computations."""

    return f"{m}_{n}_{model_type}"


class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
        self.key_map_file = path.join(cache_dir, "key_map.bin")
        self.key_map = self._load_key_map()

    def _ensure_cache_dir(self):
        """Ensure cache directory exists."""

        makedirs(self.cache_dir, exist_ok=True)

    def _load_key_map(self):
        """Load the key set file or create if it doesn't exist."""

        if path.exists(self.key_map_file):
            try:
                with open(self.key_map_file, "rb") as f:
                    return load(f)
            except (PickleError, MemoryError) as e:
                logging.warning(f"Key map file corrupted, resetting: {e}")
        return set()

    def _save_key_map(self):
        """Save the key set file."""

        try:
            with NamedTemporaryFile(mode="wb", dir=self.cache_dir, delete=False) as temp_file:
                dump(self.key_map, temp_file)
                temp_file_name = temp_file.name
            replace(temp_file_name, self.key_map_file)
        except Exception as e:
            logging.exception(f"Failed to save key map: {e}")
            if "temp_file_name" in locals() and path.exists(temp_file_name):
                try:
                    remove(temp_file_name)
                except OSError:
                    pass

    def _reset_key_map(self):
        """Reset the key set file."""

        self.key_map = set()
        self._save_key_map()

    def save(self, key: str, data: Any) -> None:
        """Save data to a separate file using """

        try:
            # Use the key directly for file name instead of hashing
            file_path = path.join(self.cache_dir, f"{key}.pkl")

            # Add key to the set
            self.key_map.add(key)
            self._save_key_map()

            # Serialize using pickle
            with NamedTemporaryFile(mode="wb", dir=self.cache_dir, delete=False) as temp_file:
                dump(data, temp_file)
                temp_file_name = temp_file.name

            replace(temp_file_name, file_path)
            logging.info(f"Saved to cache: {key}")

        except Exception as e:
            logging.exception(f"Cache save failed: {e}")
            if "temp_file_name" in locals() and path.exists(temp_file_name):
                try:
                    remove(temp_file_name)
                except OSError as remove_error:
                    logging.error(f"Failed to remove temporary file: {remove_error}")

    def load(self, key: str) -> Optional[Any]:
        """Load data from a file using """

        try:
            # Check if key exists in set
            if key not in self.key_map:
                return None

            file_path = path.join(self.cache_dir, f"{key}.pkl")
            if not path.exists(file_path):
                return None

            with open(file_path, "rb") as f:
                return load(f)

        except (PickleError, MemoryError) as e:
            logging.warning(f"Failed to load cache entry {key}: {e}")
            return None
        except Exception as e:
            logging.exception(f"Cache load failed: {e}")
            return None

    def clear_cache(self):
        """Clear all cache files."""

        for filename in listdir(self.cache_dir):
            if filename.endswith(".pkl"):
                try:
                    remove(path.join(self.cache_dir, filename))
                except OSError as e:
                    logging.error(f"Failed to remove cache file {filename}: {e}")

        self._reset_key_map()
