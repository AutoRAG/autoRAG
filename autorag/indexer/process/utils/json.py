import os
import json


class JsonFileLoader:
    """
    Load all JSON files in the specified directory.

    Iterate through each file in the directory. Check if it's a JSON file
    by its extension. Load it using the _load_a_file method, and add to
    the all_files dict.

    :param data_dir (str): Path to the data directory.
    """

    def __init__(self, data_dir: str = None) -> None:
        # Initialize the loader with a directory path. Check if the path is valid.
        if not data_dir:
            raise ValueError("Must provide `data_dir`.")
        # If the path is empty or not a directory, raise an error.
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory {data_dir} does not exist.")

        self.data_dir = data_dir

    def load(self) -> dict:
        all_files = {}
        for filename in os.listdir(self.data_dir):
            data = self._load_a_file(filename)
            # Store data with filename as key
            if data:
                filename = filename.split(".")[0]
                all_files[filename] = data
        return all_files

    def _load_a_file(self, filename: str):
        # Internal method to load a single JSON file.
        # Open the file, load the JSON content, and return the data.
        if filename.endswith(".json"):
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, "r") as file:
                return json.load(file)
