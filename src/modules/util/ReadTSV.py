import pandas as pd
import os

class read_tsv:
    def __init__(self, homepath):
        self.homepath = homepath  # Correct assignment

    def _read(self, dataset_name):
        file_load_dir = os.path.join(self.homepath, dataset_name)  # Use os.path.join
        return pd.read_csv(file_load_dir, delimiter='\t', header=0)

    def run(self, dataset_name):  # Make sure parameter name matches usage
        return self._read(dataset_name)
