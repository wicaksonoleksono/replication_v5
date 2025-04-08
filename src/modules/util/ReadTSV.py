import pandas as pd
import os


class read_tsv:
    def __init__(self, homepath):
        self.homepath = homepath

    def _read(self, dataset_name):
        filepath = os.path.join(self.homepath, dataset_name)
        _, ext = os.path.splitext(filepath)  # ext will be ".tsv" or ".csv"
        if ext == ".tsv":
            df = pd.read_csv(filepath, delimiter='\t', header=0)
        elif ext == ".csv":
            df = pd.read_csv(filepath, delimiter=',', header=0)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        return df

    def run(self, dataset_name):
        return self._read(dataset_name)
