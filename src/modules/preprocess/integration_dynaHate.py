import os
import pandas as pd

class integration_dyna:
    def __init__(self, load_dir="dataset/DynaHate", dataset_filename="DynaHate_v0.2.2.csv"):

        self.load_dir = load_dir
        self.dataset_filename = dataset_filename
        self.dataset_path = os.path.join(self.load_dir, self.dataset_filename)
        self.dataset = None
        self.train = None
        self.dev = None
        self.test = None
    def load_dataset(self):
        self.dataset = pd.read_csv(self.dataset_path, delimiter=',', header=0)
        # Drop the first column (assuming it's an index column)
        self.dataset = self.dataset.drop(self.dataset.columns[0], axis=1)
        return self.dataset

    def split_dataset(self):

        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please call load_dataset() before splitting.")

        self.train = self.dataset[self.dataset['split'] == 'train']
        self.dev = self.dataset[self.dataset['split'] == 'dev']
        self.test = self.dataset[self.dataset['split'] == 'test']
        return self.train, self.dev, self.test

    def save_splits(self, output_dir=None):
 
        if output_dir is None:
            output_dir = self.load_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.train is None or self.dev is None or self.test is None:
            raise ValueError("Dataset splits not available. Please call split_dataset() before saving.")
        self.train.to_csv(os.path.join(output_dir, "train.csv"), sep=",", index=False)
        self.dev.to_csv(os.path.join(output_dir, "dev.csv"), sep=",", index=False)
        self.test.to_csv(os.path.join(output_dir, "test.csv"), sep=",", index=False)
    def process(self):
        self.load_dataset()
        self.split_dataset()
        self.save_splits()


