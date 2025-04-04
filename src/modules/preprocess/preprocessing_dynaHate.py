import pandas as pd
import pickle
import numpy as np
import random
import os
from transformers import AutoTokenizer

np.random.seed(0)
random.seed(0)
class preprocessor_dyna:
    def __init__(self, 
                 data_home='dataset/DynaHate',
                 tokenizer_type='bert-base-uncased',
                 output_dir='./dataset'):
        self.data_home = data_home
        self.tokenizer_type = tokenizer_type
        self.output_dir = output_dir
        self.class2int = {'nothate': 0, 'hate': 1}
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)
        os.makedirs(self.output_dir, exist_ok=True)
    def _process_split(self, datatype):
        """Process a single data split (train/dev/test) following the reference logic."""
        datafile = os.path.join(self.data_home, f"{datatype}.csv")
        data = pd.read_csv(datafile, sep=',')
        labels = [self.class2int[label] for label in data["label"]]
        posts = data["text"].tolist()
        print(f"Tokenizing {datatype} data...")
        tokenized_posts = self.tokenizer.batch_encode_plus(posts,truncation=False,padding=False,add_special_tokens=True).input_ids
        processed_data = {
            "tokenized_post": tokenized_posts,
            "label": labels,
            "post": posts
        }
        return pd.DataFrame.from_dict(processed_data)
    def process(self):
        data_dict = {}
        for datatype in ["train", "dev", "test"]:
            data_dict[datatype] = self._process_split(datatype)
        output_path = os.path.join(self.output_dir, "dynahate_preprocessed_bert.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Processing complete. Data saved to {output_path}")
