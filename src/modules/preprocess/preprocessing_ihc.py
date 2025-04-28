import pandas as pd
import pickle
import numpy as np
import random
import os
from transformers import AutoTokenizer
np.random.seed(0)
random.seed(0)

class preprocessor:
    def __init__(self, 
                 data_home='dataset/ihc_pure/',
                 tokenizer_type='bert-base-uncased',
                 augmentation='imp',
                 output_dir='dataset'):
        self.data_home = data_home
        self.tokenizer_type = tokenizer_type
        self.augmentation = augmentation
        self.output_dir = output_dir
        self.class2int = {'not_hate': 0, 'implicit_hate': 1}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        os.makedirs(output_dir, exist_ok=True)
    def _process_split(self, datatype):
        """Process a single data split (train/valid/test)"""
        datafile = os.path.join(self.data_home, f"{datatype}.tsv")
        data = pd.read_csv(datafile, sep='\t')
        labels = data["class"].map(self.class2int).tolist()
        posts = data["post"].tolist()
        if datatype == "train" and self.augmentation in ['imp', 'aug']:
            augmented_posts = []
            for i, class_name in enumerate(data["class"]):
                if self.augmentation == 'imp':
                    if class_name == 'implicit_hate':
                        aug_text = data["implied_statement"][i]
                    else:  # not_hate
                        aug_text = data["aug_sent1_of_post"][i]
                else:  # aug
                    aug_text = data["aug_sent1_of_post"][i]
                if pd.isna(aug_text):
                    aug_text = posts[i]  # Fallback to original
                augmented_posts.append(aug_text)
            tokenized_original = self.tokenizer.batch_encode_plus(posts).input_ids
            tokenized_augmented = self.tokenizer.batch_encode_plus(augmented_posts).input_ids
            return {
                "tokenized_post": [[o, a] for o, a in zip(tokenized_original, tokenized_augmented)],
                "label": [[l, l] for l in labels],
                "post": [[p, a] for p, a in zip(posts, augmented_posts)]
            }
        else:
            tokenized_posts = self.tokenizer.batch_encode_plus(posts).input_ids
            return {
                "tokenized_post": tokenized_posts,
                "label": labels,
                "post": posts
            }
    def process(self):
        data_dict = {}
        for datatype in ["train", "valid", "test"]:
            print(f"Processing {datatype} data...")
            processed_data = self._process_split(datatype)
            data_dict[datatype] = pd.DataFrame.from_dict(processed_data)
        output_filename = f"ihc_{self.augmentation}_preprocessed_{self.tokenizer_type.split('-')[0]}.pkl"
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"Processing complete. Data saved to {output_path}")