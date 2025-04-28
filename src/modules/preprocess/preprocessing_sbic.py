import os
import pandas as pd
import pickle
from transformers import AutoTokenizer


class preprocessor_sbic:
    def __init__(self,
                 dataset="sbic",
                 aug_type="imp",
                 data_home="../dataset/sbic_pure/",
                 tokenizer_type="bert-base-uncased",
                 output_dir="preprocessed_data"):
        self.dataset = dataset
        self.aug_type = aug_type
        self.data_home = data_home
        self.tokenizer_type = tokenizer_type
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.class2int = {'not_offensive': 0, 'offensive': 1}
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)

    def _process_split(self, datatype):
        datafile = os.path.join(self.data_home, f"{datatype}.csv")
        data = pd.read_csv(datafile, sep=',').fillna('')
        data["post"] = data["post"].fillna("")
        labels = [self.class2int[l] for l in data["offensiveLABEL"]]
        posts = data["post"].astype(str).tolist()
        if datatype == "train" and self.aug_type == "imp":
            augmented_posts = []
            for _, row in data.iterrows():
                sel = row["selectedStereotype"].strip()
                if sel:
                    augmented_posts.append(sel)
                else:
                    augmented_posts.append(row["aug_sent1_of_post"].strip())
            print("Tokenizing data (with augmentation)...")
            tokenized_posts = self.tokenizer(posts,    padding=True, truncation=True).input_ids
            tokenized_augments = self.tokenizer(augmented_posts, padding=True, truncation=True).input_ids
            tokenized_combined = [list(pair) for pair in zip(tokenized_posts, tokenized_augments)]
            combined_posts = [[a, b] for a, b in zip(posts, augmented_posts)]
            combined_labels = [[y, y] for y in labels]
            processed_data = {
                "tokenized_post": tokenized_combined,
                "post":           combined_posts,
                "label":          combined_labels
            }
        else:
            print("Tokenizing data...")
            tokenized_posts = self.tokenizer.batch_encode_plus(posts).input_ids
            processed_data = {
                "tokenized_post": tokenized_posts,
                "post":           posts,
                "label":          labels
            }
        return pd.DataFrame.from_dict(processed_data)

    def process(self):
        """
        Processes all data splits and saves the combined dictionary as a pickle file.
        """
        data_dict = {}
        for split in ["train", "dev", "test"]:
            print(f"Processing {split} data...")
            processed_df = self._process_split(split)
            data_dict[split] = processed_df

        # Build the filename based on the augmentation type.
        if self.aug_type is not None:
            filename = f"{self.dataset}_{self.aug_type}_preprocessed_{self.tokenizer_type.split('-')[0]}.pkl"
        else:
            filename = f"{self.dataset}_preprocessed_{self.tokenizer_type.split('-')[0]}.pkl"
        output_path = os.path.join(self.output_dir, filename)
        # Save the processed data as a pickle file.
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"Processing complete. Data saved to {output_path}")
