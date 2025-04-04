import os
import pandas as pd
import pickle
from transformers import AutoTokenizer

class preprocessor_sbic:
    def __init__(self, 
                 dataset="sbic", 
                 aug_type="full",  # Set to None if you do not want augmentation.
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
        data = pd.read_csv(datafile, sep=',')
        
        # Ensure that missing posts are filled with empty strings.
        data["post"] = data["post"].fillna("")
        if self.dataset == "sbic_imp":
            data = data.fillna('')
        
        # Convert the posts to strings.
        posts = data["post"].astype(str).tolist()
        labels = [self.class2int[label] for label in data["offensiveLABEL"]]
        
        # If processing the train split and augmentation is desired.
        if datatype == "train" and self.aug_type is not None:
            if self.aug_type == "aug":
                augmented_posts = data["aug_sent1_of_post"].fillna("").astype(str).tolist()
            elif self.aug_type == "imp":
                augmented_posts = []
                for i in range(len(data)):
                    selected = str(data["selectedStereotype"].iloc[i]).strip()
                    if selected:  # Use the selected stereotype if available.
                        augmented_posts.append(selected)
                    else:
                        aug_text = str(data["aug_sent1_of_post"].iloc[i]).strip()
                        augmented_posts.append(aug_text)
            else:
                raise ValueError(f"Unknown augmentation type: {self.aug_type}")
            
            print("Tokenizing data (with augmentation)...")
            tokenized_posts = self.tokenizer.batch_encode_plus(posts).input_ids
            tokenized_augmented = self.tokenizer.batch_encode_plus(augmented_posts).input_ids
            
            # Combine the original and augmented data into pairs.
            tokenized_combined = [list(pair) for pair in zip(tokenized_posts, tokenized_augmented)]
            combined_posts = [list(pair) for pair in zip(posts, augmented_posts)]
            combined_labels = [list(pair) for pair in zip(labels, labels)]
            
            processed_data = {
                "tokenized_post": tokenized_combined,
                "label": combined_labels,
                "post": combined_posts
            }
        else:
            print("Tokenizing data...")
            tokenized_posts = self.tokenizer.batch_encode_plus(posts).input_ids
            processed_data = {
                "tokenized_post": tokenized_posts,
                "label": labels,
                "post": posts
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

