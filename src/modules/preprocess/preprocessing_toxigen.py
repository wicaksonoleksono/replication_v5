import pandas as pd
import pickle
import os
from transformers import AutoTokenizer
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(0)
random.seed(0)

class preprocess_toxigen:
    def __init__(self, 
                 datafile="dataset/ToxiGen/toxigen_all.csv",
                 tokenizer_type='bert-base-uncased',
                 output_dir='./preprocessed_data'):
        """
        Preprocessor for the ToxiGen dataset.

        Parameters:
            datafile (str): Path to the ToxiGen CSV file.
            tokenizer_type (str): Pretrained tokenizer type from HuggingFace 
                                  (default: 'bert-base-uncased').
            output_dir (str): Directory where the preprocessed data will be saved.
        """
        self.datafile = datafile
        self.tokenizer_type = tokenizer_type
        self.output_dir = output_dir
        
        # Define label mapping
        self.class2int = {'not_hate': 0, 'implicit_hate': 1}
        
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process(self):
        """
        Process the ToxiGen dataset by:
          - Loading the CSV file.
          - Mapping the labels using self.class2int.
          - Tokenizing the 'generation' column.
          - Saving the processed data as a pickle file.
        """
        print(f"Loading ToxiGen dataset from {self.datafile}")
        data = pd.read_csv(self.datafile)
        
        labels = []
        posts = []
        
        # Iterate over the rows to extract labels and posts
        for i, one_class in enumerate(data["prompt_label"]):
            # Map the label; if it doesn't exist in the mapping, use the original value.
            labels.append(self.class2int.get(one_class, one_class))
            posts.append(data["generation"].iloc[i])
        
        print("Tokenizing data...")
        tokenized_post = self.tokenizer.batch_encode_plus(
            posts,
            truncation=True,
            padding=True
        )['input_ids']
        
        # Create a dictionary with the processed data
        processed_data = {
            "tokenized_post": tokenized_post,
            "label": labels,
            "post": posts
        }
        
        # Convert the dictionary to a DataFrame for consistency
        processed_data_df = pd.DataFrame.from_dict(processed_data)
        data_dict = {"test": processed_data_df}
        
        # Create the output filename based on the tokenizer type
        output_filename = f"toxigen_all_preprocessed_{self.tokenizer_type.split('-')[0]}.pkl"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save the processed data to a pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f)
            
        print(f"Processing complete. Data saved to {output_path}")

# if __name__ == '__main__':
#     preprocessor = toxigen_preprocessor(
#         datafile="dataset/ToxiGen/toxigen_all.csv",
#         tokenizer_type='bert-base-uncased',
#         output_dir='./preprocessed_data'
#     )
#     preprocessor.process()
