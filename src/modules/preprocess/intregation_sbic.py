import os
import random
import argparse
import numpy as np
import pandas as pd
import spacy
from tqdm import trange
import nlpaug.augmenter.word as naw

# Set seeds for reproducibility
np.random.seed(0)
random.seed(0)

class integration_sbic:
    def __init__(self, load_dir, output_dir):
        """
        Args:
            load_dir (str): Directory containing the SBIC CSV files.
            output_dir (str): Directory to save the integrated output files.
        """
        self.load_dir = load_dir
        self.output_dir = output_dir
        # Load the spaCy model once to be reused for all transformations
        self.nlp = spacy.load("en_core_web_sm")
        # Initialize the synonym augmenter from nlpaug
        self.aug = naw.SynonymAug(aug_src='wordnet')

    def _aggregate_annotations(self, split):
        """
        Aggregates annotations for a given split.
        
        In the original SBIC CSV, a post may appear multiple times with annotations
        from different workers. For each post, this method aggregates its annotations
        into a single row (for evaluation splits) or creates a single instance per post
        (for training) by randomly selecting one valid annotation.
        
        Args:
            split (str): One of 'trn', 'dev', or 'tst'.
            
        Returns:
            pd.DataFrame: DataFrame with columns 
              ["post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"]
        """
        file_path = os.path.join(self.load_dir, f"SBIC.v2.{split}.csv")
        df = pd.read_csv(file_path)
        columns = ["post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"]
        aggregated_data = []
        visited_posts = []

        for i in trange(len(df), desc=f"Aggregating {split} data"):
            post = df.loc[i, "post"]
            if post in visited_posts:
                continue
            visited_posts.append(post)

            # A post is offensive if at least half of the annotators say it is.
            post_rows = df[df["post"] == post]
            offensiveYN_frac = post_rows["offensiveYN"].sum() / float(len(post_rows["offensiveYN"]))
            offensiveYN_label = 1.0 if offensiveYN_frac >= 0.5 else 0.0

            # A post targets a demographic group if at least half say so.
            whoTarget_frac = post_rows["whoTarget"].sum() / float(len(post_rows["whoTarget"]))
            whoTarget_label = 1.0 if whoTarget_frac >= 0.5 else 0.0

            targetMinority_label = None
            targetStereotype_label = None

            if whoTarget_label == 1.0:
                minorities = post_rows["targetMinority"]
                stereotypes = post_rows["targetStereotype"]

                if split in ['dev', 'tst']:
                    # For evaluation: combine all valid annotations
                    targetMinority_labels = []
                    targetStereotype_labels = []
                    for m, s in zip(minorities, stereotypes):
                        if not pd.isna(s):
                            targetMinority_labels.append(m)
                            targetStereotype_labels.append(s)
                    targetMinority_label = ' [SEP] '.join(targetMinority_labels)
                    targetStereotype_label = ' [SEP] '.join(targetStereotype_labels)
                    aggregated_data.append([post, offensiveYN_label, whoTarget_label,
                                            targetMinority_label, targetStereotype_label])
                else:
                    # For training: choose one valid annotation instance per post.
                    temp_aggregated_data = []
                    for m, s in zip(minorities, stereotypes):
                        if not pd.isna(s):
                            temp_aggregated_data.append([post, offensiveYN_label, whoTarget_label, m, s])
                    if len(temp_aggregated_data) > 0:
                        one_data_for_one_post = random.choice(temp_aggregated_data)
                    else:
                        # In the unlikely case that none of the annotations are valid, leave as None.
                        one_data_for_one_post = [post, offensiveYN_label, whoTarget_label, None, None]
                    aggregated_data.append(one_data_for_one_post)
            else:
                aggregated_data.append([post, offensiveYN_label, whoTarget_label,
                                        targetMinority_label, targetStereotype_label])
        df_new = pd.DataFrame(aggregated_data, columns=columns)
        return df_new

    def _turn_implied_statements_to_explanations(self, split, df):
        """
        Transforms annotations about the targeted identity group and associated stereotypes
        into a coherent explanation sentence.
        
        For example, if a post implies that "women can't drive", it may transform the annotation 
        into: "this post implies that women can't drive."
        
        Args:
            split (str): The dataset split (e.g., 'trn').
            df (pd.DataFrame): DataFrame that includes the columns:
              - offensiveLABEL
              - whoTarget
              - targetMinority (may be multiple values separated by [SEP])
              - targetStereotype (may be multiple values separated by [SEP])
              
        Returns:
            pd.DataFrame: The DataFrame with an added column "selectedStereotype" that contains
                          the transformed explanation.
        """
        if df is None:
            raise NotImplementedError("The input dataframe is None.")

        df['selectedStereotype'] = pd.Series(dtype="object")
        group_attack_no_implied_statement = 0
        personal_attack = 0
        not_offensive = 0
        group_offensive = 0
        offensive_na_whotarget = 0

        for i in trange(len(df), desc=f"Transforming implied statements for {split} data"):
            offensive_label = df.loc[i, "offensiveLABEL"]

            if offensive_label == 'offensive' and (pd.isna(df.loc[i, "whoTarget"]) or df.loc[i, "whoTarget"] == ''):
                offensive_na_whotarget += 1
                continue

            if offensive_label == 'offensive' and df.loc[i, "whoTarget"] == 1.0:
                if pd.isna(df.loc[i, "targetStereotype"]) or df.loc[i, "targetStereotype"] == '':
                    group_attack_no_implied_statement += 1
                    continue

                inferences = df.loc[i, "targetStereotype"].split(' [SEP] ')
                target_minorities = df.loc[i, "targetMinority"].split(' [SEP] ')
                new_inferences = []
                for target_minority, inference in zip(target_minorities, inferences):
                    # Use spaCy to get token annotations for the inference
                    doc = self.nlp(inference)
                    inference_annotations = [{'token': token.text, 'pos': token.pos_, 'tag': token.tag_} 
                                             for token in doc]
                    if inference_annotations:
                        first_word = inference_annotations[0]
                    else:
                        first_word = {'token': '', 'pos': '', 'tag': ''}
                    changed = False

                    # Apply a series of rules to transform the statement.
                    if first_word['token'] == 'claims' and first_word['tag'] not in ['VBD', 'VBZ']:
                        new_inferences.append(f'this post {inference}')
                        changed = True

                    if first_word['tag'] in ['VBD', 'VBZ']:
                        new_inferences.append(f'this post {inference}')
                        changed = True

                    if (first_word['tag'] == 'VBG' and first_word['token'] not in ['implying', 'calling']) or first_word['token'] == 'being':
                        new_inferences.append(f'this post implies that {inference}')
                        changed = True

                    if first_word['token'] in ['implying', 'calling']:
                        new_inferences.append(f'this post is {inference}')
                        changed = True

                    if first_word['token'] == 'rape':
                        new_inferences.append(f'this post implies that {target_minority} {inference}')
                        changed = True

                    if first_word['tag'] in ['MD', 'VB'] or (first_word['pos'] == 'AUX' and first_word['token'] != 'being'):
                        if (target_minority in ["women", "men"] or "folks" in target_minority or len(target_minority.split(',')) > 1) and first_word['token'] == 'is':
                            split_inference = ' '.join(inference.split(' ')[1:])
                            new_inferences.append(f'this post implies that {target_minority} are {split_inference}')
                            changed = True
                        else:
                            new_inferences.append(f'this post implies that {target_minority} {inference}')
                            changed = True

                    if first_word['token'] == "all":
                        new_inferences.append(f'this post implies that {target_minority} are {inference}')
                        changed = True

                    if not changed:
                        new_inferences.append(f'this post implies that {inference}')
                    group_offensive += 1

                # If multiple explanations exist, select one at random.
                if len(new_inferences) > 1:
                    df.loc[i, "selectedStereotype"] = random.choice(new_inferences)
                else:
                    df.loc[i, "selectedStereotype"] = new_inferences[0]

            if offensive_label == 'offensive' and df.loc[i, "whoTarget"] == 0.0:
                personal_attack += 1

            if offensive_label == 'not_offensive':
                not_offensive += 1

        print("---------------------------------------------------")
        print(f"Split: {split}")
        print(f"offensive_na_whotarget: {offensive_na_whotarget}")
        print(f"Group attack but no implied statement: {group_attack_no_implied_statement}")
        print(f"Personal attacks: {personal_attack}")
        print(f"Group offensive: {group_offensive}")
        print(f"Not offensive: {not_offensive}")
        print("---------------------------------------------------")
        return df

    def run(self):
        """
        Executes the full integration process:
         1. Aggregate annotations for train, dev, and test splits.
         2. Create a binary 'offensiveLABEL' column.
         3. Save dev and test splits.
         4. For training data, add augmented sentences, transform implied statements, 
            and then save the training set.
        """
        # Aggregate annotations for each split.
        print("Aggregating training data...")
        sbic_train = self._aggregate_annotations('trn')
        print("Aggregating dev data...")
        sbic_dev = self._aggregate_annotations('dev')
        print("Aggregating test data...")
        sbic_test = self._aggregate_annotations('tst')
        sbic_train['offensiveLABEL'] = np.where(sbic_train['offensiveYN'] >= 0.5, 'offensive', 'not_offensive')
        sbic_dev['offensiveLABEL'] = np.where(sbic_dev['offensiveYN'] >= 0.5, 'offensive', 'not_offensive')
        sbic_test['offensiveLABEL'] = np.where(sbic_test['offensiveYN'] >= 0.5, 'offensive', 'not_offensive')
        os.makedirs(self.output_dir, exist_ok=True)
        sbic_dev.to_csv(os.path.join(self.output_dir, "dev.csv"), index=False)
        sbic_test.to_csv(os.path.join(self.output_dir, "test.csv"), index=False)

        sbic_train['aug_sent1_of_post'] = pd.Series(dtype="object")
        sbic_train['aug_sent2_of_post'] = pd.Series(dtype="object")
        print("Augmenting training data...")
        for i, one_post in enumerate(sbic_train["post"]):
            sbic_train.loc[i, 'aug_sent1_of_post'] = self.aug.augment(one_post)
            sbic_train.loc[i, 'aug_sent2_of_post'] = self.aug.augment(one_post)

        # Transform implied statements to coherent explanations.
        print("Transforming implied statements for training data...")
        sbic_train = self._turn_implied_statements_to_explanations('trn', sbic_train)

        # Save the processed training set.
        sbic_train.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
        print("Integration complete. Files saved to:", self.output_dir)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Integrate SBIC annotations and create augmented files.")
#     parser.add_argument('--load_dir', type=str, default="dataset/SBIC.v2", help="Directory containing SBIC CSV files.")
#     parser.add_argument('--output_dir', type=str, default="dataset/SBIC.v2", help="Directory to save output files.")
#     args = parser.parse_args()

    # integrator = integration_sbic(load_dir=args.load_dir, output_dir=args.output_dir)
    # integrator.run()
