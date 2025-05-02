from tqdm import trange
import pandas as pd
import random


def _aggregate_annotations(self, split):
    file_path = os.path.join(self.load_dir, f"SBIC.v2.{split}.csv")
    df = pd.read_csv(file_path)
    columns = ["post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"]
    aggregated_data = []

    for post, group in trange(df.groupby('post'), desc=f"Aggregating {split} data", total=df['post'].nunique()):
        offensiveYN_label = 1.0 if group['offensiveYN'].mean() >= 0.5 else 0.0
        whoTarget_label = 1.0 if group['whoTarget'].mean() >= 0.5 else 0.0
        target_minority, target_stereotype = None, None

        if whoTarget_label == 1.0:
            valid = group.dropna(subset=['targetStereotype'])
            if split in ['dev', 'tst']:
                target_minority = ' [SEP] '.join(valid['targetMinority'].astype(str))
                target_stereotype = ' [SEP] '.join(valid['targetStereotype'].astype(str))
            else:
                if not valid.empty:
                    selected = valid.sample(1).iloc[0]
                    target_minority, target_stereotype = selected[['targetMinority', 'targetStereotype']]
                else:
                    last = group.iloc[-1]
                    target_minority, target_stereotype = last['targetMinority'], last['targetStereotype']

        aggregated_data.append([
            post,
            offensiveYN_label,
            whoTarget_label,
            target_minority if target_minority else None,
            target_stereotype if target_stereotype else None
        ])
    return pd.DataFrame(aggregated_data, columns=columns)


def turn_implied_statements_to_explanations(nlp, df, split):
    if df is None:
        raise ValueError("Input dataframe cannot be None")

    # work on a copy to avoid side-effects
    out = df.copy()
    out['selectedStereotype'] = pd.Series(dtype="object")

    for idx, row in trange(out.iterrows(), total=len(out), desc=f"Processing {split} data"):
        if row['offensiveLABEL'] != 'offensive':
            continue
        if row['whoTarget'] != 1.0 or pd.isna(row['targetStereotype']):
            continue
        minorities = row['targetMinority'].split(' [SEP] ')
        inferences = row['targetStereotype'].split(' [SEP] ')
        explanations = []
        for target_minority, inference in zip(minorities, inferences):
            doc = nlp(inference)
            if len(doc) > 0:
                token = doc[0]
                word, tag, pos = token.text, token.tag_, token.pos_
            else:
                word, tag, pos = "", "", ""

            if word == 'claims' and tag not in ('VBD', 'VBZ'):
                expl = f"this post {inference}"
            elif tag in ('VBD', 'VBZ'):
                expl = f"this post {inference}"
            elif (tag == 'VBG' and word not in ('implying', 'calling')) or word == 'being':
                expl = f"this post implies that {inference}"
            elif word in ('implying', 'calling'):
                expl = f"this post is {inference}"
            elif word == 'rape':
                expl = f"this post implies that {target_minority} {inference}"
            elif tag in ('MD', 'VB') or (pos == 'AUX' and word != 'being'):
                if word == 'is' and (
                    target_minority in ("women", "men")
                    or "folks" in target_minority
                    or ',' in target_minority
                ):
                    rest = " ".join(inference.split()[1:])
                    expl = f"this post implies that {target_minority} are {rest}"
                else:
                    expl = f"this post implies that {target_minority} {inference}"
            elif word == 'all':
                expl = f"this post implies that {target_minority} are {inference}"
            else:
                expl = f"this post implies that {inference}"

            explanations.append(expl)

        out.at[idx, 'selectedStereotype'] = random.choice(explanations) if explanations else None

    return out


def run(self):
    # Aggregate splits
    sbic_train = self._aggregate_annotations('trn')
    sbic_dev = self._aggregate_annotations('dev')
    sbic_test = self._aggregate_annotations('tst')
    for df_ in (sbic_train, sbic_dev, sbic_test):
        df_['offensiveLABEL'] = np.where(
            df_['offensiveYN'] >= 0.5,
            'offensive',
            'not_offensive'
        )
    os.makedirs(self.output_dir, exist_ok=True)
    sbic_train['aug_sent1_of_post'] = pd.Series(dtype="object")
    sbic_train['aug_sent2_of_post'] = pd.Series(dtype="object")
    for i, post in enumerate(sbic_train['post']):
        sbic_train.at[i, 'aug_sent1_of_post'] = self.aug.augment(post)
        sbic_train.at[i, 'aug_sent2_of_post'] = self.aug.augment(post)
    sbic_train = self._turn_implied_statements_to_explanations('trn', sbic_train)
    sbic_train.to_csv(os.path.join(self.output_dir, "train.csv"), index=False)
