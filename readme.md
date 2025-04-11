````conda env create -f env.yml --name <env_name>

conda activate <env_name>```
````

python main.py --config "\*.yaml"

python preprocess.py --config "./config_preprocess.yaml"

or

```

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers nlpaug tqdm PyYAML spacy nltk  numpy matplotlib seaborn scikit-learn scipy jupyterlab pandas
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger');nltk.download('averaged_perceptron_tagger_eng')"
```

```
python preprocess.py --config "./config_preprocess.yaml"

CONTRATIVE
python main.py --config "./config_contrastive_ihc.yaml"
python main.py --config "./config_contrastive_sbic.yaml"

SEMIHARD
python main.py --config "./config_semi_hard_ihc.yaml"
python main.py --config "./config_semi_hard_sbic.yaml"

```

new combo copy paste

```
8x instances. Hopes it's good.

python main.py --config "./configs/IHC/config_semi_hard_ihc_bert_ang.yaml"
python main.py --config "./configs/IHC/config_semi_hard_ihc_bert_cos.yaml"
python main.py --config "./configs/IHC/config_semi_hard_ihc_hbert_ang. yaml"
python main.py --config "./configs/IHC/config_semi_hard_ihc_hbert_cos. yaml"
python main.py --config "./configs/SBIC/config_semi_hard_sbic_bert_ang.yaml"
python main.py --config "./configs/SBIC/config_semi_hard_sbic_bert_cos.yaml"
python main.py --config "./configs/SBIC/config_semi_hard_sbic_hbert_ang. yaml"
python main.py --config "./configs/SBIC/config_semi_hard_sbic_hbert_cos. yaml
```
