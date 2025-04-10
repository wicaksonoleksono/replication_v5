````conda env create -f env.yml --name <env_name>

conda activate <env_name>```
````

python main.py --config "\*.yaml"

python preprocess.py --config "./config_preprocess.yaml"

or

```
conda create -n torch_env python=3.12.8 numpy pandas matplotlib seaborn scikit-learn scipy jupyterlab
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers nlpaug tqdm PyYAML spacy nltk
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
