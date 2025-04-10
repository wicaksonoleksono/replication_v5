````conda env create -f env.yml --name <env_name>

conda activate <env_name>```
````

python main.py --config "\*.yaml"

python preprocess.py --config "./config_preprocess.yaml"

or

conda create -n torch_env python=3.12.8 numpy pandas matplotlib seaborn scikit-learn scipy jupyterlab
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers nlpaug pickle tqdm PyYAML
