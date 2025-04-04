import argparse
import yaml
import os
# ======= Import your existing modules ======= #
from modules import integration_dyna, preprocessor_dyna
from modules import preprocessor, integrate
from modules import integration_sbic, preprocessor_sbic
# from modules import perform_precluster
import warnings
warnings.simplefilter("ignore", FutureWarning)
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
def preprocess_dyna(data_home, out_dir, dyna_config):
    print(f"[INFO] Processing DynaHate data with data_home={data_home}, out_dir={out_dir}")
    print(f"      Using tokenizer={dyna_config['tokenizer_type']}, augmentation={dyna_config['augmentation']}")
    dataset_processor = integration_dyna(load_dir=os.path.join(data_home, "DynaHate"))
    dataset_processor.process()
    preproc = preprocessor_dyna(
        data_home=os.path.join(data_home, "DynaHate"),
        tokenizer_type=dyna_config["tokenizer_type"],
        output_dir=out_dir
    )
    preproc.process()
    print("[INFO] Finished DynaHate.")
def preprocess_ihc(data_home, out_dir, ihc_config):
    print(f"[INFO] Processing IHC data with data_home={data_home}, out_dir={out_dir}")
    print(f"      Using tokenizer={ihc_config['tokenizer_type']}, augmentation={ihc_config['augmentation']}")
    i = integrate(
        load_dir=os.path.join(data_home, "implicit-hate-corpus"),
        output_dir=out_dir
    )
    i.run()
    p = preprocessor(
        data_home=os.path.join(out_dir, "ihc_pure"),  # or if integration writes to data_home
        tokenizer_type=ihc_config["tokenizer_type"],
        augmentation=ihc_config["augmentation"],
        output_dir=out_dir
    )
    p.process()
    print("[INFO] Finished IHC.")
def preprocess_sbic(data_home, out_dir, sbic_config):
    print(f"[INFO] Processing SBIC data with data_home={data_home}, out_dir={out_dir}")
    print(f"      Using tokenizer={sbic_config['tokenizer_type']}, augmentation={sbic_config['augmentation']}")
    i = integration_sbic(
        load_dir=os.path.join(data_home, "SBIC"),
        output_dir=os.path.join(out_dir, "sbic_pure")
    )
    i.run()
    p = preprocessor_sbic(
        dataset="sbic",
        aug_type=sbic_config["augmentation"],
        tokenizer_type=sbic_config["tokenizer_type"],
        data_home=os.path.join(out_dir, "sbic_pure"),
        output_dir=out_dir
    )
  
    p.process()
    print("[INFO] Finished SBIC.")

def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Run data integration + preprocessing for DynaHate, IHC, and SBIC with YAML config."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)
    config = load_config(args.config)
    print("Configuration loaded:", config)
    data_home = config["data_home"]
    out_dir = config["out_dir"]
    chosen_dataset = config["dataset"]  
    dyna_config = config.get("dynahate", {})
    ihc_config = config.get("ihc", {})
    sbic_config = config.get("sbic", {})
    if chosen_dataset == "dyna":
        preprocess_dyna(data_home, out_dir, dyna_config)
    elif chosen_dataset == "ihc":
        preprocess_ihc(data_home, out_dir, ihc_config)
    elif chosen_dataset == "sbic":
        preprocess_sbic(data_home, out_dir, sbic_config)
    else:
        preprocess_dyna(data_home, out_dir, dyna_config)
        preprocess_ihc(data_home, out_dir, ihc_config)
        preprocess_sbic(data_home, out_dir, sbic_config)
if __name__ == "__main__":
    main()

