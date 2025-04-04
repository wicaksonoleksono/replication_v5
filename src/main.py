import argparse
import yaml 
import os 
from itertools import product
from modules import update_progress,load_progress,reset_progress,set_seed
from pipeline import pipeline
def parse_args(args=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,required=True)
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
def main(args=None):
    all_combinations=[]
    parsed_args = parse_args(args)
    config = load_config(parsed_args.config)
    # 1. Glob param . 

    data_path = config.get("data_path") # str
    output_base = config.get("output_base") # str 
    
    encoders=config.get("encoders") # str
    learning_rates=config.get("learning_rates") #float64
    batch_size = config.get("batch_sizes") #int
    lambda_weights=config.get("lambda_weights") #float64
    num_epochs=config.get("num_epochs") # int
    seed = config.get("seed")

    os.makedirs(output_base,exist_ok=True)
    progress_path=os.path.join(output_base,"progress.json")
    # 2. Parse datamains+methods 
    data_mains_config = config.get("data_mains")
    all_combinations = []
    for data_main_dict in data_mains_config:
        data_main_name = data_main_dict["name"]
        methods_list = data_main_dict.get("methods")
        for method_dict in methods_list:
            method_name = method_dict["name"]
            if method_name == "contrastive":
                temperatures = method_dict.get("temperatures")
                # ------------------------------------------------------
                #    CONTRASTIVE: product of (encoders, lr, lam, temperature)
                # ------------------------------------------------------
                for (enc, lr, lam, temp) in product(encoders, learning_rates, lambda_weights, temperatures):
                    combo = {
                        "data_main": data_main_name,
                        "method": method_name,
                        "encoder": enc,
                        "learning_rate": lr,
                        "lambda_weight": lam,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "output_base": output_base,
                        # method-specific
                        "temperature": temp,
                        "margin": None,
                        "fallback": None,
                        "reducer": None,
                        "beta": None,
                    }
                    all_combinations.append(combo)
            elif method_name == "semi-hard":
                fallback_vals = method_dict.get("fallback")
                margins = method_dict.get("margins")
                reducers_list = method_dict.get("reducers" )
                # ------------------------------------------------------
                #   SEMI-HARD: product of (encoders, lr, lam, margins, fallback, reducers, possibly beta)
                # ------------------------------------------------------
                all_reducer_beta_pairs = []
                for r in reducers_list:
                    if isinstance(r, dict):
                        reducer_name = r["name"]
                        for b in r.get("beta_values", [None]):
                            all_reducer_beta_pairs.append((reducer_name, b))
                    else:
                        all_reducer_beta_pairs.append((r, None))

                for (enc, lr, lam, fb, marg, (reducer_name, beta_val)) in product(
                    encoders, learning_rates, lambda_weights, fallback_vals, margins, all_reducer_beta_pairs
                ):
                    combo = {
                        "data_main": data_main_name,
                        "method": method_name,
                        "encoder": enc,
                        "learning_rate": lr,
                        "lambda_weight": lam,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "output_base": output_base,
                        # method-specific
                        "temperature": None,
                        "margin": marg,
                        "fallback": fb,
                        "reducer": reducer_name,
                        "beta": beta_val
                    }
                    all_combinations.append(combo)
            # You could have more elif blocks for other method names (For replication)
    total_combos = len(all_combinations)
    print(f"Found {total_combos} total combinations to run.")
    progress_data = load_progress(progress_path)
    if progress_data.get("total_combinations") != total_combos:
        print("Detected different combination count or initial run. Resetting progress.")
        progress_data = {
            "last_completed_index": -1,
            "total_combinations": total_combos
        }
        update_progress(progress_data, progress_path)
    start_index = progress_data["last_completed_index"] + 1
    print(f"Resuming from combination index {start_index} of {total_combos }.")
    for idx in range(start_index, total_combos):
        combo = all_combinations[idx]
        print(
            f"\n=== Combo {idx + 1}/{total_combos} ===\n"
            f"data_main={combo['data_main']} | method={combo['method']} | encoder={combo['encoder']} | "
            f"learning_rate={combo['learning_rate']} | lambda_weight={combo['lambda_weight']}\n"
            f"temperature={combo['temperature']} | margin={combo['margin']} | fallback={combo['fallback']} | "
            f"reducer={combo['reducer']} | beta={combo['beta']}\n"
        )
        pipeline(
        data_path=data_path,
        output_base=combo["output_base"],
        data_main=combo["data_main"],
        seed=seed,


        encoder_name=combo["encoder"],
        learning_rate=combo["learning_rate"],
        batch_size=combo["batch_size"],
        num_epochs=combo["num_epochs"],
        lambda_weight=combo["lambda_weight"],
        method=combo["method"],

        # triplet loss
        margin=combo["margin"] if combo["method"] == "semi-hard" else None,
        beta=combo["beta"] if combo["method"] == "semi-hard" else None,
        reducer=combo["reducer"] if combo["method"] == "semi-hard" else None,
        fallback=combo["fallback"] if combo["method"] == "semi-hard" else None,
        # Contrastive
        temperature=combo["temperature"] if combo["method"] == "contrastive" else None,
        )
        progress_data["last_completed_index"] = idx
        update_progress(progress_data, progress_path)
        # 6. If we finished all combos, optionally reset progress
    # ------------------------------------------------------
    if progress_data["last_completed_index"] >= total_combos - 1:
        print("\nAll combinations have completed successfully!")
        reset_progress(progress_path)
if __name__ == "__main__":
    main()