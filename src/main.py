import argparse
import yaml 
import os 
from itertools import product
from modules import update_progress,load_progress,reset_progress,set_seed
from pipeline import pipeline
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args(args)
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
    batch_size = config.get("batch_size") #int
    lambda_weights=config.get("lambda_weights") #float64
    num_epochs=config.get("num_epochs") # int
    seed = config.get("seed") #seed
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
                fallback_vals = method_dict["fallback"]
                margins = method_dict["margins"]
                reducers_list = method_dict["reducers"]
                
                all_reducer_beta_pairs = []
                for r in reducers_list:
                    if isinstance(r, dict):
                        for b in r["beta_values"]:
                            all_reducer_beta_pairs.append( (r["name"], b) )
                    else:
                        all_reducer_beta_pairs.append( (r, None) )

                for (enc, lr, marg, lam, fb, (reducer_name, beta_val)) in product(
                        encoders, 
                        learning_rates, 
                        margins,          
                        lambda_weights,   
                        fallback_vals, 
                        all_reducer_beta_pairs
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
    for combo in all_combinations:
        method_dir = f"{combo['output_base']}.{combo['method']}"
        os.makedirs(method_dir, exist_ok=True)
        progress_path = os.path.join(method_dir, "progress.json")
        assert combo["encoder"] in ["bert-base-uncased", "GroNLP/hateBERT"], f"Expected encoder to be one of ['bert-base-uncased', 'GroNLP/hateBERT'], got {combo['encoder']}"        
        assert combo["learning_rate"] == 2e-5, f"Expected learning_rate to be 2e-05, got {combo['learning_rate']}"
        assert combo["lambda_weight"]==0.25 , f"Expected lambda_weight to be 0.25, got {combo['lambda_weight']}"
        assert combo["batch_size"] in [8, 16, 32], f"Expected batch_size to be one of [8, 16, 32], got {combo['batch_size']}"
        assert 0 <= combo["num_epochs"] <= 6, f"Expected num_epochs to be in the range [0, 6], got {combo['num_epochs']}"
        assert isinstance(combo["output_base"], str), f"Expected output_base to be a string, got {type(combo['output_base'])}"
        # Check method-specific parameters
        if combo["method"] == "semi-hard":
            assert combo["margin"] in [0.3,0.4,0.45], f"Expected margin to be in the range (0.3,0.4,0.5), got {combo['margin']}"
            assert isinstance(combo["fallback"], bool), f"Expected fallback to be a boolean, got {combo['fallback']}"
            assert combo["reducer"] in ["mean", "sum", "softmax"], f"Expected reducer_name to be one of ['mean', 'sum', 'softmax'], got {combo['reducer']}"
            if combo["reducer"] == "softmax":
                assert 5 <= combo["beta"] <= 15, f"Expected beta to be in the range (5, 15), got {combo['beta']}"
            else:
                assert combo["beta"] is None  
        elif combo["method"] == "contrastive":
            assert combo["temperature"] == 0.3, f"Expected temperatures to be 0.3 for contrastive method, got {combo['temperature']}"
        else:
            raise ValueError(f"Unsupported method: {combo['method']}")
    progress_path=os.path.join(f"{output_base}.{method_name}","progress.json")
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
            f"num epoch={combo['num_epochs']} | seed={seed} | batch_size={combo['batch_size']}|"
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
        margin=combo["margin"], #if combo["method"] == "semi-hard" else None,
        beta=combo["beta"], # if combo["method"] == "semi-hard" else None,
        reducer=combo["reducer"],# if combo["method"] == "semi-hard" else None,
        fallback=combo["fallback"],# if combo["method"] == "semi-hard" else None,
        # Contrastive
        temperature=combo["temperature"],#if combo["method"] == "contrastive" else None,
        )
        progress_data["last_completed_index"] = idx
        update_progress(progress_data, progress_path)
    # ------------------------------------------------------
    if progress_data["last_completed_index"] >= total_combos - 1:
        print("\nAll combinations have completed successfully!")
        reset_progress(progress_path)
if __name__ == "__main__":
    main()


    