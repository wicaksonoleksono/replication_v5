import argparse
import yaml
import os
from itertools import product
from modules import update_progress, load_progress, reset_progress, set_seed
from pipeline import pipeline


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args(args)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args=None):
    all_combinations = []
    parsed_args = parse_args(args)
    config = load_config(parsed_args.config)
    # 1. Glob param .
    data_path = config.get("data_path")  # str
    output_base = config.get("output_base")  # str

    encoders = config.get("encoders")  # str
    learning_rates = config.get("learning_rates")  # float64
    batch_size = config.get("batch_size")  # int
    lambda_weights = config.get("lambda_weights")  # float64
    num_epochs = config.get("num_epochs")  # int
    seed = config.get("seed")  # seed
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
                        "method_dir": None,
                        # method-specific
                        "temperature": temp,
                        "am": None,
                        "margin": None,
                        "fallback": None,
                        "reducer": None,
                        "beta": None,
                        "distance_fn": None

                    }
                    all_combinations.append(combo)

            elif method_name == "semi-hard" or "SST":
                fallback_vals = method_dict["fallback"]
                margins = method_dict["margins"]
                d_fns = method_dict["d_fn"]
                reducers_list = method_dict["reducers"]
                all_reducer_beta_pairs = []
                for r in reducers_list:
                    if isinstance(r, str):
                        all_reducer_beta_pairs.append((r, None))
                        continue
                    reducer_names = r["name"]
                    if isinstance(reducer_names, str):
                        reducer_names = [reducer_names]
                    beta_values = r.get("beta", [])
                    for name in reducer_names:
                        if name in ["softmax", "softmax_sh", "sm_learnable"]:
                            if not beta_values:
                                all_reducer_beta_pairs.append((name, None))
                            else:
                                for b in beta_values:
                                    all_reducer_beta_pairs.append((name, b))
                        else:
                            all_reducer_beta_pairs.append((name, None))
                print(all_reducer_beta_pairs)
                all_combinations = []
                for (enc, lr, marg, lam, fb, d_fn, (reducer_name, beta_val)) in product(
                    encoders,
                    learning_rates,
                    margins,
                    lambda_weights,
                    fallback_vals,
                    d_fns,
                    all_reducer_beta_pairs
                ):
                    combo = {
                        "data_main":    data_main_name,
                        "method":       method_name,
                        "encoder":      enc,
                        "learning_rate": lr,
                        "lambda_weight": lam,
                        "batch_size":   batch_size,
                        "num_epochs":   num_epochs,
                        "method_dir":   None,
                        # method-specific
                        "temperature":  None,
                        "margin":       marg,
                        "fallback":     fb,
                        "reducer":      reducer_name,
                        "beta":         beta_val,
                        "distance_fn":  d_fn
                    }
                    all_combinations.append(combo)
    print(combo)
    for combo in all_combinations:
        encoder_short_name = "bert" if "bert-base-uncased" in combo["encoder"] else "hatebert"
        assert combo["encoder"] in ["bert-base-uncased", "GroNLP/hateBERT"], \
            f"Expected encoder to be one of ['bert-base-uncased', 'GroNLP/hateBERT'], got {combo['encoder']}"
        assert combo["learning_rate"] == 2e-5, \
            f"Expected learning_rate to be 2e-05, got {combo['learning_rate']}"
        assert combo["lambda_weight"] == 0.25, \
            f"Expected lambda_weight to be 0.25, got {combo['lambda_weight']}"
        assert combo["batch_size"] in [8, 16, 32], \
            f"Expected batch_size to be one of [8, 16, 32], got {combo['batch_size']}"
        assert 0 <= combo["num_epochs"] <= 6, \
            f"Expected num_epochs to be in the range [0, 6], got {combo['num_epochs']}"
        # Check method-specific parameters
        if combo["method"] in ("semi-hard", "SST"):
            assert 0.0 <= combo["margin"] <= 1.0, f"Expected margin between 0 and 1, got {combo['margin']}"
            assert isinstance(combo["fallback"], bool), \
                f"Expected fallback to be a boolean, got {combo['fallback']}"
            valid_reducers = ["mean", "sum", "softmax", "softmax_sh", "sm_learnable"]
            r = combo["reducer"]
            b = combo["beta"]
            assert r in valid_reducers, (
                f"Expected reducer to be one of {valid_reducers}, got {r}"
            )
            if r in ["softmax", "softmax_sh", "sm_learnable"]:
                assert b is not None, f"For reducer '{r}', beta must be set."
                assert 1 <= b <= 15, f"Expected beta to be in the range [1, 15], got {b}"
            else:
                assert b is None, f"For reducer '{r}', beta must be None."
            assert combo["distance_fn"] in ["angular", "cos", "angular_w", "cos_w", "angular_f", "cos_f", "angular_fw", "cos_fw"], \
                f"Expected distance_fn to be either 'angular' or 'cos','chord','scaled_chord','maha' got {combo['distance_fn']}"
        elif combo["method"] == "contrastive":
            assert combo["temperature"] == 0.3, \
                f"Expected temperature to be 0.3 for contrastive method, got {combo['temperature']}"
        else:
            raise ValueError(f"Unsupported method: {combo['method']}")
        if combo['method'] in ("semi-hard", "SST"):
            combo['method_dir'] = (
                f"{output_base}.{combo['method']}.{combo['data_main']}.{encoder_short_name}."
                f"{combo['distance_fn']}."
                f"{combo['reducer']}"
            )
            os.makedirs(combo['method_dir'], exist_ok=True)
        elif combo['method'] == "contrastive":
            combo['method_dir'] = (
                f"{output_base}.{combo['method']}."
                f"{combo['data_main']}.{encoder_short_name}"
            )
            os.makedirs(combo['method_dir'], exist_ok=True)
        else:
            raise ValueError(f"Unknown method: {combo['method']}")
    print(f"method dir :{combo['method_dir']}")
    progress_path = os.path.join(combo['method_dir'], "progress.json")
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
            f"reducer={combo['reducer']} | beta={combo['beta']} | dist_fn={combo['distance_fn']}\n"
        )
        pipeline(
            data_path=data_path,
            method_dir=combo['method_dir'],
            data_main=combo["data_main"],
            seed=seed,
            encoder_name=combo["encoder"],
            learning_rate=combo["learning_rate"],
            batch_size=combo["batch_size"],
            num_epochs=combo["num_epochs"],
            lambda_weight=combo["lambda_weight"],
            method=combo["method"],
            # triplet loss
            # if combo["method"] == "semi-hard" else None,
            d_fn=combo["distance_fn"],
            margin=combo["margin"],
            beta=combo["beta"],  # if combo["method"] == "semi-hard" else None,
            # if combo["method"] == "semi-hard" else None,
            reducer=combo["reducer"],
            # if combo["method"] == "semi-hard" else None,
            fallback=combo["fallback"],
            # Contrastive
            # if combo["method"] == "contrastive" else None,
            temperature=combo["temperature"],
        )

        progress_data["last_completed_index"] = idx
        update_progress(progress_data, progress_path)
    # ------------------------------------------------------
    if progress_data["last_completed_index"] >= total_combos - 1:
        print("\nAll combinations have completed successfully!")
        reset_progress(progress_path)


if __name__ == "__main__":
    main()
