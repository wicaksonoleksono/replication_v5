import argparse
import yaml
import os
from itertools import product
from modules import update_progress, load_progress, reset_progress, set_seed
from pipeline import pipeline
import math


def process(config_path):
    all_combinations = []
    config = load_config(config_path)
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
                        "distance_fn": None,

                        "cam_angular_margin_m": None,
                        "cam_lambda_a":         None,
                        "cam_lambda_r":         None,


                    }
                    all_combinations.append(combo)
            elif method_name == "cam":
                cam_angular_margins_m = method_dict.get(
                    "cam_angular_margins_m", [1.57])  # Example default: pi/2 radians
                cam_lambda_as = method_dict.get("cam_lambda_as", [1.0])           # Weight for attractor loss
                cam_lambda_rs = method_dict.get("cam_lambda_rs", [1.0])           # Weight for repeller loss
                print(
                    f"⚙️ Configuring CAM method with angular_margins: {cam_angular_margins_m}, lambda_as: {cam_lambda_as}, lambda_rs: {cam_lambda_rs}")
                all_combinations = []
                for (enc, lr, ang_m, l_a, l_r, lam) in product(
                    encoders,
                    learning_rates,
                    cam_angular_margins_m,
                    cam_lambda_as,
                    cam_lambda_rs,
                    lambda_weights
                ):
                    combo = {
                        "data_main":        data_main_name,
                        "method":           method_name,
                        "encoder":          enc,
                        "learning_rate":    lr,
                        "batch_size":       batch_size,    # Defined outside
                        "num_epochs":       num_epochs,    # Defined outside
                        "method_dir":       None,          # To be filled later
                        # CAM-specific parameters for the CamLoss class
                        "cam_angular_margin_m": ang_m,
                        "cam_lambda_a":         l_a,
                        "cam_lambda_r":         l_r,
                        # Set other method-specific params (from triplet, etc.) to None or omit
                        # if your downstream config processor expects them.
                        "lambda_weight":    lam,  # CAM has its own lambda_a, lambda_r
                        "temperature":      None,
                        "margin":           None,  # Triplet margin, CAM uses cam_angular_margin_m
                        "fallback":         None,
                        "reducer":          None,
                        "beta":             None,
                        "distance_fn":      None  # CAM uses angular distance internally
                    }
                    all_combinations.append(combo)
                print(f"Generated {len(all_combinations)} combinations for CAM.")
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
                for (enc, lr, m, lam, fb, d_fn, (reducer_name, beta_val)) in product(
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
                        "margin":       m,
                        "fallback":     fb,
                        "reducer":      reducer_name,
                        "beta":         beta_val,
                        "distance_fn":  d_fn,
                        "cam_angular_margin_m": None,
                        "cam_lambda_a":         None,
                        "cam_lambda_r":         None,
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
            # assert 0.0 <= combo["mine_margin"] <= 1.0, f"Expected margin between 0 and 1, got {combo['mine_margin']}"
            assert 0.0 <= combo["margin"] <= 1.0, f"Expected margin between 0 and 1, got {combo['margin']}"
            assert isinstance(combo["fallback"], bool), \
                f"Expected fallback to be a boolean, got {combo['fallback']}"
            valid_reducers = ["mean", "sum", "softmax", "softmax_sh",
                              "sm_learnable", "freedom_softmax", "freedom_softmax_sh"]
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
            assert combo["distance_fn"] in ["ang", "cos", "ang_w", "cos_w", "ang_f", "cos_f", "ang_fw", "cos_fw"], \
                f"Expected distance_fn to be either 'ang' or 'cos','chord','scaled_chord','maha' got {combo['distance_fn']}"
        elif combo["method"] == "contrastive":
            assert combo["temperature"] == 0.3, \
                f"Expected temperature to be 0.3 for contrastive method, got {combo['temperature']}"
        elif combo["method"] == "cam":
            cam_ang_m = combo.get("cam_angular_margin_m")
            assert cam_ang_m is not None, \
                f"CAM Config Error: 'cam_angular_margin_m' cannot be None. Combo: {combo}"
            assert isinstance(cam_ang_m, (int, float)), \
                f"CAM Config Error: 'cam_angular_margin_m' must be a number. Got type {type(cam_ang_m)} with value '{cam_ang_m}'. Combo: {combo}"
            # Angular margin should be positive and not excessively large (e.g., up to pi radians)
            assert 0 < cam_ang_m <= math.pi, \
                f"CAM Config Error: 'cam_angular_margin_m' ({cam_ang_m}) is out of the recommended range (0, pi approx {math.pi:.4f}]. Combo: {combo}"

            # Assertions for 'cam_lambda_a' (Attractor loss weight)
            cam_lambda_a_val = combo.get("cam_lambda_a")
            assert cam_lambda_a_val is not None, \
                f"CAM Config Error: 'cam_lambda_a' cannot be None. Combo: {combo}"
            assert isinstance(cam_lambda_a_val, (int, float)), \
                f"CAM Config Error: 'cam_lambda_a' must be a number. Got type {type(cam_lambda_a_val)} with value '{cam_lambda_a_val}'. Combo: {combo}"
            # Lambda weights are typically non-negative. Allowing 0 means the component can be turned off.
            assert 0.0 <= cam_lambda_a_val <= 10.0, \
                f"CAM Config Error: 'cam_lambda_a' ({cam_lambda_a_val}) is out of the recommended range [0.0, 10.0]. Combo: {combo}"
            # Assertions for 'cam_lambda_r' (Repeller loss weight)
            cam_lambda_r_val = combo.get("cam_lambda_r")
            assert cam_lambda_r_val is not None, \
                f"CAM Config Error: 'cam_lambda_r' cannot be None. Combo: {combo}"
            assert isinstance(cam_lambda_r_val, (int, float)), \
                f"CAM Config Error: 'cam_lambda_r' must be a number. Got type {type(cam_lambda_r_val)} with value '{cam_lambda_r_val}'. Combo: {combo}"
            assert 0.0 <= cam_lambda_r_val <= 10.0, \
                f"CAM Config Error: 'cam_lambda_r' ({cam_lambda_r_val}) is out of the recommended range [0.0, 10.0]. Combo: {combo}"

            # You might also check the global 'lambda_weight' for CAM if it's used for CCE+Metric combination
            global_lambda_w = combo.get("lambda_weight")
            assert global_lambda_w is not None, \
                f"CAM Config Error: Global 'lambda_weight' (for CCE+Metric) cannot be None. Combo: {combo}"
            assert isinstance(global_lambda_w, (int, float)), \
                f"CAM Config Error: Global 'lambda_weight' must be a number. Got type {type(global_lambda_w)} with value '{global_lambda_w}'. Combo: {combo}"
            assert 0.0 <= global_lambda_w <= 1.0, \
                f"CAM Config Error: Global 'lambda_weight' ({global_lambda_w}) is out of range [0.0, 1.0]. Combo: {combo}"

        elif combo["method"] in ["semi-hard", "SST"]:
            # Your existing assertions for 'lambda_weight' for these methods
            lambda_w = combo.get("lambda_weight")
            assert lambda_w is not None, \
                f"{combo['method']} Config Error: 'lambda_weight' cannot be None. Combo: {combo}"
            assert isinstance(lambda_w, (int, float)), \
                f"{combo['method']} Config Error: 'lambda_weight' must be a number. Got type {type(lambda_w)} with value '{lambda_w}'. Combo: {combo}"
            # Adjust range as per its meaning (e.g., if it's CCE+Metric weight, 0-1 is common)
            assert 0.0 <= lambda_w <= 1.0, \
                f"{combo['method']} Config Error: 'lambda_weight' ({lambda_w}) for {combo['method']} out of range [0.0, 1.0]. Combo: {combo}"
            # Add assertions for 'margin', 'reducer', 'beta', 'distance_fn', 'fallback' here
            # Example for margin:
            margin_val = combo.get("margin")
            assert margin_val is not None, \
                f"{combo['method']} Config Error: 'margin' cannot be None. Combo: {combo}"
            assert isinstance(margin_val, (int, float)), \
                f"{combo['method']} Config Error: 'margin' must be a number. Got {type(margin_val)}. Combo: {combo}"
            assert 0.0 < margin_val < 2.0, \
                f"{combo['method']} Config Error: 'margin' ({margin_val}) out of typical range (0.0, 2.0). Combo: {combo}"

        else:
            raise ValueError(f"Unsupported method: {combo['method']}")
        if combo['method'] in ("semi-hard", "SST"):
            combo['method_dir'] = (
                f"{output_base}.{combo['method']}.{combo['data_main']}.{encoder_short_name}."
                f"{combo['distance_fn']}."
                f"{combo['reducer']}"
            )
            os.makedirs(combo['method_dir'], exist_ok=True)
        elif combo['method'] in ["contrastive", "cam"]:
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
            am=combo["cam_angular_margin_m"],
            a=combo["cam_lambda_a"],
            r=combo["cam_lambda_r"],
        )

        progress_data["last_completed_index"] = idx
        update_progress(progress_data, progress_path)
    # ------------------------------------------------------
    if progress_data["last_completed_index"] >= total_combos - 1:
        print("\nAll combinations have completed successfully!")
        reset_progress(progress_path)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args(args=None):  # args=None allows it to pick up sys.argv by default from CLI
    parser = argparse.ArgumentParser(description="Process configurations for model training.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",  # Allows one or more config files
        required=True,  # This is fine, as we'll ensure arguments are passed
        help="One or more YAML config paths"
    )
    return parser.parse_args(args)  # Pass the list of args here


def main(args_list=None):
    args = parse_args(args_list)  # If args_list is None, argparse uses sys.argv

    for cfg_path in args.config:
        process(cfg_path)  # Call the refactored processing function

    print("\n🎉 All configurations processed.")


# --- This part is for when you run the script directly from command line ---
if __name__ == "__main__":
    # Example: python your_script_name.py --config config1.yaml config2.yaml
    main()
