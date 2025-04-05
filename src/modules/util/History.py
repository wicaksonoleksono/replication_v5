import os
import json
import torch
from torch import nn

class HistoryTracker:
    def __init__(self, output_path):
        self.output_path = output_path
        self.history = {
            "train": {},
            "valid": {},
            "best": {"f1_macro": -1, "epoch": -1,"path": None}
        }
        os.makedirs(output_path, exist_ok=True)
    def save_model(self, model, optimizer, epoch):
        """Save model and optimizer state"""
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": self.history
        }
        path = os.path.join(self.output_path, f"epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        return path
    def load_model(self, path, model, optimizer=None):
        """Load model and optionally optimizer state"""
        checkpoint = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(checkpoint["model_state"])
        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        return model, optimizer, checkpoint.get("epoch", 0)
    def update_loss(self, epoch, batch_idx, mixed_loss, ce_loss, metric_loss):
        """Update batch-level losses"""
        epoch_key = f"epoch_{epoch}"
        if epoch_key not in self.history["train"]:
            self.history["train"][epoch_key] = {
                "losses": [],
            }
            
        self.history["train"][epoch_key]["losses"].append({
            "batch": batch_idx,
            "mixed_loss": mixed_loss,
            "ce_loss": ce_loss,
            "metric_loss": metric_loss
        })
    def update_metrics(self, epoch,avg_loss ,acc, f1_m,f1_w, precision, recall, is_validation=False):
        """Update epoch-level metrics"""
        epoch_key = f"epoch_{epoch}"
        target = "valid" if is_validation else "train"
        if epoch_key not in self.history[target]:
            self.history[target][epoch_key] = {}
        self.history[target][epoch_key].update({
            "acc": acc,
            "f1_weighted": f1_w,
            "f1_macro":f1_m,
            "precision": precision,
            "recall": recall,
            "average_loss":avg_loss,
        })
    def best_f1_score(self, epoch, current_f1, model, optimizer):
        """Save best model and delete previous best if exists"""
        if current_f1 is None or not isinstance(current_f1, (int, float)):
            raise ValueError(f"Invalid F1 score: {current_f1}")
        # Get current best with type safety
        current_best = self.history["best"].get("f1_macro", -1)
        if not isinstance(current_best, (int, float)):
            current_best = -1
        # Comparison with safe values
        if current_f1 > current_best:
            # Delete previous best model if exists
            previous_best = self.history["best"].get("path")
            if previous_best and os.path.exists(previous_best):
                try:
                    os.remove(previous_best)
                    print(f"🗑️ Deleted previous best: {os.path.basename(previous_best)}")
                except Exception as e:
                    print(f"⚠️ Failed to delete previous best: {e}")
            
            # Save new best model
            best_path = os.path.join(self.output_path, f"best_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": self.history
            }, best_path)

            # Update history
            self.history["best"] = {
                "f1_macro": current_f1,
                "epoch": epoch,
                "path": best_path
            }
            self.save()
            return True
        return False
    def save(self):
        """Save history to JSON"""
        path = os.path.join(self.output_path, "training_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path
    def get_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.output_path) 
                    if f.startswith("epoch_") and "best" not in f]
        if not checkpoints:
            return None
        epochs = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
        latest_epoch = max(epochs)
        return os.path.join(self.output_path, f"epoch_{latest_epoch}.pth")
    @classmethod
    def load(cls, output_path):
        instance = cls(output_path)
        history_path = os.path.join(output_path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                loaded_history = json.load(f)
                instance.history = {
                    "train": loaded_history.get("train", {}),
                    "valid": loaded_history.get("valid", {}),
                    "best": {
                        **instance.history["best"],  # Defaults
                        **loaded_history.get("best", {})  # Loaded values
                    }
                }
        # Ensure numeric types after loading
        if not isinstance(instance.history["best"].get("f1_macro"), (int, float)):
            instance.history["best"]["f1_macro"] = -1
        if not isinstance(instance.history["best"].get("epoch"), int):
            instance.history["best"]["epoch"] = -1
        return instance