import json
import os

def load_progress(json_path='progress.json'):
    """Load progress from a JSON file or return a default progress dictionary if it doesn't exist."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {"last_completed_index": -1, "total_combinations": None}

def update_progress(progress_data, json_path='progress.json'):
    """Save the current progress data to a JSON file."""
    with open(json_path, "w") as f:
        json.dump(progress_data, f, indent=4)
        
def reset_progress(json_path='progress.json'):
    """Reset progress to default values."""
    default_progress = {"last_completed_index": -1, "total_combinations": None}
    with open(json_path, "w") as f:
        json.dump(default_progress, f, indent=4)
    print("Progress has been reset.")