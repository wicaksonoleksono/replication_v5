#!/usr/bin/env python3
import os
import argparse  # Import the argparse module

# Default values (can be overridden by command-line arguments)
DEFAULT_BASE_DIR = "./configs"
DEFAULT_PARALLEL_SLOTS = 9


def discover_configs(base_dir):
    s_list = []
    other_list = []
    for root, _, files in os.walk(base_dir):
        for fn in sorted(files):  # Sort files for consistent ordering
            if not fn.endswith(".yaml"):
                continue
            full = os.path.join(root, fn)
            if fn.startswith("s_"):
                s_list.append(full)
            else:
                other_list.append(full)
    return s_list, other_list


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Discover and distribute YAML config files into parallel slots.")
    parser.add_argument(
        "-b", "--basedir",
        default=DEFAULT_BASE_DIR,
        help=f"The base directory to search for config files (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "-ps", "--parallelslots",
        type=int,
        default=DEFAULT_PARALLEL_SLOTS,
        help=f"The number of parallel slots to distribute configs into (default: {DEFAULT_PARALLEL_SLOTS})"
    )
    args = parser.parse_args()

    # Use the parsed arguments
    base_dir_to_use = args.basedir
    parallel_slots_to_use = args.parallelslots
    # --- End Argument Parsing ---

    print(f"# Using BASE_DIR: {base_dir_to_use}")
    print(f"# Using PARALLEL_SLOTS: {parallel_slots_to_use}")
    print("-" * 20)

    s_list, other_list = discover_configs(base_dir_to_use)
    buckets = [[] for _ in range(parallel_slots_to_use)]

    # 1. Distribute initial s_ files (one per slot if available)
    s_slots_indices = []  # Keep track of which slots received an initial s_ file
    for i in range(min(len(s_list), parallel_slots_to_use)):
        buckets[i].append(s_list[i])
        s_slots_indices.append(i)

    # 2. Stack remaining s_ files onto slots that already started with an s_ file
    # Corrected index for remaining_s_files_idx to start from the next s_ file after initial distribution
    remaining_s_files_start_idx = min(len(s_list), parallel_slots_to_use)
    if s_slots_indices:  # Only if there are slots that started with s_ files
        for i in range(remaining_s_files_start_idx, len(s_list)):
            # Round-robin onto s_slots_indices
            # (i - remaining_s_files_start_idx) gives the 0-based index of the current *remaining* s_ file
            target_bucket_idx_in_s_slots = (i - remaining_s_files_start_idx) % len(s_slots_indices)
            actual_bucket_idx = s_slots_indices[target_bucket_idx_in_s_slots]
            buckets[actual_bucket_idx].append(s_list[i])

    # 3. Distribute "other" files
    empty_slot_indices = [i for i, bucket in enumerate(buckets) if not bucket]
    other_files_current_idx = 0

    # First, try to fill completely empty slots with "other" files
    for slot_idx in empty_slot_indices:
        if other_files_current_idx < len(other_list):
            buckets[slot_idx].append(other_list[other_files_current_idx])
            other_files_current_idx += 1
        else:
            break  # No more other_list files

    # Then, distribute any remaining "other" files round-robin across ALL slots
    if other_files_current_idx < len(other_list):
        for i in range(other_files_current_idx, len(other_list)):
            # k-th *remaining* other file we are distributing in this step
            kth_remaining_other_file = i - other_files_current_idx
            target_bucket_idx = kth_remaining_other_file % parallel_slots_to_use
            # To make the round-robin distribution a bit more even after filling empty slots,
            # we can try to find the slot with the fewest items, or simply continue round-robin.
            # For simplicity, let's find the least filled slot among all.
            # This ensures a more balanced distribution for the remaining 'other' files.

            # A better approach for distributing remaining 'other' files:
            # find the slot with the minimum number of items currently.
            # If multiple slots have the minimum, pick the first one.
            # This can be less predictable than pure round-robin but aims for balance.
            # For now, sticking to a predictable round-robin for the remaining:
            # The index should be based on the total number of "other" files distributed so far in this step
            # This ensures it continues the sequence across all slots.
            # Let's restart round-robin from slot 0 for the remaining 'other' files for simplicity here.
            # To ensure it's distributed fairly, we can sort slots by current load for the remainder.
            # However, a simple round-robin should suffice based on your previous logic.

            # Let's use a round-robin that starts from slot 0 for the remaining 'other' files.
            # The (i - other_files_current_idx) is the index of the file being placed in this loop.
            current_other_file_to_place_index = i - other_files_current_idx
            target_bucket_idx = current_other_file_to_place_index % parallel_slots_to_use
            buckets[target_bucket_idx].append(other_list[i])

    # 5) Print commands
    print("# Commands grouped by parallel slot:")
    for idx, bucket in enumerate(buckets):
        if bucket:  # Only print if bucket is not empty
            args_str = " ".join(f'"{p}"' for p in bucket)  # Renamed to avoid conflict
            print(f"# Slot {idx+1}")
            print(f'python main.py --config {args_str}')
            print("-" * 20)
        # else: # Optionally print for empty slots
            # print(f"# Slot {idx+1} (empty)")
            # print("-" * 20)


if __name__ == "__main__":
    main()
