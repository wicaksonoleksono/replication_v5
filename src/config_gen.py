#!/usr/bin/env python3
import os

BASE_DIR = "./configs"
PARALLEL_SLOTS = 9


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
    s_list, other_list = discover_configs(BASE_DIR)
    buckets = [[] for _ in range(PARALLEL_SLOTS)]

    # 1. Distribute initial s_ files (one per slot if available)
    s_slots_indices = []  # Keep track of which slots received an initial s_ file
    for i in range(min(len(s_list), PARALLEL_SLOTS)):
        buckets[i].append(s_list[i])
        s_slots_indices.append(i)

    # 2. Stack remaining s_ files onto slots that already started with an s_ file
    remaining_s_files_idx = PARALLEL_SLOTS
    if s_slots_indices:  # Only if there are slots that started with s_ files
        for i in range(remaining_s_files_idx, len(s_list)):
            # Round-robin onto s_slots_indices
            target_bucket_idx_in_s_slots = (i - remaining_s_files_idx) % len(s_slots_indices)
            actual_bucket_idx = s_slots_indices[target_bucket_idx_in_s_slots]
            buckets[actual_bucket_idx].append(s_list[i])

    # 3. Distribute "other" files
    # First, try to fill completely empty slots
    empty_slot_indices = [i for i, bucket in enumerate(buckets) if not bucket]
    other_files_current_idx = 0

    for slot_idx in empty_slot_indices:
        if other_files_current_idx < len(other_list):
            buckets[slot_idx].append(other_list[other_files_current_idx])
            other_files_current_idx += 1
        else:
            break  # No more other_list files to fill empty slots
    # Then, distribute any remaining "other" files round-robin across ALL slots
    if other_files_current_idx < len(other_list):
        for i in range(other_files_current_idx, len(other_list)):
            # The starting point for round-robin can be tricky.
            # Let's use a simple round-robin over all PARALLEL_SLOTS.
            # The offset ensures it doesn't just restart from slot 0 if some empty slots were filled.
            # (i - other_files_current_idx) is the count of files we are distributing in this step.
            # len(empty_slot_indices) where files were added is other_files_current_idx before this loop.

            # A simpler round-robin index:
            # Find the slot with the fewest items to append to, or just simple round robin.
            # For simplicity, let's do a round-robin that considers how many "other" files were
            # already placed in previously empty slots.

            # Count how many files were *already* placed (initial s_ + others in empty slots)
            # This is not a simple (sN + j) anymore due to prioritized s_ stacking.
            # We can just round-robin the remaining 'other' files across all slots.
            # The index for placing the k-th *remaining* other file.
            kth_remaining_other_file = i - other_files_current_idx
            target_bucket_idx = kth_remaining_other_file % PARALLEL_SLOTS
            buckets[target_bucket_idx].append(other_list[i])

    # 4) Split buckets for printing order (no change here from your original logic)
    # This sorting is for the final printout order, not for distribution.
    # The distribution logic above now handles the prioritization.
    # We can simplify the printing if the order is just slot by slot.
    # If you still want to sort buckets based on whether they *initially* started with s_
    # vs other, that logic needs to be based on the state *after step 1*.
    # For now, let's print them in slot order, which is simpler.

    # 5) Print commands
    # The original sorting for printout was based on the *first element*
    # If you want to maintain that, we can re-implement.
    # For now, a simpler print:
    print("# Commands grouped by parallel slot:")
    for idx, bucket in enumerate(buckets):
        if bucket:  # Only print if bucket is not empty
            args = " ".join(f'"{p}"' for p in bucket)
            print(f"# Slot {idx+1}")
            print(f'python main.py --config {args}')
            print("-" * 20)


if __name__ == "__main__":
    main()
