#!/usr/bin/env python3
import os

BASE_DIR = "./configs"
PARALLEL_SLOTS = 12


def discover_configs(base_dir):
    s_list = []
    other_list = []
    for root, _, files in os.walk(base_dir):
        for fn in sorted(files):
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
    for i, cfg in enumerate(s_list):
        buckets[i % PARALLEL_SLOTS].append(cfg)

    # 3) Append “other” files so that
    #    - if s_count < 12, the first empty slots get one other each
    #    - any remaining others get round-robined on top of all 12
    sN = len(s_list)
    for j, cfg in enumerate(other_list):
        idx = (sN + j) % PARALLEL_SLOTS
        buckets[idx].append(cfg)

    # 4) Split buckets into those starting with other vs s_, sort by original index
    other_buckets = []
    s_buckets = []
    for idx, bucket in enumerate(buckets):
        if not bucket:
            continue
        first_file = os.path.basename(bucket[0])
        if not first_file.startswith("s_"):
            other_buckets.append((idx, bucket))
        else:
            s_buckets.append((idx, bucket))

    other_buckets.sort(key=lambda x: x[0])
    s_buckets.sort(key=lambda x: x[0])

    # 5) Print commands: “other-first” buckets, then the pure-s_ buckets
    for _, bucket in other_buckets + s_buckets:
        args = " ".join(f'"{p}"' for p in bucket)
        print(f'python main.py --config {args}')


if __name__ == "__main__":
    main()
