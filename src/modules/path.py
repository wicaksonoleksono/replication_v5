# src/modules/path.py
import sys
import os

# Get the absolute path to the 'src' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, '..'))  # This gets "src/"

# Add 'src' to sys.path if it's not already there
if src_root not in sys.path:
    sys.path.insert(0, src_root)
