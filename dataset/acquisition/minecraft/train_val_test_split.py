import argparse
import os
import shutil
from distutils.dir_util import copy_tree
from pathlib import Path

import pandas as pd

splits_file = "dataset/acquisition/minecraft/minecraft_annotations/splits.csv"
root_directory = "data/tmp_2_minecraft_v1"
output_directory = "data/minecraft_v1"

copy = True


if __name__ == "__main__":
    '''
    Divides a dataset in a directory in train, validation and test splits
    '''

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_file", type=str, required=True)
    parser.add_argument("--root_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--copy", action="store_true")

    arguments = parser.parse_args()

    splits_file = arguments.splits_file
    root_directory = arguments.root_directory
    output_directory = arguments.output_directory
    copy = arguments.copy

    # Reads the video annotations
    splits = pd.read_csv(splits_file)

    # Paths of the split directories
    splits_directories = {
        "train": os.path.join(output_directory, "train"),
        "validation": os.path.join(output_directory, "val"),
        "test": os.path.join(output_directory, "test"),
    }
    # id for the next video assigned to a split
    splits_counters = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    print("- Creating output directories")
    # Creates the output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    for current_directory in splits_directories.values():
        Path(current_directory).mkdir(parents=True, exist_ok=True)

    for current_idx in range(len(splits)):
        current_sequence_id = splits.iloc[current_idx]["sequence"]
        current_split = splits.iloc[current_idx]["split"]
        current_split_sequence_id = splits_counters[current_split]
        splits_counters[current_split] = splits_counters[current_split] + 1

        source_path = os.path.join(root_directory, f"{current_sequence_id:05d}")
        target_path = os.path.join(splits_directories[current_split], f"{current_split_sequence_id:05d}")

        # Decides whether to copy or to move
        if copy:
            print(f"- Copying '{source_path}' to '{target_path}'")
            copy_tree(source_path, target_path)
        else:
            print(f"- Moving '{source_path}' to '{target_path}'")
            shutil.move(source_path, target_path)
























