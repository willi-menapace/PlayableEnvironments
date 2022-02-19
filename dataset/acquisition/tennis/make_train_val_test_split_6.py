import argparse
import os
from distutils.dir_util import copy_tree
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    '''
    Divides a dataset in a directory in train, validation and test splits
    '''

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, required=True, help="Directory with the unsplitted dataset")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory where to output the splitted dataset")
    parser.add_argument("--annotations_file_name", type=str, default="dataset/acquisition/tennis/tennis_annotations/tennis_youtube_splits.csv")

    arguments = parser.parse_args()

    dataset_directory = arguments.dataset_directory
    output_directory = arguments.output_directory
    annotations_file_name = arguments.annotations_file_name

    # Reads the video annotations
    splits = pd.read_csv(annotations_file_name)

    # Paths of the split directories
    splits_directories = {
        "train": os.path.join(output_directory, "train"),
        "val": os.path.join(output_directory, "val"),
        "test": os.path.join(output_directory, "test"),
    }
    # id for the next video assigned to a split
    splits_counters = {
        "train": 0,
        "val": 0,
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

        source_path = os.path.join(dataset_directory, f"{current_sequence_id:05d}")
        target_path = os.path.join(splits_directories[current_split], f"{current_split_sequence_id:05d}")

        print(f"- Copying '{source_path}' to '{target_path}'")
        copy_tree(source_path, target_path)
























