import argparse
import glob
import os
import random

import pandas as pd


def is_acquisition_successful(directory: str):
    '''
    Checks whether the sequence was successfully acquired or not
    :param directory:
    :return:
    '''

    success = False

    # If the frames have already been successfully extracted
    if os.path.isfile(os.path.join(directory, "frame_extraction_success.txt")):
        success = True

    # If a failure happened during previous preprocessing, do not extract frames
    if os.path.isfile(os.path.join(directory, "failure.txt")) or os.path.isfile(os.path.join(directory, "boxes_failure.txt")):
        if success:
            raise Exception(f"Directory {directory} has both success and failure files, something went wrong during dataset preprocessing")

        success = False

    return success

def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, required=True, help="Directory with the unsplitted dataset")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory where to output the splits file")
    parser.add_argument("--annotations_file_name", type=str, default="tennis_youtube_splits.csv")
    parser.add_argument("--val_test_sequences", type=int, default=200, help="Number of sequences to include into the train and validation splits")

    arguments = parser.parse_args()

    dataset_directory = arguments.dataset_directory
    output_directory = arguments.output_directory
    annotations_file_name = arguments.annotations_file_name
    val_test_sequences = arguments.val_test_sequences

    successfully_acquired_videos = []
    video_folders = list(sorted(glob.glob(os.path.join(dataset_directory, "*"))))
    for current_video_folder in video_folders:
        inner_directory = os.path.join(current_video_folder, "00000")

        if is_acquisition_successful(inner_directory):
            successfully_acquired_videos.append(current_video_folder)

    # Shuffles the annotations
    random.shuffle(successfully_acquired_videos)

    test_begin_index = len(successfully_acquired_videos) - val_test_sequences
    val_begin_index = len(successfully_acquired_videos) - 2 * val_test_sequences

    splits = ["train"] * (len(successfully_acquired_videos) - 2 * val_test_sequences)
    splits += ["val"] * val_test_sequences
    splits += ["test"] * val_test_sequences

    sequences = [int(os.path.basename(file)) for file in successfully_acquired_videos]

    dataframe = pd.DataFrame({
        "sequence": sequences,
        "split": splits
    })

    # Saves the splits to csv
    dataframe.to_csv(os.path.join(output_directory, annotations_file_name), index=False)

    print("- All Done")


if __name__ == "__main__":
    '''
    Extracts the subsequences of the youtube videos that contain relevant content
    '''

    main()