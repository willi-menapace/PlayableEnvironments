import argparse
import glob
import os
from pathlib import Path

from dataset.video import Video

target_size = [1024, 576]
crop_region = [0, 0, 1024, 576]

if __name__ == "__main__":
    '''
    Subsamples videos in a dataset and makes them fixed length
    '''

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--frame_skip", type=int, required=True)
    parser.add_argument("--sequence_length", type=int, required=True)
    parser.add_argument("--min_sequence_length", type=int, required=True)
    parser.add_argument("--extension", type=str, default="png")

    arguments = parser.parse_args()

    root_directory = arguments.root_directory
    output_directory = arguments.output_directory
    frame_skip = arguments.frame_skip
    sequence_length = arguments.sequence_length
    min_sequence_length = arguments.min_sequence_length
    extension = arguments.extension


    current_output_idx = 0
    # Creates the output directory
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    # Gets all directories
    video_paths_list = sorted(glob.glob(os.path.join(root_directory, "*")))
    video_paths_list = [current_path for current_path in video_paths_list if os.path.isdir(current_path)]

    for idx, current_video_path in enumerate(video_paths_list):

        print(f"- Splitting sequence '{current_video_path}'")
        # Split the video
        current_video = Video()
        current_video.load(current_video_path)
        video_splits = current_video.subsample_split_resize(frame_skip, sequence_length, crop_region, target_size, min_sequence_length)
        print(f"  - Sequence split to {len(video_splits)} sequences'")

        # Save each output sequence
        for current_split in video_splits:



            output_path = os.path.join(output_directory, f"{current_output_idx:05d}")
            current_output_idx += 1

            print(f"  - Saving split to '{output_path}'")
            current_split.save(output_path, extension=extension)

        del current_video
        del video_splits
        del current_split























