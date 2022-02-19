import argparse
import glob
import os
from pathlib import Path

from dataset.multicamera_video import MulticameraVideo

if __name__ == "__main__":
    '''
    Subsamples videos in a dataset and makes them fixed length
    '''

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directories", type=str, nargs="+", help="Directories with the variable length dataset split")
    parser.add_argument("--output_directories", type=str, nargs="+", help="Directory where to output the fixed length dataset split")
    parser.add_argument("--frame_skip", type=int, default=0)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--extension", type=str, default="png")

    arguments = parser.parse_args()

    root_directories = arguments.root_directories
    output_directories = arguments.output_directories
    frame_skip = arguments.frame_skip
    sequence_length = arguments.sequence_length
    extension = arguments.extension

    for root_directory, output_directory in zip(root_directories, output_directories):

        current_output_idx = 0
        # Creates the output directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # Gets all directories
        video_paths_list = sorted(glob.glob(os.path.join(root_directory, "*")))
        video_paths_list = [current_path for current_path in video_paths_list if os.path.isdir(current_path)]

        for current_video_path in video_paths_list:

            print(f"- Splitting sequence '{current_video_path}'")
            # Split the video
            current_video = MulticameraVideo()
            current_video.load(current_video_path)
            video_splits = current_video.subsample_split_resize(frame_skip, sequence_length)
            print(f"  - Sequence split to {len(video_splits)} sequences'")

            # Save each output sequence
            for current_split in video_splits:
                output_path = os.path.join(output_directory, f"{current_output_idx:05d}")
                current_output_idx += 1

                print(f"  - Saving split to '{output_path}'")
                current_split.save(output_path, extension=extension)























