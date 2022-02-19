import argparse
import math
import subprocess
import glob
import os
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Dict, List

import yaml
from PIL import Image


def from_split_annotations_to_original_video_annotations(annotations: Dict, splits_duration: float) -> Dict[str, List[Tuple]]:
    '''
    Transforms the annotations from annotations relative to video splits to annotations relative to the original videos
    :param annotations: Dictionary with annotations
    :param splits_duration: Duration in seconds of a full split
    :return: Dictionary with translated annotations.
                original_video_name -> List of (begin, end) times in seconds for each split
    '''

    # Loads the new annotations translated to absolute values in the original video
    new_annotations = {}
    for current_video_split_name in sorted(annotations.keys()):

        # Gets the original name of the video
        basename_end_index = current_video_split_name.find("_split_")
        original_video_name = current_video_split_name[:basename_end_index] + "." + current_video_split_name.split(".")[-1]
        # Gets the id for the current split
        split_id = int(current_video_split_name[-9:-4])

        splits_in_original_video = []
        # Translates each segment to the original time in the full video. Ensure annotations are sorted
        annotations[current_video_split_name].sort(key=lambda x: x["start"])
        for current_annotation in annotations[current_video_split_name]:
            current_annotation_begin = current_annotation["start"]
            current_annotation_end = current_annotation["end"]

            if current_annotation_end > splits_duration:
                print(f"Warning: the current annotation {current_annotation} of video f{original_video_name} has an end time greater than the original splits duration of f{splits_duration}")

            # Begin and end seconds in the original video
            absolute_video_start = current_annotation_begin + split_id * splits_duration
            absolute_video_end = current_annotation_end + split_id * splits_duration

            splits_in_original_video.append((absolute_video_start, absolute_video_end))

        if original_video_name not in new_annotations:
            new_annotations[original_video_name] = []
        new_annotations[original_video_name].extend(splits_in_original_video)

    # Merges the splits so that video sequences spanning multiple segments are reunited
    new_annotations_unified = {}
    for original_video_name in new_annotations:

        unified_splits_in_original_video = []
        splits_in_original_video = new_annotations[original_video_name]

        if splits_in_original_video:
            unified_splits_in_original_video.append(splits_in_original_video[0])

        for idx in range(1, len(splits_in_original_video)):
            old_split = unified_splits_in_original_video[-1]
            current_split = splits_in_original_video[idx]

            old_split_end = old_split[1]
            current_split_begin = current_split[0]

            # If there are less than 0.5 seconds between the splits then it must be the same split
            # Patch the old split to end at the current end
            if current_split_begin - 0.5 < old_split_end:
                print(f"Unifying {old_split}, {current_split} in video {original_video_name}")
                unified_splits_in_original_video[-1] = (old_split[0], current_split[1])
            # Instead if they are different insert the new split
            else:
                unified_splits_in_original_video.append(current_split)

        if original_video_name not in new_annotations_unified:
            new_annotations_unified[original_video_name] = []
        # Inserts the unified splits in the new annotations
        new_annotations_unified[original_video_name].extend(unified_splits_in_original_video)

    return new_annotations_unified

def split_video(args):
    '''
    Splits the videos
    :param args:
    :return:
    '''

    current_video_path, current_split_time, output_video_path, splits_framerate, splits_resolution = args
    command_parameters = ["/usr/bin/ffmpeg", "-ss", str(current_split_time[0]), "-i", current_video_path, "-c:v", "libx264", "-preset", "slow", "-crf", "0", "-vf", f"scale={splits_resolution[0]}:{splits_resolution[1]}", "-r", str(splits_framerate), "-t", str(current_split_time[1] - current_split_time[0]), output_video_path]

    subprocess.run(command_parameters)

if __name__ == "__main__":
    '''
    Extracts the subsequences of the youtube videos that contain relevant content
    '''

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_directory", type=str, required=True, help="Directory with youtube videos")
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--annotations_path", type=str, required=True, help="Path to the youtube video temporal annotations")
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits_duration", type=float,  help="Duration of each split as annotated by the annotators")
    parser.add_argument("--splits_framerate", type=int, default=25)
    parser.add_argument("--splits_height", type=int, default=560)
    parser.add_argument("--splits_width", type=int, default=996)

    arguments = parser.parse_args()

    workers_count = arguments.workers

    video_directory = arguments.video_directory
    output_directory = arguments.output_directory
    video_extension = arguments.video_extension
    annotations_path = arguments.annotations_path
    workers = arguments.workers
    splits_duration = arguments.splits_duration
    splits_framerate = arguments.splits_framerate
    splits_height = arguments.splits_height
    splits_width = arguments.splits_width

    splits_resolution = (splits_width, splits_height)

    # Checks that the output directory is empty
    if os.path.exists(output_directory) and any(os.scandir(output_directory)):
        raise Exception(f"Non-empty directory {output_directory} already exists")
    Path(output_directory).mkdir(exist_ok=True)

    print("- Processing annotations")
    with open(annotations_path, 'r') as stream:
        try:
            annotations = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    unified_annotations = from_split_annotations_to_original_video_annotations(annotations, splits_duration)

    # Creates work items for splitting the videos
    work_items = []
    for original_video_name in unified_annotations:
        current_video_path = os.path.join(video_directory, original_video_name)
        split_times = unified_annotations[original_video_name]

        if len(original_video_name.split(".")) != 2:
            raise Exception(f"{original_video_name} has additional dots in its path, ensure videos have no additional dots before the extension")

        for split_idx, current_split_time in enumerate(split_times):
            output_video_path = os.path.join(output_directory, original_video_name.split(".")[0] + f"_split_{split_idx:05d}." + original_video_name.split(".")[-1])
            current_work_item = current_video_path, current_split_time, output_video_path, splits_framerate, splits_resolution
            work_items.append(current_work_item)

    pool = mp.Pool(workers)

    print("- Start splitting")
    pool.map(split_video, work_items)
    pool.close()

    print("- All Done")
