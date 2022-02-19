import argparse
import copy
import multiprocessing as mp
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from dataset.acquisition.utils.video_fragmenter import VideoFragmenter
from dataset.calibration.tennis.field_calibrator import FieldCalibrator
from utils.lib_3d.pose_parameters import PoseParametersNumpy

x_singles = 4.115
x_doubles = 5.485

y_small = 6.40
y_large = 11.885

# World coordinates for tennis court keypoints
world_points = [
    (-x_doubles, y_large, 0),
    (-x_doubles, -y_large, 0),
    (x_doubles, -y_large, 0),
    (x_doubles, y_large, 0),
    (-x_singles, y_large, 0),
    (-x_singles, -y_large, 0),
    (x_singles, -y_large, 0),
    (x_singles, y_large, 0),
    (-x_singles, y_small, 0),
    (x_singles, y_small, 0),
    (-x_singles, -y_small, 0),
    (x_singles, -y_small, 0),
    (0.0, y_small, 0),
    (0.0, -y_small, 0),
    (-x_doubles, 0.0, 0),
    (x_doubles, 0.0, 0),
]

world_points = np.asarray(world_points, dtype=np.float32)

def check_missing_values_density(values):
    '''
    Checks whether the number of missing values is admissible
    :param values:
    :return: True if the number of missing values is admissible
    '''

    nones = [value for value in values if value is None]

    none_fraction = len(nones) / len(values)

    if none_fraction > 0.333:
        return False
    else:
        return True

def is_valid_variance(cameras: PoseParametersNumpy):
    '''
    Checks whether variance in the camera positions is excessive or not
    :param cameras:
    :return: True if variance in camera positions is admissible
    '''

    all_params = []
    for current_camera in cameras:
        all_params.append(current_camera.translation)
    all_params = np.stack(all_params, axis=0)
    var = all_params.var(axis=0)

    # Empirically, we determine that bad videos have a variance greater than 8 on the y axis, so we reject them
    return float(var[1]) < 8.0

def is_valid(rotations: np.ndarray, translations: np.ndarray, focal: float) -> bool:
    '''
    Checks whether the camera parameters are plausible

    :param rotations: (3) array with camera rotations
    :param translations: (3) array with camera translations
    :param focal: focal length
    :return: True if the given camera parameters are plausible
    '''

    print(f"Current translations: {translations}\nCurrent rotations: {rotations}\nCurrent focal: {focal}")

    # Defines the valid ranges of camera positions
    # Camera position on y should be ~ -39.5 Wimbledon ~-34.0 Roland ~-32.4 US ~-38.7 AO
    # Camera position on z should be ~ 8.5 Wimbledon ~10.0 Roland ~9.8 US  ~9.24 AO
    # Camera position on x should be ~ 0.0 Wimbledon ~0.0 Roland ~0.0 US  ~-1.1 AO
    valid_ranges = [
        [[-1.5, 1.5], [-40.0, -38.0], [7.5, 9.5]],  # Wimbledon
        [[-1.5, 1.5], [-35.5, -32.5], [9.0, 11.0]],  # Roland
        [[-1.5, 1.5], [-34.0, -31.0], [8.8, 10.8]],  # US
        [[-2.5, 1.5], [-40.0, -37.0], [8.3, 10.3]]  # AO
    ]

    # For each range, check if its constraints are satisfied. If none is satisfied, the camera is not valid
    for current_range in valid_ranges:
        range_satisfied = True
        for current_translation, current_range in zip(translations, current_range):
            if current_translation < current_range[0] or current_translation > current_range[1]:
                range_satisfied = False
        if range_satisfied:
            return True

    return False

def load_points(filename: str) -> np.ndarray:
    '''
    Loads the points from the given file

    :param filename:
    :return: (points_count, 2) array of points, None if the points could not be read
    '''

    all_points = []

    if not os.path.exists(filename):
        return None

    try:
        # Reads one point per line
        with open(filename) as file:
            all_lines = file.readlines()
            for current_line in all_lines:
                splits = current_line.split(";")
                all_points.append((float(splits[0]), float(splits[1])))

        return np.asarray(all_points).astype(np.float32)

    except IOError:
        return None


def fix_sequence(begin_idx: int, end_idx: int, calibration_results: List):
    '''
    Fixes in place the gap between two specified indexes in the calibration results

    :param begin_idx:
    :param end_idx:
    :param calibration_results:
    :return:
    '''

    begin_values = calibration_results[begin_idx]
    end_values = calibration_results[end_idx]
    steps = end_idx - begin_idx

    # For each position in the gap
    for current_step in range(1, steps):
        current_results = []
        # For each value to interpolate at that step
        for idx in range(len(begin_values)):
            # Computes and add the linear interpolation delta
            delta = (end_values[idx] - begin_values[idx]) / steps
            current_results.append(begin_values[idx] + delta * current_step)

        current_results = tuple(current_results)
        calibration_results[begin_idx + current_step] = current_results


def smoothen_sequence(calibration_results: List, window_size=1):
    '''
    Smooths a sequence of calibration parameters in place

    :param calibration_results: List of calibration parameters to smooth
    :param window_size: Size of the window to use for smoothing. The current element and the window_size preceding and
                        succeeding elements are used for the smoothing
    :return:
    '''

    cumsum = []

    # Computes the comulative sum of elements
    first_result = [0 * element for element in calibration_results[0]]
    cumsum.append(tuple(first_result))

    for current_element in calibration_results:

        current_result = [first + second for first, second in zip(cumsum[-1], current_element)]
        cumsum.append(tuple(current_result))

    results_count = len(calibration_results)
    for idx in range(results_count):
        # if the current window is too big for the current element skip it
        if idx < window_size or results_count - window_size - 1 < idx:
            continue

        current_smoothed_element = [(first - second) / (2 * window_size + 1) for first, second in zip(cumsum[idx + window_size + 1], cumsum[idx - window_size])]
        calibration_results[idx] = current_smoothed_element


def add_missing_values(calibration_results: List):
    '''
    Fills in place the missing calibration results with interpolated ones

    :param calibration_results: List of calibration results (rotations, translations, focals)
    :return:
    '''

    results_count = len(calibration_results)

    # Finds the first and last elements
    first_result_idx = -1
    last_result_idx = -1
    for idx, element in enumerate(calibration_results):
        if first_result_idx == -1 and element is not None:
            first_result_idx = idx
        if element is not None:
            last_result_idx = idx

    if first_result_idx == -1:
        raise Exception("No valid calibration result is present in the sequence")

    # Fills initial and final missing values with the closest non missing value
    for idx in range(0, first_result_idx):
        calibration_results[idx] = copy.deepcopy(calibration_results[first_result_idx])
    for idx in range(last_result_idx + 1, results_count):
        calibration_results[idx] = copy.deepcopy(calibration_results[last_result_idx])

    # Fix every gap between successful camera calibration by interpolation
    void_begin_index = -1
    none_seen = False
    for idx in range(first_result_idx, last_result_idx + 1):

        current_element = calibration_results[idx]

        # Found end of a gap with a None
        if current_element is not None and none_seen:
            # Fixes the gap
            fix_sequence(void_begin_index, idx, calibration_results)

        # Records that a none was seen
        if current_element is None:
            none_seen = True
        # Found potential beginning of a gap
        if current_element is not None:
            void_begin_index = idx
            none_seen = False


def calibrate_camera_folder(path: str, image_extension: str) -> bool:
    '''
    Extracts camera intrinsics and extrinsics for a given folder

    :param path: path of the folder to process
    :return: True if the folder was calibrated successfully
    '''

    print(f"- Calibrating '{path}'")

    calibration_is_good = True  # Keeps track of whether calibration for the current sequence was ok or not

    calibration_results = []

    frame_filenames = VideoFragmenter.get_generated_images(path, image_extension)

    # Some video splits may be empty, regard this as a calibration failure
    if len(frame_filenames) == 0:
        calibration_is_good = False

    if calibration_is_good:
        image_size = Image.open(frame_filenames[0]).size

        for current_frame_filename in frame_filenames:

            # Name of the image without extension
            base_filename = current_frame_filename[:-(len(image_extension) + 1)]
            calibration_points_filename = base_filename + f".txt"

            image_points = load_points(calibration_points_filename)

            results = None
            # If image points could be extracted
            if image_points is not None:
                # Calibrate the camera
                unvalidated_results = FieldCalibrator.calibrate_camera(world_points, image_points, image_size)
                # Add the results if they are plausible
                if is_valid(*unvalidated_results):
                    results = unvalidated_results

            calibration_results.append(results)

        # If too many values are missing from calibration, abort the computation
        if not check_missing_values_density(calibration_results):
            calibration_is_good = False
        # Otherwise fix the sequence
        else:

            # Fixes missing or undeasible values in the calibration results
            add_missing_values(calibration_results)

            # Smoothens the calibration results
            smoothen_sequence(calibration_results)

            cameras = [PoseParametersNumpy(rotations, translations) for rotations, translations, _ in calibration_results]
            focals = [focal for _, _, focal in calibration_results]

            if not is_valid_variance(cameras):
                calibration_is_good = False

    # Outputs the cameras only if calibration went well
    if calibration_is_good:
        cameras_filename = "cameras.pkl"
        focals_filename = "focals.pkl"

        # Writes updated cameras and focals
        with open(os.path.join(path, os.path.join(path, cameras_filename)), 'wb') as f:
            pickle.dump(cameras, f)
        with open(os.path.join(path, os.path.join(path, focals_filename)), 'wb') as f:
            pickle.dump(focals, f)
    # Creates a file signalling the failure
    else:
        failure_filename = "failure.txt"
        failure_path = os.path.join(path, os.path.join(path, failure_filename))
        open(failure_path, 'a').close()

    return calibration_is_good


def process_video(args):
    '''
    Creates the camera annotations for a certain video, outputting them in a directory under the output root specified by the output index

    :param video_path: directory for the original videos
    :param output_root: root where the dataset is to be created
    :param output_index: index of the current video
    :param output_framerate: framerate for the output video
    :param output_size: (width, height) of the output video
    :param image_extension:
    :param line_extractor_command: location of the line extractor
    :param resume: whether the current folder may already contain results
    :return:
    '''

    video_path, output_root, output_index, output_framerate, output_size, image_extension, line_extractor_command, resume = args


    print(f"-- [{datetime.now()}] Computing video {video_path}")
    output_directory = os.path.join(output_root, f"{output_index:05d}", "00000")

    # Clean possible leftovers from previous executions
    VideoFragmenter.clean_frames(output_directory, image_extension)

    # If both cameras and focals are present or a failure file is present, the video must have been already calibrated
    if resume and (os.path.isfile(os.path.join(output_directory, "failure.txt")) or (os.path.isfile(os.path.join(output_directory, "cameras.pkl")) and os.path.isfile(os.path.join(output_directory, "focals.pkl")))):
        print(f"-- [{datetime.now()}] Video {video_path} already calibrated")

        return

    print(f"-- [{datetime.now()}] Extracting frames of {video_path}")
    # Extracts frames from the video and creates the output directory
    frames = VideoFragmenter.extract_frames(video_path, output_directory, output_framerate, output_size, image_extension)

    print(f"-- [{datetime.now()}] Extracting lines of {video_path}")

    command_parameters = [line_extractor_command, output_directory]
    subprocess.run(command_parameters, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    print(f"-- [{datetime.now()}] Calibrating cameras of {video_path}")
    # Creates camera annotations
    calibration_ok = calibrate_camera_folder(output_directory, image_extension)

    print(f"-- [{datetime.now()}] Completed calibration with result ({calibration_ok}) of {video_path}")

    # Cleans all the frames
    VideoFragmenter.clean_frames(output_directory, image_extension)


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_directory", type=str, required=True, help="Directory with splitted youtube videos")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory where to output the dataset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--image_extension", type=str, default="png")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--framerate", type=int, default=5, help="Dataset output framerate")
    parser.add_argument("--height", type=int, default=560, help="Dataset output height")
    parser.add_argument("--width", type=int, default=996, help="Dataset output width")
    parser.add_argument("--line_extractor", type=str, default="dataset/acquisition/court_detector/detect", help="Directory where to output the dataset")

    arguments = parser.parse_args()

    video_directory = arguments.video_directory
    output_directory = arguments.output_directory
    resume = arguments.resume
    video_extension = arguments.video_extension
    image_extension = arguments.image_extension
    workers = arguments.workers
    framerate = arguments.framerate
    height = arguments.height
    width = arguments.width
    line_extractor = arguments.line_extractor

    output_size = (width, height)

    # Checks that the output directory is empty
    if not resume and  os.path.exists(output_directory) and any(os.scandir(output_directory)):
        raise Exception(f"Non-empty directory {output_directory} already exists")
    Path(output_directory).mkdir(exist_ok=True)

    # Gets all input videos
    input_videos = VideoFragmenter.get_videos(video_directory, video_extension)

    # Creates work items for splitting the videos
    work_items = []
    for video_idx, current_video in enumerate(input_videos):
        current_work_item = current_video, output_directory, video_idx, framerate, output_size, image_extension, line_extractor, resume
        work_items.append(current_work_item)

    pool = mp.Pool(workers)

    print("- Start extracting cameras")
    #for item in work_items:
    #    process_video(item)
    pool.map(process_video, work_items)
    pool.close()
    print("- All Done")


if __name__ == "__main__":
    '''
    Extracts the subsequences of the youtube videos that contain relevant content
    '''

    main()