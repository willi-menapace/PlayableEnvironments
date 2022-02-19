import argparse
import glob
import os
import pickle
import random
import re
from datetime import datetime
from typing import List

import torch
import torchvision
from PIL import Image
from dataset.acquisition.player_detector.tennis_player_detector import TennisPlayerDetector
from dataset.acquisition.utils.video_fragmenter import VideoFragmenter

from utils.drawing.bounding_box_drawer import BoundingBoxDrawer
from utils.lib_3d.ray_helper import RayHelper

x_singles = 4.115
x_doubles = 5.485

y_small = 6.40
y_large = 11.885

delta_y = 6.4
delta_x = 2.5

# World coordinates defining the portion of the field that a player is never expected to leave
world_points = [
    (-x_doubles - delta_x, y_large + delta_y, 0),
    (x_doubles + delta_x, y_large + delta_y, 0),
    (-x_doubles, 0, 0),
    (x_doubles, 0, 0),
    (-x_doubles - delta_x, -y_large - delta_y, 0),
    (x_doubles + delta_x, -y_large - delta_y, 0),
]

world_points = torch.tensor(world_points, dtype=torch.float)


def calibration_successful(directory: str) -> bool:
    '''
    Checks whether camera calibration was successful in this directory
    :param directory:
    :return:
    '''

    cameras_path = os.path.join(directory, "cameras.pkl")
    focals_path = os.path.join(directory, "focals.pkl")
    success = os.path.isfile(cameras_path) and os.path.isfile(focals_path)

    if not success:
        failure_path = os.path.join(directory, "failure.txt")
        failure_present = os.path.isfile(failure_path)
        if not failure_present:
            print(f"Warning: calibration unsuccessful and failure file not present in {directory}")

    return success


def detection_already_performed(directory: str) -> bool:
    '''
    Checks whether bounding boxes have already been computed
    :param directory:
    :return:
    '''

    boxes_path = os.path.join(directory, "bounding_boxes.pkl")
    boxes_validity_path = os.path.join(directory, "bounding_box_validity.pkl")
    detection_already_performed = os.path.isfile(boxes_path) and os.path.isfile(boxes_validity_path)

    if not detection_already_performed:
        boxes_path = os.path.join(directory, "boxes_failure.txt")
        detection_already_performed = os.path.isfile(boxes_path)

    return detection_already_performed


def add_missing_values(detection_results: List[torch.Tensor], validity_results: List[torch.Tensor], max_gap=4):
    '''
    Fills in place the missing detection results with interpolated ones

    :param detection_results: List of lists with detection results (left, top, right, bottom) for each object
    :param validity_results: List of lists with a boolean value for each object
    :param max_gap: maximum length of the missing detection gap that the procedure will fix
    :return: False if the sequence could not be fixed due to missing values
    '''

    results_count = len(detection_results)
    # Counts the number of objects
    objects_count = len(detection_results[0][0])

    for object_idx in range(objects_count):

        # Finds the first and last elements
        first_result_idx = -1
        last_result_idx = -1
        for idx, current_validities in enumerate(validity_results):
            current_object_validity = current_validities[object_idx]
            if first_result_idx == -1 and current_object_validity == True:
                first_result_idx = idx
            if current_object_validity == True:
                last_result_idx = idx

        if first_result_idx == -1:
            return False

        # Fix every gap between successful detections by interpolation
        void_begin_index = -1
        invalid_element_seen = False
        for idx in range(first_result_idx, last_result_idx + 1):

            current_object_validity = validity_results[idx][object_idx]

            # Found end of a gap with a None
            if current_object_validity == True and invalid_element_seen:
                # Fixes the gap if it is not too big
                if idx - void_begin_index - 1 < max_gap:
                    fix_sequence(void_begin_index, idx, detection_results, validity_results, object_idx)

            # Records that a none was seen
            if current_object_validity == False:
                invalid_element_seen = True
            # Found potential beginning of a gap
            if current_object_validity == True:
                void_begin_index = idx
                invalid_element_seen = False

    return True

def fix_sequence(begin_idx: int, end_idx: int, detection_results: List[torch.Tensor], validity_results: List[torch.Tensor], object_idx):
    '''
    Fixes in place the gap between two specified indexes in the results for the specified object

    :param begin_idx:
    :param end_idx:
    :param detection_results:
    :param validity_results:
    :param object: id of the object for which to fix the gap
    :return:
    '''

    begin_values = detection_results[begin_idx]
    end_values = detection_results[end_idx]
    steps = end_idx - begin_idx

    # For each position in the gap
    for current_step in range(1, steps):

        # Computes and add the linear interpolation delta
        delta = (end_values[:, object_idx] - begin_values[:, object_idx]) / steps
        current_results = begin_values[:, object_idx] + delta * current_step

        detection_results[begin_idx + current_step][:, object_idx] = current_results
        validity_results[begin_idx + current_step][object_idx] = True


def detect_image(detector: TennisPlayerDetector, image: Image, w2c_transformation: torch.tensor, focal_length: float):
    '''
    Computes player bounding boxes for a given image

    :param detector: tennis player detector
    :param image: image where to detect players
    :param w2c_transformation: (4, 4) transformation matrix from world to camera coordinates
    :param focal_length: focal length of the camera
    :return: (4, 2) tensor with normalized bounding box coordinates (left, top, bottom, right) in [0, 1]
             (2) boolean tensor with True if that bounding box is valid or not
    '''

    width, height = image.size

    camera_points = RayHelper.transform_points(world_points, w2c_transformation.unsqueeze(0))
    # Minus compensates for the negative 0
    camera_points = -camera_points / camera_points[..., 2:3] * focal_length
    # Compensates for the y axis that must go from top to bottom
    camera_points[..., 1] *= -1
    # Moves the origin of the image coordinates on the top left
    camera_points[..., 1] += height / 2
    camera_points[..., 0] += width / 2
    camera_points = camera_points[..., :2]
    camera_points = camera_points.cuda()

    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    transforms = torchvision.transforms.Compose(transforms)

    # Transforms the image
    observation = transforms(image).cuda()

    bounding_boxes, is_valid = detector(observation, camera_points)

    # Normalizes the bounding boxes
    bounding_boxes = torch.clone(bounding_boxes)
    bounding_boxes[0] /= width
    bounding_boxes[1] /= height
    bounding_boxes[2] /= width
    bounding_boxes[3] /= height

    assert(bounding_boxes.min().item() >= 0.0)
    assert(bounding_boxes.max().item() <= 1.0)
    bounding_boxes = torch.clamp(bounding_boxes, min=0.0, max=1.0)

    return bounding_boxes, is_valid


def detect_folder(detector: TennisPlayerDetector, path: str, image_extension="png"):
    '''
    Extracts camera intrinsics and extrinsics for a given folder

    :param detector: tennis player detector
    :param path: path of the folder to process
    :return:
    '''

    cameras_filename = "cameras.pkl"
    focals_filename = "focals.pkl"

    print(f"- Detecting players in '{path}'")

    # Finds all frames
    frame_filenames = list(sorted(glob.glob(os.path.join(path, f"*.{image_extension}"))))
    frame_filenames = [filename for filename in frame_filenames if re.search(fr'[0-9][0-9][0-9][0-9][0-9].{image_extension}$', filename)]

    all_bounding_boxes = []
    all_is_valid = []

    # Reads cameras and focals
    with open(os.path.join(path, cameras_filename), 'rb') as f:
        cameras = pickle.load(f)
    with open(os.path.join(path, focals_filename), 'rb') as f:
        focals = pickle.load(f)

    # Computes the detections for each frame
    for idx, current_frame_filename in enumerate(frame_filenames):
        print(f"Detecting frame {current_frame_filename}")

        # Gets camera parameters
        current_camera = cameras[idx]
        current_focal = focals[idx]
        current_w2c_transformation = current_camera.to_torch().get_inverse_homogeneous_matrix()

        current_image = Image.open(current_frame_filename)

        current_bounding_boxes, current_is_valid = detect_image(detector, current_image, current_w2c_transformation, current_focal)
        all_bounding_boxes.append(current_bounding_boxes.cpu())
        all_is_valid.append(current_is_valid.cpu())

    # Adds valies for missing detections. Also checks whether the sequence is good or not
    detection_success = add_missing_values(all_bounding_boxes, all_is_valid)

    if detection_success:
        # Draws the bounding boxes for each frame with certain probability
        if random.random() < 0.1:
            for idx, current_frame_filename in enumerate(frame_filenames):

                bounding_box_image = Image.open(current_frame_filename)

                # Draws the bounding boxes on the image
                BoundingBoxDrawer.draw_bounding_box(bounding_box_image, all_bounding_boxes[idx][:, 0].numpy())
                BoundingBoxDrawer.draw_bounding_box(bounding_box_image, all_bounding_boxes[idx][:, 1].numpy())
                # Saves the bounding boxes
                current_output_path = current_frame_filename + ".bounding_boxes.png"
                bounding_box_image.save(current_output_path)
                bounding_box_image.save(current_output_path)

        bounding_boxes_filename = "bounding_boxes.pkl"
        bounding_boxes_validity_filename = "bounding_box_validity.pkl"

        # Converts to numpy
        all_bounding_boxes = [current_bounding_box.cpu().numpy() for current_bounding_box in all_bounding_boxes]
        all_is_valid = [current_is_valid.cpu().numpy() for current_is_valid in all_is_valid]

        # Saves bounding boxes and bounding box validity
        with open(os.path.join(path, bounding_boxes_filename), 'wb') as f:
            pickle.dump(all_bounding_boxes, f)
        with open(os.path.join(path, bounding_boxes_validity_filename), 'wb') as f:
            pickle.dump(all_is_valid, f)
    # Creates a file signalling the failure
    else:
        failure_filename = "boxes_failure.txt"
        failure_path = os.path.join(path, os.path.join(path, failure_filename))
        open(failure_path, 'a').close()

    return None


def process_video(args):
    '''
    Creates bounding box annotations for the given video, outputting them in the output directory.
    Camera calibration results must be already present

    :param detector: tennis player detector
    :param video_path: the original video to process
    :param output_directory: directory where to output annotations
    :param directory_index: index of the current video
    :param output_framerate: framerate for the output video
    :param output_size: (width, height) of the output video
    :param image_extension:
    :param resume: if True tries to resume the process from where it left
    :return:
    '''

    detector, video_path, output_directory, directory_index, output_framerate, output_size, image_extension, resume = args

    # Cleans possible leftover frames
    VideoFragmenter.clean_frames(output_directory, image_extension)

    # If calibration failed, do not proceed
    print(f"-- [{datetime.now()}] Computing video {video_path}")
    if not calibration_successful(output_directory):
        print(f"-- [{datetime.now()}] Calibration was not successful, bounding boxes will not be computed for video {video_path}")
        return False

    # If bounding boxes are already there, do not proceed
    if detection_already_performed(output_directory):
        print(f"-- [{datetime.now()}] Bounding boxes already present for video {video_path}")
        return True

    print(f"-- [{datetime.now()}] Extracting frames of {video_path}")
    # Extracts frames from the video and creates the output directory
    frames = VideoFragmenter.extract_frames(video_path, output_directory, output_framerate, output_size, image_extension)

    print(f"-- [{datetime.now()}] Extracting bounding boxes of {video_path}")
    detect_folder(detector, output_directory, image_extension)

    print(f"-- [{datetime.now()}] Completed bounding box extraction of {video_path}")
    # Cleans all the frames
    VideoFragmenter.clean_frames(output_directory, image_extension)

    return True


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_directory", type=str, required=True, help="Directory with splitted youtube videos")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory where to output the dataset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--image_extension", type=str, default="png")
    parser.add_argument("--framerate", type=int, default=5, help="Dataset output framerate")
    parser.add_argument("--height", type=int, default=560, help="Dataset output height")
    parser.add_argument("--width", type=int, default=996, help="Dataset output width")

    arguments = parser.parse_args()

    video_directory = arguments.video_directory
    output_directory = arguments.output_directory
    resume = arguments.resume
    testing = arguments.testing
    video_extension = arguments.video_extension
    image_extension = arguments.image_extension
    framerate = arguments.framerate
    height = arguments.height
    width = arguments.width

    if testing:
        print("Warning: testing mode is enable, is that intended?")

    output_size = (width, height)

    detector = TennisPlayerDetector()
    detector.threshold = 0.60

    # Gets all input videos
    input_videos = VideoFragmenter.get_videos(video_directory, video_extension)

    print("- Start extracting bounding boxes")
    # Creates work items for splitting the videos
    work_items = []
    for video_idx, current_video in enumerate(input_videos):
        current_output_directory = os.path.join(output_directory, f"{video_idx:05d}", "00000")

        if testing and not os.path.isdir(current_output_directory):
            continue

        process_video((detector, current_video, current_output_directory, video_idx, framerate, output_size, image_extension, resume))

    print("- All Done")


if __name__ == "__main__":
    '''
    Extracts the subsequences of the youtube videos that contain relevant content
    '''

    main()
