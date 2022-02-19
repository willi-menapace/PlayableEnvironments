import glob
import math
import os
import shutil
import subprocess

import json
from pathlib import Path
from typing import Tuple, List

import torch
import numpy as np

from dataset.video import Video
from utils.lib_3d.bounding_box import BoundingBox
from utils.lib_3d.pose_parameters import PoseParametersNumpy
from utils.lib_3d.ray_helper import RayHelper


class SplittedRecording:
    '''
    Class representing a minecraft recording with associated annotations and splits
    '''

    def __init__(self, video_file: str, annotation_file: str, split_annotation_file: str):
        '''

        :param video_file: path of the original video file
        :param annotation_file: path of the video annotation file
        :param split_annotation_file: path of the video splits file
        '''

        self.video_file = video_file
        self.annotation_file = annotation_file
        self.split_annotation_file = split_annotation_file

        # Loads the annotations
        with open(annotation_file) as f:
            self.annotations = json.load(f)

        # Loads the split annotations
        self.split_annotations, self.world_center = self.read_split_annotations(self.split_annotation_file)

    def get_splits_count(self) -> int:
        '''
        Gets the number of splits for the current video
        :return:
        '''
        return len(self.split_annotations)

    def read_split_annotations(self, filename: str) -> List[Tuple[int]]:
        '''
        Reads the world center_coordinates and time split annotations and returns them as a list of (start_time, end_time)
        :param filename: name of the file containing the annotations
        :return:
        '''

        first_line = True

        all_times = []
        world_center_coordinates = None
        with open(filename) as f:
            for currentline in f:
                currentline = currentline.strip()
                # Skip empty lines
                if len(currentline) == 0:
                    continue
                # Ignore comments
                if currentline[0] == "#":
                    continue
                # The first line contains the world center coordinates
                if first_line:
                    world_center_coordinates = currentline.split(" ")
                    world_center_coordinates = [int(current_coordinate.strip()) for current_coordinate in world_center_coordinates]
                    first_line = False
                else:
                    times = currentline.split(" ")
                    # Transforms to integers
                    times = [int(current_time.strip()) for current_time in times]
                    if len(times) != 2:
                        raise Exception("Split times are expected to be in the 'start_time end_time' format, but not exactly 2 values were found")
                    all_times.append(times)

        return all_times, world_center_coordinates

    def compute_object_bounding_boxes(self, bounding_box: BoundingBox, transformation_matrix_o2w: torch.Tensor, transformation_matrix_w2c: torch.Tensor,
                                      focal: float, height: int, width: int) -> Tuple[torch.Tensor, bool]:
        '''
        For each object, compute its bounding boxes in the image plane of each camera

        :param transformation_matrix_o2w: (4, 4) tensor with transformation matrix for each object from object to world coordinates
        :param transformation_matrix_w2c: (4, 4) tensor with transformation matrix for each object from world to each
                                                                     camera coordinates
        :param focal: focal length associated to each camera
        :param height: height in pixels of the image plane
        :param width: width in pixels of the image plane
        :return: (4) tensor with left, top, right, bottom coordinates for the bounding box of
                     in the image plane of each camera. Coordinates refer to
                     the top left corner of the image plane and are normalized in [0, 1]
                 boolean set to true if the bounding box is visible in the current frame
        '''

        current_object_points = bounding_box.get_corner_points()
        origin_point = torch.zeros_like(current_object_points[0:1])
        # Adds also the origin point to the bounding box to compute the position of the feet
        current_object_points = torch.cat([current_object_points, origin_point], dim=0)

        # Creates a dimension for the number of points in each bounding box
        current_transformation_matrix_o2w = transformation_matrix_o2w.unsqueeze(-3)
        # Transforms the points into world coordinates
        current_world_points = RayHelper.transform_points(current_object_points, current_transformation_matrix_o2w)
        # Creates a dimension for the number of cameras
        #current_world_points = current_world_points.unsqueeze(-3)
        # Creates a dimension for the number of points in each bounding box
        current_transformation_matrix_w2c = transformation_matrix_w2c.unsqueeze(-3)
        current_camera_points = RayHelper.transform_points(current_world_points, current_transformation_matrix_w2c)

        # Adds dimensions to account for the 8 and 3 dimensions of the bounding box points
        focal = torch.as_tensor(focal, dtype=torch.float, device=transformation_matrix_o2w.device)
        unsqueezed_focal = focal.unsqueeze(-1).unsqueeze(-1)

        # Projects the points on the image plane. - Accounts for the camera pointing in the -z dimension
        # (..., cameras_count, 8, 3)
        projected_points = -current_camera_points[..., :2] / current_camera_points[..., 2:3] * unsqueezed_focal
        projected_points[..., 1] *= -1  # y coordinates grow going down in the image plane due to matrix indexing

        # Computes bounding boxes positions
        left = torch.min(projected_points[..., 0], dim=-1)[0]
        right = torch.max(projected_points[..., 0], dim=-1)[0]
        top = torch.min(projected_points[..., 1], dim=-1)[0]
        # Changes the implementation of bottom from the lowest point to the point representing the position
        # of the feet in order to have a more precise localization
        #bottom = torch.max(projected_points[..., 1], dim=-1)[0]
        bottom = projected_points[-1, 1]

        current_bounding_box = torch.stack([left, top, right, bottom], dim=-1)

        # Transforms the coordinate reference point from the image plane center to the top left center and normalizes
        current_bounding_box[0] = (current_bounding_box[0] + (width / 2)) / width
        current_bounding_box[2] = (current_bounding_box[2] + (width / 2)) / width
        current_bounding_box[1] = (current_bounding_box[1] + (height / 2)) / height
        current_bounding_box[3] = (current_bounding_box[3] + (height / 2)) / height

        # Clamps values outsize the image plane
        current_bounding_box = torch.clamp(current_bounding_box, min=0.0, max=1.0)

        bounding_box_visible = True
        # If all points are behind the camera (positive z) the object is not visible
        if (current_camera_points[..., 2] > 0).all():
            bounding_box_visible = False
        # If one of these happens the box was outside of the frame, was clamped and collapsed
        if current_bounding_box[0].item() == current_bounding_box[2].item() or \
            current_bounding_box[1].item() == current_bounding_box[3].item():
            bounding_box_visible = False
        # Makes a fake bounding box if it is not visible so that we don't have to handle special cases for collapsed
        # bounding boxes
        if not bounding_box_visible:
            current_bounding_box[0] = 0.25
            current_bounding_box[1] = 0.25
            current_bounding_box[2] = 0.75
            current_bounding_box[3] = 0.75

        return current_bounding_box, bounding_box_visible

    def get_minecraft_man_bounding_box(self) -> BoundingBox:
        ''''
        Gets the bounding box for the minecraft men
        '''

        width = 0.15
        depth = 0.15
        height = 0.5
        dimensions = [
            (-width, width),  # x
            (0.0, height),    # y
            (-depth, depth),  # z
        ]

        return BoundingBox(dimensions)

    def get_directory_name_by_index(self, index: int):
        return f'{index:05d}'

    def get_frame_name_by_index(self, index: int):
        return f'{index:05d}'

    def output_video_frames(self, output_directory: str, subdirectory_begin_index: int, extension="png"):
        '''
        Transforms the video into frames and outputs them to a subdirectory of the specified directory
        The subdirectories to be created start with 'subdirectory_begin_index' and increase by one for each directory

        :param output_directory: directory where to output the frames
        :param subdirectory_begin_index: index to assign to the subdirectory for the first split
        :return:
        '''

        # Makes the output directory
        Path(output_directory).mkdir(exist_ok=True)

        # Produces each split
        current_index = subdirectory_begin_index
        for current_begin_seconds, current_end_seconds in self.split_annotations:

            # Creates the output directory for the current video
            current_output_directory = os.path.join(output_directory, self.get_directory_name_by_index(current_index))
            Path(current_output_directory).mkdir(exist_ok=True)
            current_output_pattern = os.path.join(current_output_directory, f'%05d.{extension}')

            # Deletes existing images to avoid possibly mixing the frames of two different videos
            existing_images = list(sorted(glob.glob(os.path.join(current_output_directory, f"*.{extension}"))))
            for current_image in existing_images:
                os.remove(current_image)

            split_length = current_end_seconds - current_begin_seconds
            # Command to split the video and render each frame
            command_parameters = ["ffmpeg", '-ss', f'{current_begin_seconds}', '-t', f'{split_length}', "-i", self.video_file, current_output_pattern]

            # Splits the video
            subprocess.run(command_parameters)

            current_index += 1

            # Shifts all image names from [1, frames_count] to [0, frames_count - 1]
            all_images = list(sorted(glob.glob(os.path.join(current_output_directory, f"*.{extension}"))))
            for idx, current_image in enumerate(all_images):
                new_image_name = self.get_frame_name_by_index(idx) + f".{extension}"
                new_image_name = os.path.join(current_output_directory, new_image_name)
                shutil.move(current_image, new_image_name)

    def get_framerate(self, filename: str) -> float:
        '''
        Returns the framerate in of the specified video
        :param filename: video file of which to compute the framerate
        :return:
        '''
        #ffmpeg - i filename 2 > & 1 | sed -n "s/.*, \(.*\) fp.*/\1/p"

        output = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-of", "default=noprint_wrappers=1:nokey=1", "-show_entries", "stream=r_frame_rate", filename]).decode('ascii')
        numerator, denominator = output.split("/")
        numerator = int(numerator)
        denominator = int(denominator)

        if numerator % denominator != 0:
            raise Exception(f"Expected fps to be an integral value but got '{numerator / denominator}'")

        framerate = numerator // denominator

        return framerate

    def output_annotations(self, output_directory: str, subdirectory_begin_index: int,):
        '''
        Outputs the annotations in the dataset format using output_directory as the root directory. Each split is assigned
        a subdirectory.
        The subdirectories to be created start with 'subdirectory_begin_index' and increase by one for each directory

        :param output_directory: directory where to output the annotations
        :param subdirectory_begin_index: index to assign to the subdirectory for the first split
        :return:
        '''

        # Makes the output directory
        Path(output_directory).mkdir(exist_ok=True)

        framerate = self.get_framerate(self.video_file)

        # Produces each split
        current_index = subdirectory_begin_index
        for current_begin_seconds, current_end_seconds in self.split_annotations:

            # Creates the output directory for the current video
            current_output_directory = os.path.join(output_directory, self.get_directory_name_by_index(current_index))
            Path(current_output_directory).mkdir(exist_ok=True)

            camera_intrinsics = self.annotations["camera_intrinsics"]
            current_annotations = self.annotations["timesteps"][current_begin_seconds * framerate:current_end_seconds * framerate]

            # Outputs the current annotations
            self.output_splitted_annotations(camera_intrinsics, current_annotations, current_output_directory)

            current_index += 1

    def compute_focal_length(self, image_height: int, y_fov: float):
        '''
        Computes the focal length of the camera in pixels
        :param image_height:
        :param y_fov:

        :return:
        '''

        focal_length = (image_height / 2) / math.tan(y_fov / 2)
        return focal_length

    def minecraft_rotations_to_dataset(self, rotation: float, modulo_threesixty: bool=False) -> float:
        '''
        Converts rotations from minecraft format to the dataset format

        :param rotation: The rotation to convert
        :param modulo_threesixty: Whether the output rotation must be in the range [0, 2 * pi)
        :return:
        '''
        dataset_rotation = (rotation * -1 + 360.0)
        dataset_rotation = dataset_rotation * math.pi / 180.0

        # Values outside the range (-360, +360) degrees should always be redimensioned
        while dataset_rotation <= -2 * math.pi:
            dataset_rotation += 2 * math.pi
        while dataset_rotation >= 2 * math.pi:
            dataset_rotation -= 2 * math.pi

        if modulo_threesixty:
            while dataset_rotation < 0:
                dataset_rotation += 2 * math.pi
            while dataset_rotation >= 2 * math.pi:
                dataset_rotation -= 2 * math.pi

        return dataset_rotation

    def annotation_to_pose_parameters(self, annotation) -> PoseParametersNumpy:
        '''
        Extract the pose parameters from the current annotation
        :param annotation: The current annotation from which to extract pose parameters
        :return:
        '''

        # Reads the rotations and translates them to minecraft coordinates
        rotations = [self.minecraft_rotations_to_dataset(annotation["rotX"]),
                     self.minecraft_rotations_to_dataset(annotation["rotY"]),
                     self.minecraft_rotations_to_dataset(annotation["rotZ"], modulo_threesixty=True)]  # Normalizes rotation on Z to [0, 2 * pi)
        # Reads translations and transforms them to the world coordinate system specified by the user
        translations = [annotation["posX"] - self.world_center[0],
                        annotation["posY"] - self.world_center[1],
                        annotation["posZ"] - self.world_center[2]]

        return PoseParametersNumpy(rotation=rotations, translation=translations)

    def output_splitted_annotations(self, camera_intrinsics, annotations: List, output_directory: str):
        '''
        Outputs the annotations in the dataset format for the current split

        :param camera_intrinsics: the camera intrinsics for the scene
        :param annotations: list of annotations for each frame
        :param output_directory: directory where to output the annotations
        :return:
        '''

        # Gets camera intrinsics
        image_height = camera_intrinsics["image_height"]
        image_width = camera_intrinsics["image_width"]
        y_fov = camera_intrinsics["y_fov"]
        z_near = camera_intrinsics["z_near"]
        z_far = camera_intrinsics["z_far"]
        focal = self.compute_focal_length(image_height, y_fov)

        frames_count = len(annotations)

        # Structures for storing pose and bounding boxes annotations
        all_camera_annotations = []
        all_object_annotations = []
        all_bounding_boxes = []
        all_bounding_boxes_validity = []

        # Gets the bounding box for the minecraft man
        bounding_box_3d = self.get_minecraft_man_bounding_box()

        # Gets the annotation for each step
        for current_annotation in annotations:
            current_camera_annotation = current_annotation["camera"]

            # Extracts camera pose parameters
            current_camera_pose_parameters = self.annotation_to_pose_parameters(current_camera_annotation)
            all_camera_annotations.append(current_camera_pose_parameters)
            # Computes transformation matrices
            transformation_matrix_c2w = current_camera_pose_parameters.to_torch().as_homogeneous_matrix_torch()
            transformation_matrix_w2c = transformation_matrix_c2w.inverse()

            # Object annotations sorted by uuid
            current_all_object_annotation = current_annotation["entities"]["objects"]
            current_all_object_annotation.sort(key=lambda x: x["uuid"])

            # Extract object pose parameters
            current_all_object_pose_parameters = []
            current_all_object_bounding_boxes = []
            current_all_object_boudning_boxes_validity = []
            for current_object_annotation in current_all_object_annotation:
                current_object_pose_parameters = self.annotation_to_pose_parameters(current_object_annotation)
                current_all_object_pose_parameters.append(current_object_pose_parameters)

                transformation_matrix_o2w = current_object_pose_parameters.to_torch().as_homogeneous_matrix_torch()

                # Extracts 2d bounding boxes and their validity
                current_bounding_box, current_bounding_box_valid = self.compute_object_bounding_boxes(bounding_box_3d, transformation_matrix_o2w,
                                                                                                      transformation_matrix_w2c, focal, image_height, image_width)
                current_bounding_box = current_bounding_box.cpu().numpy()
                current_all_object_bounding_boxes.append(current_bounding_box)
                current_all_object_boudning_boxes_validity.append(current_bounding_box_valid)

            all_object_annotations.append(current_all_object_pose_parameters)
            # Stacks all object bounding boxes and their validity in a single tensor
            current_all_object_bounding_boxes = np.stack(current_all_object_bounding_boxes, axis=-1)
            current_all_object_boudning_boxes_validity = np.asarray(current_all_object_boudning_boxes_validity)
            all_bounding_boxes.append(current_all_object_bounding_boxes)
            all_bounding_boxes_validity.append(current_all_object_boudning_boxes_validity)

        focals = [focal] * frames_count
        actions = [0] * frames_count  # No actions
        rewards = [0] * frames_count  # No rewards
        dones = [False] * frames_count  # Never done
        # Dumps all the annotations as metadata
        metadata = [current_annotation for current_annotation in annotations]

        # Creates and saves the video object
        current_video = Video()
        current_video.add_content(output_directory, actions, rewards, metadata, dones, all_camera_annotations, focals, all_bounding_boxes, all_bounding_boxes_validity, object_poses=all_object_annotations)
        current_video.save(output_directory, exists_ok=True)


def main():

    begin_index = 3
    output_directory = "/home/willi/.local/share/multimc/instances/1.16.5/.minecraft/replay_videos/"

    video_file = "/home/willi/.local/share/multimc/instances/1.16.5/.minecraft/replay_videos/2021_06_08_13_40_05.mp4"
    annotations_file = "/home/willi/.local/share/multimc/instances/1.16.5/.minecraft/replay_videos/2021_06_08_13_40_05.json"
    splits_file = "/home/willi/.local/share/multimc/instances/1.16.5/.minecraft/replay_videos/2021_06_08_13_40_05_splits.txt"

    recorded_video = SplittedRecording(video_file, annotations_file, splits_file)
    recorded_video.output_video_frames(output_directory, begin_index)
    recorded_video.output_annotations(output_directory, begin_index)
    pass


if __name__ == "__main__":
    main()



























