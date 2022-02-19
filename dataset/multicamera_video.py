from __future__ import annotations

import os
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

from dataset.video import Video
from utils.lib_3d.pose_parameters import PoseParametersNumpy


class MulticameraVideo:
    '''
     A multivamera video from the dataset
     Metadata are always kept into memory, while frames are loaded from disk on demand only
    '''

    def __init__(self):
        self.videos = None

    def add_content(self, videos: List[Video]):
        '''
        Adds the contents to the video
        :param videos: list of videos taken by each camera. Each must have the same length
        :return:
        '''

        if len(videos) == 0:
            raise Exception("At least one video must be added to MulticameraVideo, but an empty video list was received")

        video_length = None
        for current_video in videos:
            current_length = current_video.get_frames_count()
            if video_length is None:
                video_length = current_length
            if video_length != current_length:
                raise Exception(f"Videos with differing lengths found: ({video_length}) and ({current_length})")

        self.videos = videos

    def _index_to_filename(self, idx):
        return f'{idx:05}'

    def load(self, path):

        if not os.path.isdir(path):
            raise Exception(f"Cannot load multicamera video: '{path}' is not a directory")

        all_videos = []
        directory_contents = sorted(list(os.listdir(path)))

        # Loads each directory found in the specified path
        for current_file in directory_contents:
            current_file_path = os.path.join(path, current_file)
            if os.path.isdir(current_file_path):
                current_video = Video()
                current_video.load(current_file_path)
                all_videos.append(current_video)

        self.videos = all_videos

    def is_initialized(self):
        return self.videos is not None and len(self.videos) > 0

    def get_available_cameras_count(self) -> int:
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        return len(self.videos)

    def has_flow(self) -> bool:
        '''
        Check whether optical flow data is available
        :return: True if optical flow data is present
        '''

        return self.videos[0].has_flow()

    def has_keypoints(self) -> bool:
        '''
        Check whether keypoints data is available
        :return: True if keypoints data is present
        '''

        return self.videos[0].has_keypoints()

    def has_object_poses(self) -> bool:
        '''
        Check whether object poses are available
        :return: True if object poses are available
        '''

        return self.videos[0].has_object_poses()

    def has_crop_regions(self) -> bool:
        '''
        Check whether crop regions are available
        :return: True if crop regions are available
        '''

        return self.videos[0].has_crop_region()

    def get_frames_count(self) -> int:
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        return self.videos[0].get_frames_count()

    def get_flow_at(self, frame_idx: int, cameras_idx: List[int] = None) -> List[np.ndarray]:
        '''
        Returns the optical flow corresponding to the specified frame index for each of the specified cameras

        :param frame_idx: index of the optical flow to retrieve in [0, frames_count-1]
        :param cameras_idx: List of camera indexes in [0, cameras_count-1] specifying the frames to retrieve.
                            None to retrieve frames from all available cameras
        :return: The normalized optical flow in the range [0, 1] at the specified index from the specified cameras
                 (2, height, width) the two channels dimensions are in y, x order instead of the usual x, y order for optical flow
                 Flows are returned in the order of specified by the camera indexes
        '''

        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        if cameras_idx is None:
            flows = [current_video.get_flow_at(frame_idx) for current_video in self.videos]
        else:
            flows = []
            for current_id in cameras_idx:
                flows.append(self.videos[current_id].get_flow_at(frame_idx))

        return flows

    def get_frames_at(self, frame_idx: int, cameras_idx: List[int] = None) -> List[Image]:
        '''
        Returns the frame corresponding to the specified frame index for each of the specified camera

        :param frame_idx: index of the frame to retrieve in [0, frames_count-1]
        :param cameras_idx: List of camera indexes in [0, cameras_count-1] specifying the frames to retrieve.
                            None to retrieve frames from all available cameras
        :return: The frame at the specified index from the specified cameras. Frames are returned in the order of
                 specified by the camera indexes
        '''
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        if cameras_idx is None:
            images = [current_video.get_frame_at(frame_idx) for current_video in self.videos]
        else:
            images = []
            for current_id in cameras_idx:
                images.append(self.videos[current_id].get_frame_at(frame_idx))

        return images

    def get_summed_rewards_at(self, start_index: int, end_index: int, camera_idx:int = 0) -> Image:
        '''
        Returns the summed rewards in the range [start_index, end_index] included, for the specified camera

        :param start_index: starting index for the sum
        :param end_index: end index (included) for the sum
        :param cameras_idx: camera index for which to retrieve the rewards
        :return: The summed rewards from the specified camera
        '''
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        current_video = self.videos[camera_idx]
        summed_rewards = sum(current_video.rewards[start_index:end_index + 1])

        return summed_rewards

    def get_multicamera_attribute_elements_at(self, attribute_name: str, attribute_idx: int, cameras_idx: List[int] = None):
        '''
        Gets the attribute elements at position idx for each of the specified cameras
        :param attribute_name: name of the Video attribute to retrieve
        :param attribute_idx: index of the attribute position to retrieve in [0, frames_count-1]
        :param cameras_idx: List of camera indexes in [0, cameras_count-1] specifying the frames to retrieve.
                            None to retrieve frames from all available cameras
        :return: The attribute at the specified index from the specified cameras. Attributes are returned in the order of
                 specified by the camera indexes
        '''

        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        if cameras_idx is None:
            attributes = [getattr(current_video, attribute_name)[attribute_idx] for current_video in self.videos]
        else:
            attributes = []
            for current_id in cameras_idx:
                current_video = self.videos[current_id]
                attributes.append(getattr(current_video, attribute_name)[attribute_idx])

        return attributes

    def get_camera_attribute_elements_at(self, attribute_name: str, attribute_idx: int, camera_idx: int = 0):
        '''
        Gets the attribute elements at position idx for the specified cameras
        :param attribute_name: name of the Video attribute to retrieve
        :param attribute_idx: index of the attribute position to retrieve in [0, frames_count-1]
        :param camera_idx: index of the camera of which to retrieve the attribute
        :return: The attribute at the specified index from the specified camera.
        '''
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        attributes = getattr(self.videos[camera_idx], attribute_name)[attribute_idx]

        return attributes

    def get_actions_at(self, idx: int, camera_idx: int = 0) -> List[int]:
        return self.get_camera_attribute_elements_at("actions", idx, camera_idx)

    def get_rewards_at(self, idx: int, camera_idx: int = 0) -> List[float]:
        return self.get_camera_attribute_elements_at("rewards", idx, camera_idx)

    def get_metadata_at(self, idx: int, cameras_idx: List[int] = None) -> List[Dict]:
        return self.get_multicamera_attribute_elements_at("metadata", idx, cameras_idx)

    def get_dones_at(self, idx: int, camera_idx: int = 0) -> List[bool]:
        return self.get_camera_attribute_elements_at("dones", idx, camera_idx)

    def get_cameras_at(self, idx: int, cameras_idx: List[int] = None) -> List[PoseParametersNumpy]:
        return self.get_multicamera_attribute_elements_at("cameras", idx, cameras_idx)

    def get_focals_at(self, idx: int, cameras_idx: List[int] = None) -> List[float]:
        return self.get_multicamera_attribute_elements_at("focals", idx, cameras_idx)

    def get_bounding_boxes_at(self, idx: int, cameras_idx: List[int] = None) -> List[np.ndarray]:
        return self.get_multicamera_attribute_elements_at("bounding_boxes", idx, cameras_idx)

    def get_bounding_boxes_validity_at(self, idx: int, cameras_idx: List[int] = None) -> List[np.ndarray]:
        return self.get_multicamera_attribute_elements_at("bounding_boxes_validity", idx, cameras_idx)

    def get_keypoints_at(self, idx: int, cameras_idx: List[int] = None) -> List[np.ndarray]:
        return self.get_multicamera_attribute_elements_at("keypoints", idx, cameras_idx)

    def get_keypoints_validity_at(self, idx: int, cameras_idx: List[int] = None) -> List[np.ndarray]:
        return self.get_multicamera_attribute_elements_at("keypoints_validity", idx, cameras_idx)

    def get_object_poses_at(self, idx: int, cameras_idx: List[int] = None) -> List[List[PoseParametersNumpy]]:
        return self.get_multicamera_attribute_elements_at("object_poses", idx, cameras_idx)

    def get_crop_region_at(self, idx: int, cameras_idx: List[int] = None) -> List[List[float]]:
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        if cameras_idx is None:
            all_crops = [current_video.crop_region for current_video in self.videos]
        else:
            all_crops = []
            for current_id in cameras_idx:
                current_video = self.videos[current_id]
                all_crops.append(current_video.crop_region)

        return all_crops

    def get_frame_paths_at(self, idx: int, cameras_idx: List[int] = None):
        '''
        Gets the path of the frames at position idx for each of the specified cameras
        :param idx: index of the attribute position to retrieve in [0, frames_count-1]
        :param cameras_idx: List of camera indexes in [0, cameras_count-1] specifying the frames to retrieve.
                            None to retrieve frames from all available cameras
        :return: The path of the frame at the specified index from the specified cameras. Attributes are returned in the order of
                 specified by the camera indexes
        '''

        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call load()?")

        if cameras_idx is None:
            attributes = [current_video.get_frame_path_at(idx) for current_video in self.videos]
        else:
            attributes = []
            for current_id in cameras_idx:
                current_video = self.videos[current_id]
                attributes.append(current_video.get_frame_path_at(idx))

        return attributes

    def subsample_split_resize(self, frame_skip: int, output_sequence_length: int, crop_size: Tuple[int]=None, target_size: Tuple[int]=None, min_sequence_length=None) -> List[MulticameraVideo]:
        '''
        Splits the current multicameras sequence into a number of multicamera sequences of the specified length, skipping the specified number
        of frames in the source sequence between successive frames in the target sequence
        Resizes the output sequence to the target_size

        :param frame_skip: frames to skip in the source sequence between successive frames in the target sequence
        :param output_sequence_length: number of frames in each output sequence. -1 if length should not be modified
        :param crop_size: (left_index, upper_index, right_index, lower_index) size of the crop to take before resizing
        :param target_size: (width, height) size of the frames in the output sequence
        :param min_sequence_length: Minimum length of the sequences to retain. If None, sequences shorter than output_sequence_length are discarded
        :return: List of videos representing the split and subsampled source video
        '''

        # Splits the video from each camera
        splitted_videos = []
        for current_video in self.videos:
            current_splitted_video = current_video.subsample_split_resize(frame_skip, output_sequence_length, crop_size, target_size, min_sequence_length)
            splitted_videos.append(current_splitted_video)

        # Creates a multicamera video from each split
        splits_count = len(splitted_videos[0])
        multicamera_videos = []
        for split_idx in range(splits_count):
            current_videos = [current_splitted_video[split_idx] for current_splitted_video in splitted_videos]
            current_multicamera_video = MulticameraVideo()
            current_multicamera_video.add_content(current_videos)
            multicamera_videos.append(current_multicamera_video)

        return multicamera_videos

    def save(self, path: str, extension="png"):
        if not self.is_initialized():
            raise Exception("MulticameraVideo has not been initialized. Did you forget to call add_content()?")
        if os.path.isdir(path):
            raise Exception(f"A directory at '{path}' already exists")

        # Creates the directory
        os.mkdir(path)

        # Save all videos
        for idx, video in enumerate(self.videos):
            padded_index = self._index_to_filename(idx)
            video_path = os.path.join(path, f'{padded_index}')
            video.save(video_path, extension)

