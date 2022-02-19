import os
import sys
import time
from typing import List, Dict, Callable

from torch.utils.data import Dataset, DataLoader

from dataset.batching import BatchElement, single_batch_elements_collate_fn
from dataset.multicamera_video import MulticameraVideo


class MulticameraVideoDataset(Dataset):
    '''
    Dataset of multicamera video objects
    Expects a directory where each children directory represents a MulticameraVideo object on disk
    '''

    def __init__(self, path, batching_config: Dict, frame_transform: Callable, optical_flow_transform: Callable=None):
        '''
        Initializes the dataset with the videos in a directory

        :param path: path to the root of the dataset
        :param batching_config: Dict with the batching parameters to use for sampling
        :param frame_transform: transformation to apply to each frame
        :param optical_flow_transform: transformation to apply to each optical flow. Must be not None if optical flow
                                       is present
        '''

        if not os.path.isdir(path):
            raise Exception(f"Dataset directory '{path}' is not a directory")

        self.batching_config = batching_config
        self.allowed_cameras = batching_config['allowed_cameras']  # set of camera indexes allowed to be part of the dataset.
                                                                   # if not None only cameras with ids in this set are included in the dataset

        # number of frames that compose each observation
        self.observations_stacking = batching_config['observation_stacking']
        # how many frames to skip between each observation
        self.skip_frames = batching_config['skip_frames']
        self.frame_transform = frame_transform
        self.optical_flow_transforms = optical_flow_transform

        # Reads the videos in the root
        self.all_videos = self.read_all_videos(path)

        self.observations_count = None
        # number of observations to include in each dataset sample
        self.set_observations_count(batching_config['observations_count'])

        self.past_frames_by_video = self.compute_past_frames_by_video()

        if self.has_flow():
            if self.skip_frames > 0:
                print("Warning: optical flow is present in the dataset, but frames are skipped. Optical flow values are accurate only when they refer to consecutive frames", file=sys.stderr)
            if optical_flow_transform is None:
                raise Exception("Optical Flow transformation must be specified when optical flow is present")

    def set_observations_count(self, observations_count: int):
        '''
        Changes the number of observations in each future returned sequence

        :param observations_count: Number of observations in each future returned sequence
        :return:
        '''

        # Perform changes only if the parameter differs
        if self.observations_count is None or self.observations_count != observations_count:
            self.observations_count = observations_count

            self.available_samples_list = self.compute_available_samples_per_video()
            self.total_available_samples = sum(self.available_samples_list)

    def read_all_videos(self, path: str) -> List[MulticameraVideo]:
        '''
        Reads all the videos in the specified path

        :param path: path where videos are stored
        :return:
        '''

        all_videos = []
        contents = sorted(list(os.listdir(path)))

        for current_file in contents:
            current_file_path = os.path.join(path, current_file)
            print(f"- Loading video at '{current_file_path}'")
            if os.path.isdir(current_file_path):
                current_video = MulticameraVideo()
                current_video.load(current_file_path)
                all_videos.append(current_video)

        return all_videos

    def has_flow(self) -> bool:
        '''
        Check whether optical flow data is available
        :return: True if optical flow data is present
        '''

        return self.all_videos[0].has_flow()

    def has_keypoints(self) -> bool:
        '''
        Check whether keypoints data is available
        :return: True if keypoints data is present
        '''

        return self.all_videos[0].has_keypoints()

    def has_object_poses(self) -> bool:
        '''
        Check whether object poses are available
        :return: True if object poses are available
        '''

        return self.all_videos[0].has_object_poses()

    def has_crop_regions(self) -> bool:
        '''
        Check whether crop regions are available
        :return: True if crop regions are available
        '''

        return self.all_videos[0].has_crop_regions()

    def compute_past_frames_by_video(self) -> List[int]:
        '''
        Computes how many frames are there in the videos preceding each video

        :return: list with an integer for each video representing how many frames the videos before it contain
        '''

        past_frames = [0] # No frames precede the first video
        current_frames_sum = 0
        for current_video in self.all_videos[:-1]:
            current_frames_sum += current_video.get_frames_count()
            past_frames.append(current_frames_sum)

        return past_frames

    def compute_available_samples_per_video(self) -> List[int]:
        '''
        Computes how many samples can be drawn from the video sequences

        :return: list with an integer for each video representing how many samples can be drawn
        '''

        available_samples = []

        # Number of frames in the original video each sample will span
        sample_block_size = self.observations_count + (self.observations_count - 1) * self.skip_frames

        for current_video in self.all_videos:
            frames_count = current_video.get_frames_count()
            current_samples = frames_count - sample_block_size + 1
            current_samples = max(0, current_samples) # Avoids that short videos cause negative samples counts
            available_samples.append(current_samples)


        return available_samples

    def __len__(self):
        return self.total_available_samples

    def __getitem__(self, index) -> BatchElement:

        if index >= self.total_available_samples:
            raise Exception(f"Requested sample at index {index} is out of range")

        video_index = 0
        video_initial_frame_idx = 0

        # Searches the video and the frame index in that video where to start extracting the sequence
        passed_samples = 0
        for search_index, current_available_samples in enumerate(self.available_samples_list):
            if passed_samples + current_available_samples > index:
                video_index = search_index
                video_initial_frame_idx = index - passed_samples
                break
            passed_samples += current_available_samples

        current_video = self.all_videos[video_index]
        past_frames_in_dataset = self.past_frames_by_video[video_index]
        observation_indexes = []
        global_frame_indexes = []
        for i in range(self.observations_count):
            observation_indexes.append(video_initial_frame_idx + i * (self.skip_frames + 1))
            global_frame_indexes.append(past_frames_in_dataset + observation_indexes[-1])

        first_allowed_camera_index = 0
        if self.allowed_cameras is not None:
            first_allowed_camera_index = self.allowed_cameras[0]
        min_frame = video_initial_frame_idx % (self.skip_frames + 1) # The minimum frame for which the preceding would not be part of the video
        all_frames_indexes = [[max(current_observation_index - i * (self.skip_frames + 1), min_frame) for i in range(self.observations_stacking)] for current_observation_index in observation_indexes]

        all_frames = [[current_video.get_frames_at(index, self.allowed_cameras) for index in current_observation_stack] for current_observation_stack in all_frames_indexes]  # (observations_count, observation_stacking, cameras_count) images

        # Action is the one selected in the frame where the observation took place
        all_actions = [current_video.get_actions_at(current_index, first_allowed_camera_index) for current_index in observation_indexes]  # (observations_count)

        # The reward is the reward obtained for arriving in the current observation frame from the previous summing also the reward from the frames that were skipped
        all_rewards = [current_video.get_summed_rewards_at(max(current_index - self.skip_frames, 0), current_index, first_allowed_camera_index) for current_index in observation_indexes]  # (observations_count)
        all_metadata = [current_video.get_metadata_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)
        all_dones = [current_video.get_dones_at(current_index, first_allowed_camera_index) for current_index in observation_indexes]  # (observations_count)
        all_cameras = [current_video.get_cameras_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)
        all_focals = [current_video.get_focals_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)
        all_bounding_boxes = [current_video.get_bounding_boxes_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)
        all_bounding_boxes_validity = [current_video.get_bounding_boxes_validity_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)
        all_frame_paths = [current_video.get_frame_paths_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)

        all_keypoints = None
        all_keypoints_validity = None
        if self.has_keypoints():
            all_keypoints = [current_video.get_keypoints_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)
            all_keypoints_validity = [current_video.get_keypoints_validity_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)

        # Computes the optical flows if present
        all_flows = None
        if self.has_flow():
            all_flows = [current_video.get_flow_at(current_observation_stack[0], self.allowed_cameras) for current_observation_stack in all_frames_indexes]  # (observations_count, observation_stacking, cameras_count) images

        all_object_poses = None
        if self.has_object_poses():
            # Since object poses do not depend on the camera, gets the information only from the first camera
            all_object_poses = [current_video.get_object_poses_at(current_index, self.allowed_cameras)[0] for current_index in observation_indexes]  # (observations_count, objects_count)

        all_crop_regions = None
        if self.has_crop_regions():
            all_crop_regions = [current_video.get_crop_region_at(current_index, self.allowed_cameras) for current_index in observation_indexes]  # (observations_count, cameras_count)

        plain_batch_element = BatchElement(all_frames, all_actions, all_rewards, all_metadata, all_dones, all_cameras, all_focals,
                                           all_bounding_boxes, all_bounding_boxes_validity, all_frame_paths, global_frame_indexes, observation_indexes,
                                           video_index, current_video, self.frame_transform, optical_flows=all_flows,
                                           optical_flow_transforms=self.optical_flow_transforms, keypoints=all_keypoints,
                                           keypoints_validity=all_keypoints_validity, object_poses=all_object_poses,
                                           crop_regions=all_crop_regions)

        return plain_batch_element





