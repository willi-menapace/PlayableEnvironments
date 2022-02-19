from typing import List, Tuple, Dict, Callable

import numpy as np
import torch
from PIL import Image

from dataset.multicamera_video import MulticameraVideo
from utils.lib_3d.pose_parameters import PoseParametersNumpy


class BatchElement:

    def __init__(self, observations: List[Tuple[Image.Image]], actions: List, rewards: List[int], metadata: List[List[Dict]], dones: List[bool],
                 cameras: List[PoseParametersNumpy], focals: List[float], bounding_boxes: List[np.ndarray], bounding_boxes_validity: List[np.ndarray],
                 observations_paths: List[List[str]], global_frame_indexes: List[int], video_frame_indexes: List[int], video_index: int, video: MulticameraVideo, transforms,
                 optical_flows: List[Tuple[np.ndarray]]=None, optical_flow_transforms: Callable=None, keypoints: List[np.ndarray]=None,
                 keypoints_validity: List[np.ndarray]=None, object_poses: List[List[PoseParametersNumpy]]=None, crop_regions: List[List[List[float]]]=None):
        '''
        Constructs a batch element

        :param observations: list of size (observations_count, observation_stacking, cameras_count) frames each from the most recent to the oldest
        :param actions: list of size (observations_count) actions
        :param rewards: list of size (observations_count) rewards
        :param metadata: list of size (observations_count, cameras_count) metadata
        :param dones: list of size (observations_count) booleans representing whether the episode has ended
        :param cameras: list of size (observations_count, cameras_count) camera poses
        :param focals: list of size (observations_count, cameras_count) camera focals
        :param bounding_boxes: list of size (observations_count, cameras_count) bounding boxes for each dynamic object instance
        :param bounding_boxes_validity: list of size (observations_count, cameras_count) bounding boxes validities for each dynamic object instance
        :param observations_paths: list of size (observations_count, cameras_count) strings with locations on disk of each frame, None if the frame is not on disk
        :param global_frame_indexes: list of size (observations_count) of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: list of size (observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_index: index of the video in the dataset
        :param video: the original video object
        :param transforms: transform to apply to each frame in the observations. Must return torch tensors
        :param optical_flow: list of size (observations_count, cameras_count) optical flows
        :param optical_flow_transforms: transform to apply to each optical flow. Must be not None if optical flows are present. Must return torch tensors
        :param keypoints: list of size (observations_count, cameras_count) keypoints for each dynamic object instance
        :param keypoints_validity: list of size (observations_count, cameras_count) keypoints validities for each dynamic object instance
        :param object_poses: list of size (observations_count) with pose parameters for each dynamic object instance
        :param crop_regions: lsit of size (observations_count, cameras_count) with crop regions (left, top, right, bottom) normalized in [0, 1]
        '''

        self.observations_count = len(observations)
        self.observations_stacking = len(observations[0])
        self.cameras_count = len(observations[0][0])

        if len(actions) != self.observations_count or len(rewards) != self.observations_count or len(dones) != self.observations_count or len(cameras) != self.observations_count \
            or self.observations_count != len(focals) or self.observations_count != len(bounding_boxes) or self.observations_count != len(bounding_boxes_validity) \
            or self.observations_count != len(global_frame_indexes) or self.observations_count != len(video_frame_indexes):
            raise Exception("Missing elements in the current batch")
        if len(cameras[0]) != self.cameras_count:
            raise Exception("Missing elements in the current batch")

        # Checks the number of dynamic object instances in the first observation of the first camera [0][0]
        if bounding_boxes[0][0].shape[1] != bounding_boxes_validity[0][0].shape[0]:
            raise Exception(f"Bounding boxes contain {bounding_boxes[0][0].shape[1]} dynamic object instances, but"
                            f"bounding boxes valitities contain {bounding_boxes_validity[0][0].shape[0]} dynamic object instances")

        self.actions = actions
        self.rewards = rewards
        self.metadata = metadata
        self.dones = dones
        self.cameras = cameras
        self.focals = focals
        self.bounding_boxes = bounding_boxes
        self.bounding_boxes_validity = bounding_boxes_validity
        self.observations_paths = observations_paths
        self.global_frame_indexes = global_frame_indexes
        self.video_frame_indexes = video_frame_indexes
        self.video_index = video_index
        self.video = video
        self.transforms = transforms
        self.optical_flow_transforms = optical_flow_transforms
        self.keypoints = keypoints
        self.keypoints_validity = keypoints_validity
        self.object_poses = object_poses
        self.crop_regions = crop_regions

        # Transforms each observation and puts them in the (observations_count, cameras_count, observation_stacking order
        transformed_observations = []
        for observation_idx in range(self.observations_count):
            current_l1_observations = []
            for camera_idx in range(self.cameras_count):
                current_l2_observations = []
                for stack_idx in range(self.observations_stacking):
                    current_observation = observations[observation_idx][stack_idx][camera_idx]
                    # Applies a transformation if present
                    if self.transforms is not None:
                        current_observation = self.transforms(current_observation)
                    current_l2_observations.append(current_observation)
                if torch.is_tensor(current_l2_observations[0]):
                    current_l2_observations = torch.cat(current_l2_observations, dim=0)
                else:
                    assert(len(current_l2_observations) == 1)  # need observation stacking = 1 if images are not tensors
                    current_l2_observations = current_l2_observations[0]  # Eliminates stacking dimension
                current_l1_observations.append(current_l2_observations)
            if torch.is_tensor(current_l1_observations[0]):
                current_l1_observations = torch.stack(current_l1_observations)
            transformed_observations.append(current_l1_observations)
        if torch.is_tensor(transformed_observations[0]):
            transformed_observations = torch.stack(transformed_observations)
        self.observations = transformed_observations

        # Transforms each optical flow and puts them in the (observations_count, cameras_count) order
        self.optical_flows = None
        if optical_flows is not None:
            # Applies transformations to the optical flows if present
            if self.optical_flow_transforms is not None:
                optical_flows = [[self.optical_flow_transforms(current_flow) for current_flow in current_camera_flows] for current_camera_flows in optical_flows]
            else:
                raise Exception("Optical Flow transformations are None, but must at least include the conversion from Numpy to PyTorch")
            # Stacks the camera dimension
            optical_flows = [torch.stack(current_camera_flows, dim=0) for current_camera_flows in optical_flows]
            # Stacks the observations_count dimention
            self.optical_flows = torch.stack(optical_flows, dim=0)

        # Converts arrays to torch
        self.cameras = [[current_element.to_torch() for current_element in current_camera] for current_camera in self.cameras]
        self.bounding_boxes = [[torch.from_numpy(current_element) for current_element in current_camera] for current_camera in self.bounding_boxes]
        self.bounding_boxes_validity = [[torch.from_numpy(current_element) for current_element in current_camera] for current_camera in self.bounding_boxes_validity]
        if self.has_keypoints():  # Converts keypoints to torch
            self.keypoints = [[torch.from_numpy(current_element) for current_element in current_camera] for current_camera in self.keypoints]
            self.keypoints_validity = [[torch.from_numpy(current_element) for current_element in current_camera] for current_camera in self.keypoints_validity]
        if self.has_object_poses():  # Converts object poses to torch
            self.object_poses = [[current_object_pose.to_torch() for current_object_pose in current_observation_results] for current_observation_results in self.object_poses]
        if self.has_crop_regions():
            self.crop_regions = torch.as_tensor(self.crop_regions)  # tensor (observations_count, cameras_count, 4)

    def has_keypoints(self) -> bool:
        '''
        Check whether keypoints data is available
        :return: True if keypoints data is present
        '''

        return self.keypoints is not None

    def has_flow(self) -> bool:
        '''
        Check whether optical flow data is available
        :return: True if optical flow data is present
        '''

        return self.optical_flows is not None

    def has_object_poses(self) -> bool:
        '''
        Check whether object poses are available
        :return: True if object poses are available
        '''

        return self.object_poses is not None

    def has_crop_regions(self) -> bool:
        '''
        Check whether crop regions are available
        :return: True if crop regions are available
        '''

        return self.crop_regions is not None


class Batch:

    def __init__(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, metadata: List[List[List[Dict]]], dones: torch.Tensor,
                 camera_rotation: torch.Tensor, camera_translation: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor,
                 observations_paths: np.ndarray, global_frame_indexes: torch.Tensor, video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, videos: List[MulticameraVideo],
                 optical_flows: torch.Tensor=None, keypoints: torch.Tensor=None, keypoints_validity: torch.Tensor=None,
                 object_rotation: torch.Tensor=None, object_translation: torch.Tensor=None, crop_regions: torch.Tensor=None):
        '''

        :param observations: (bs, observations_count, cameras_count, 3 * observations_stacking, h, w) tensor with observed images
        :param actions: (bs, observations_count) tensor with observed actions
        :param rewards: (bs, observations_count) tensor with observed rewards
        :param metadata: list of size (bs, observations_count, cameras_count) with batch metadata
        :param dones: (bs, observations_count) tensor with observed dones
        :param camera_rotation (bs, observations_count, cameras_count, 3) tensor with camera rotations
        :param camera_translation (bs, observations_count, cameras_count, 3) tensor with camera translations
        :param focals (bs, observations_count, cameras_count) tensor with camera focals
        :param bounding_boxes (bs, observations_count, cameras_count, 4, dynamic_object_count) tensor with bounding boxes for dynamic object instances
        :param bounding_boxes_validity (bs, observations_count, cameras_count, dynamic_object_count) tensor with bounding boxes validity for dynamic object instances
        :param observations_paths (bs, observations_count, cameras_count) string array with paths for each observation on disk, None if observation is not present on disk
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param videos: list of original bs videos
        :param optical_flows: (bs, observations_count, cameras_count, 2, h, w) tensor with optical flows
        :param keypoints (bs, observations_count, cameras_count, keypoints_count, 3, dynamic_object_count) tensor with keypoints for dynamic object instances
        :param keypoints_validity (bs, observations_count, cameras_count, dynamic_object_count) tensor with keypoints validity for dynamic object instances
        :param object_rotation (bs, observations_count, dynamic_object_count) tensor with camera rotations
        :param object_translation (bs, observations_count, dynamic_object_count) tensor with camera translations
        :param crop_regions (bs, observations_count, cameras_count, 4) tensor with crop regions normalized in [0, 1]
        '''

        self.size = actions.size(1)

        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.metadata = metadata
        self.dones = dones
        self.camera_rotation = camera_rotation
        self.camera_translation = camera_translation
        self.focals = focals
        self.bounding_boxes = bounding_boxes
        self.bounding_boxes_validity = bounding_boxes_validity
        self.observations_paths = observations_paths
        self.global_frame_indexes = global_frame_indexes
        self.video_frame_indexes = video_frame_indexes
        self.video_indexes = video_indexes
        self.video = videos

        self.keypoints = keypoints
        self.keypoints_validity = keypoints_validity

        self.optical_flows = optical_flows

        self.object_rotation = object_rotation
        self.object_translation = object_translation

        self.crop_regions = crop_regions

    def to_cuda(self):
        '''
        Transfers tensors to the gpu
        :return:
        '''
        self.observations = self.observations.cuda()
        self.actions = self.actions.cuda()
        self.rewards = self.rewards.cuda()
        self.dones = self.dones.cuda()
        self.camera_rotation = self.camera_rotation.cuda()
        self.camera_translation = self.camera_translation.cuda()
        self.focals = self.focals.cuda()
        self.bounding_boxes = self.bounding_boxes.cuda()
        self.bounding_boxes_validity = self.bounding_boxes_validity.cuda()
        self.global_frame_indexes = self.global_frame_indexes.cuda()
        self.video_frame_indexes = self.video_frame_indexes.cuda()
        self.video_indexes = self.video_indexes.cuda()

        if self.has_flow():
            self.optical_flows.cuda()

        if self.has_keypoints():
            self.keypoints = self.keypoints.cuda()
            self.keypoints_validity = self.keypoints_validity.cuda()

        if self.has_object_poses():
            self.object_rotation = self.object_rotation.cuda()
            self.object_translation = self.object_translation.cuda()

    def to_tuple(self, cuda=True) -> Tuple:
        '''
        Converts the batch to an input tuple
        :param cuda If True transfers the tensors to the gpu
        :return: (observations, actions, rewards, dones, camera_rotation, camera_translation, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes) tuple
        '''

        if cuda:
            self.to_cuda()

        # Do not return optical flow not to break backward compatibility of the code
        return self.observations, self.actions, self.rewards, self.dones, self.camera_rotation, self.camera_translation, \
               self.focals, self.bounding_boxes, self.bounding_boxes_validity, self.global_frame_indexes, self.video_frame_indexes, self.video_indexes

    def to_keypoints_typle(self, cuda=True):
        '''
        Converts the batch keypoints information to an input tuple
        :param cuda If True transfers the tensors to the gpu
        :return: (keypoints, keypoints_validity) tuple
        '''

        if not self.has_keypoints():
            raise Exception("Keypoints were requested from the batch, but the batch has no keypoints information")

        if cuda:
            self.to_cuda()

        return self.keypoints, self.keypoints_validity

    def to_object_poses_tuple(self, cuda=True):
        '''
        Converts the batch object pose information to an input tuple
        :param cuda If True transfers the tensors to the gpu
        :return: (object_rotation, object_translation) tuple
        '''

        if not self.has_object_poses():
            raise Exception("Object poses were requested from the batch, but the batch has no object pose information")

        if cuda:
            self.to_cuda()

        return self.object_rotation, self.object_translation

    def pin_memory(self):

        self.observations.pin_memory()
        self.actions.pin_memory()
        self.rewards.pin_memory()
        self.dones.pin_memory()
        self.camera_rotation.pin_memory()
        self.camera_translation.pin_memory()
        self.focals.pin_memory()
        self.bounding_boxes.pin_memory()
        self.bounding_boxes_validity.pin_memory()
        self.global_frame_indexes.pin_memory()
        self.video_frame_indexes.pin_memory()
        self.video_indexes.pin_memory()

        if self.has_flow():
            self.optical_flows.pin_memory()

        if self.has_keypoints():
            self.keypoints.pin_memory()
            self.keypoints_validity.pin_memory()

        if self.has_object_poses():
            self.object_rotation.pin_memory()
            self.object_translation.pin_memory()

        return self

    def has_keypoints(self) -> bool:
        '''
        Check whether keypoints data is available
        :return: True if keypoints data is present
        '''

        return self.keypoints is not None

    def has_flow(self) -> bool:
        '''
        Check whether optical flow data is available
        :return: True if optical flow data is present
        '''

        return self.optical_flows is not None

    def has_object_poses(self) -> bool:
        '''
        Check whether object poses are available
        :return: True if object poses are available
        '''

        return self.object_rotation is not None and self.object_translation is not None

    def has_crop_regions(self) -> bool:
        '''
        Check whether crop regions data is available
        :return: True if crop regions data is present
        '''

        return self.crop_regions is not None


def single_batch_elements_collate_fn(batch: List[BatchElement]) -> Batch:
    '''
    Creates a batch starting from single batch elements

    :param batch: List of batch elements
    :return: Batch representing the passed batch elements
    '''

    observations_tensor = torch.stack([current_element.observations for current_element in batch])

    actions_tensor = torch.stack([torch.tensor(current_element.actions, dtype=torch.int) for current_element in batch], dim=0)
    rewards_tensor = torch.stack([torch.tensor(current_element.rewards) for current_element in batch], dim=0)
    dones_tensor = torch.stack([torch.tensor(current_element.dones) for current_element in batch], dim=0)

    # Converts the PoseParameter objects into separate rotation and translation lists of dimension (bs, observations_count, cameras_count)
    all_camera_rotations = []
    all_camera_translations = []
    for current_batch_element in batch:
        l1_list_rotation = []
        l1_list_translation = []
        for current_observation in current_batch_element.cameras:
            l2_list_rotation = []
            l2_list_translation = []
            for current_camera in current_observation:
                current_rotation, current_translation = current_camera.get_rotation_translation()
                l2_list_rotation.append(current_rotation)
                l2_list_translation.append(current_translation)
            l1_list_rotation.append(torch.stack(l2_list_rotation))
            l1_list_translation.append(torch.stack(l2_list_translation))
        all_camera_rotations.append(torch.stack(l1_list_rotation))
        all_camera_translations.append(torch.stack(l1_list_translation))

    camera_rotation_tensor = torch.stack(all_camera_rotations)
    camera_translation_tensor = torch.stack(all_camera_translations)

    focals = torch.as_tensor([current_element.focals for current_element in batch], dtype=torch.float)
    bounding_boxes = torch.stack([torch.stack([torch.stack(current_camera_element) for current_camera_element in current_element.bounding_boxes]) for current_element in batch])
    bounding_boxes_validity = torch.stack([torch.stack([torch.stack(current_camera_element) for current_camera_element in current_element.bounding_boxes_validity]) for current_element in batch])
    observations_paths = np.asarray([current_element.observations_paths for current_element in batch], dtype=object)  # dtype object to support strings of arbitrary length

    keypoints = None
    keypoints_validity = None
    if batch[0].has_keypoints():
        keypoints = torch.stack([torch.stack([torch.stack(current_camera_element) for current_camera_element in current_element.keypoints]) for current_element in batch])
        keypoints_validity = torch.stack([torch.stack([torch.stack(current_camera_element) for current_camera_element in current_element.keypoints_validity]) for current_element in batch])

    global_frame_indexes_tensor = torch.stack([torch.tensor(current_element.global_frame_indexes, dtype=torch.long) for current_element in batch], dim=0)
    video_frame_indexes_tensor = torch.stack([torch.tensor(current_element.video_frame_indexes, dtype=torch.long) for current_element in batch], dim=0)
    video_indexes_tensor = torch.tensor([current_element.video_index for current_element in batch], dtype=torch.long)

    batch_metadata = [batch_element.metadata for batch_element in batch]

    videos = [current_element.video for current_element in batch]

    # Stacks the optical flow if present
    optical_flows_tensor = None
    if batch[0].has_flow():
        optical_flows_tensor = torch.stack([current_element.optical_flows for current_element in batch])

    # Converts the PoseParameter object for object poses into rotation and translation tensors
    object_rotation_tensor = None
    object_translation_tensor = None
    if batch[0].has_object_poses():
        all_object_rotations = []
        all_object_translations = []
        for current_batch_element in batch:
            l1_list_rotation = []
            l1_list_translation = []
            for current_observation in current_batch_element.object_poses:
                l2_list_rotation = []
                l2_list_translation = []
                for current_pose in current_observation:
                    current_rotation, current_translation = current_pose.get_rotation_translation()
                    l2_list_rotation.append(current_rotation)
                    l2_list_translation.append(current_translation)
                l1_list_rotation.append(torch.stack(l2_list_rotation, dim=-1))  # Stacks on the last dimension to pose the object dimension in the last position
                l1_list_translation.append(torch.stack(l2_list_translation, dim=-1))
            all_object_rotations.append(torch.stack(l1_list_rotation))
            all_object_translations.append(torch.stack(l1_list_translation))

        object_rotation_tensor = torch.stack(all_object_rotations)
        object_translation_tensor = torch.stack(all_object_translations)

    crop_regions = None
    if batch[0].has_crop_regions():
        crop_regions = torch.stack([batch_element.crop_regions for batch_element in batch])

    return Batch(observations_tensor, actions_tensor, rewards_tensor, batch_metadata, dones_tensor, camera_rotation_tensor,
                 camera_translation_tensor, focals, bounding_boxes, bounding_boxes_validity, observations_paths, global_frame_indexes_tensor,
                 video_frame_indexes_tensor, video_indexes_tensor, videos, optical_flows=optical_flows_tensor,
                 keypoints=keypoints, keypoints_validity=keypoints_validity, object_rotation=object_rotation_tensor,
                 object_translation=object_translation_tensor, crop_regions=crop_regions)


def multiple_batch_elements_collate_fn(batch: List[Tuple[BatchElement]]) -> List[Batch]:
    '''
    Creates a batch starting from groups of corresponding batch elements

    :param batch: List of groups of batch elements
    :return: A List with cardinality equal to the number of batch elements of each group where
             the ith tuple item is the batch of all elements in the ith position in each group
    '''

    cardinality = len(batch[0])

    # Transforms the ith element of each group into its batch
    output_batches = []
    for idx in range(cardinality):
        # Extract ith element
        current_batch_elements = [current_elements_group[idx] for current_elements_group in batch]
        # Creates ith batch
        current_output_batch = single_batch_elements_collate_fn(current_batch_elements)
        output_batches.append(current_output_batch)

    return output_batches









