import copy
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Union

import glob
from PIL import Image
import os
import pickle

import torch
import numpy as np

from dataset.lazy_image import LazyImage
from utils.lib_3d.pose_parameters import PoseParametersNumpy, PoseParameters


class Video:
    '''
     A video from the dataset
     Metadata are always kept into memory, while frames are loaded from disk on demand only
    '''

    actions_filename = "actions.pkl"
    rewards_filename = "rewards.pkl"
    metadata_filename = "metadata.pkl"
    dones_filename = "dones.pkl"
    cameras_filename = "cameras.pkl"
    focals_filename = "focals.pkl"
    bounding_boxes_filename = "bounding_boxes.pkl"
    bounding_boxes_validity_filename = "bounding_box_validity.pkl"

    optical_flow_directory = "flow"
    optical_flow_extension = "npy"

    keypoints_filename = "keypoints.pkl"
    keypoints_validity_filename = "keypoints_validity.pkl"
    object_poses_filename = "object_poses.pkl"
    crop_region_filename = "crop_region.pkl"

    def __init__(self):
        self.frames = None
        self.actions = None
        self.rewards = None
        self.metadata = None
        self.dones = None
        self.cameras = None
        self.focals = None
        self.bounding_boxes = None
        self.bounding_boxes_validity = None

        self.keypoints = None
        self.keypoints_validity = None

        self.object_poses = None

        self.optical_flow = None  # None if not loaded into memory or not available

        self.crop_region = None

        # Stores the size of frames in the video
        self.cached_image_size = None

    def add_content(self, frames: Union[List[Image.Image], List[LazyImage], str], actions: List[int], rewards: List[float], metadata: List[Dict],
                    dones: List[bool], cameras: List[PoseParametersNumpy], focals: List[float], bounding_boxes: List[np.ndarray],
                    bounding_boxes_validity: List[np.ndarray], keypoints: List[np.ndarray]=None, keypoints_validity: List[np.ndarray]=None,
                    object_poses: List[List[PoseParametersNumpy]]=None, crop_region: List[float]=None):
        '''
        Adds the contents to the video
        :param frames: list of all video frames. Alternatively list of directory containing the frames
        :param actions: list of the actions selected in the current frame
        :param rewards: list of the reward generated for arriving in the current frame
        :param metadata: list of metadata associated to the current frame
        :param dones: list of booleans representing whether the current observation was the last of the episode
        :param cameras: list of camera parameters corresponding to each observation
        :param focals: list of camera focal lengths corresponding to each observation
        :param bounding_boxes: list of (4, dynamic_object_count) array with bounding boxes for each observation
        :param bounding_boxes_validity: list of (dynamic_object_count) array indicating the presence of objects in each observation
        :param keypoints: list of (keypoints_count, 3, dynamic_object_count) array with keypoints for each observation. The second dimension is given by (height, width, confidence_score) and the prediction confidence
        :param keypoints_validity: list of (dynamic_object_count) array indicating the presence of keypoints in each observation
        :param object_poses: list of length  dynamic_object_count with pose parameters for each object in the video
        :param crop_region: list (left, top, right, bottom) with coordinates of a crop region normalized in [0, 1]
        :return:
        '''

        if len(cameras) != len(actions) or len(cameras) != len(rewards) or len(cameras) != len(metadata) or len(cameras) != len(dones) or \
           len(cameras) != len(focals) or len(cameras) != len(bounding_boxes) or len(cameras) != len(bounding_boxes_validity):
            raise Exception("All arguments must have the same length")

        # First loads whatever is not an image
        self.actions = actions
        self.rewards = rewards
        self.metadata = metadata
        self.dones = dones
        self.cameras = cameras
        self.focals = focals
        self.bounding_boxes = bounding_boxes
        self.bounding_boxes_validity = bounding_boxes_validity

        self.keypoints = keypoints
        self.keypoints_validity = keypoints_validity
        self.object_poses = object_poses
        self.crop_region = crop_region

        # Accepts both frames on disk and frames in memory
        if isinstance(frames, str):
            self.frames = self.load_lazy_images_from_path(frames)
            self.frames_path = frames
            self.extension = self.discover_extension(frames)
        # If the frames are in memory
        else:
            self.frames = frames
            self.frames_path = None
            self.extension = None

        # Sets default values in the metadata if needed
        self.check_metadata_and_set_defaults()

        if not isinstance(self.cameras[0], PoseParametersNumpy):
            raise Exception("Received camera parameters that are not PoseParametersNumpy. Are you using the old interface?")
        if not type(self.bounding_boxes[0]).__module__ == np.__name__:
            raise Exception("Received bounding boxes that are not numpy arrays. Are you using the old interface?")
        if not type(self.bounding_boxes_validity[0]).__module__ == np.__name__:
            raise Exception("Received bounding boxes validities that are not numpy arrays. Are you using the old interface?")

        if self.bounding_boxes[0].shape[1] != self.bounding_boxes_validity[0].shape[0]:
            raise Exception(f"Bounding boxes contain {self.bounding_boxes[0].shape[1]} dynamic object instances, but"
                            f"bounding boxes valitities contain {self.bounding_boxes_validity[0].shape[0]} dynamic object instances")

        dynamic_objects_count = self.bounding_boxes[0].shape[-1]
        if object_poses is not None:
            for current_poses in object_poses:
                if len(current_poses) != dynamic_objects_count:
                    raise Exception(f"The dataset has {dynamic_objects_count} dynamic objects, but {len(current_poses)} were detected in the object poses")

        if self.crop_region is not None and len(self.crop_region) != 4:
            raise Exception(f"Crop region must have length 4, but is {self.crop_region}")

    def load_lazy_images_from_path(self, path: str) -> List[LazyImage]:
        '''
        Loads all the frames present in a certain path

        :param path: path where to load frames from
        :return: list of LazyImage representing all the images in the path
        '''

        extension = self.discover_extension(path)
        lazy_images = []

        current_idx = 0
        current_filename = self._index_to_filename(current_idx)
        current_filename = os.path.join(path, f'{current_filename}.{extension}')
        while os.path.isfile(current_filename):
            # Gets the lazy image
            lazy_images.append(LazyImage(current_filename))

            current_idx += 1
            current_filename = self._index_to_filename(current_idx)
            current_filename = os.path.join(path, f'{current_filename}.{extension}')

        return lazy_images

    def _index_to_filename(self, idx):
        return f'{idx:05}'

    def check_none_coherency(self, sequence):
        '''
        Checks that the sequence either has all values set to None or to a not None value
        Raises an exception if the sequence does not satisfy the criteria
        :param sequence: the sequence to check
        :return:
        '''

        has_none = False
        has_not_none = False

        for element in sequence:
            if element is None:
                has_none = True
            else:
                has_not_none = True
            if has_none and has_not_none:
                raise Exception(f"Video dataset at {self.frames_path} metadata error: both None and not None data are present")

    def check_metadata_and_set_defaults(self):
        '''
        Checks the medatata and sets default values if None are present
        :return:
        '''

        # Checks coherency of None values in the metadata
        self.check_none_coherency(self.actions)
        self.check_none_coherency(self.rewards)
        self.check_none_coherency(self.metadata)
        self.check_none_coherency(self.dones)
        self.check_none_coherency(self.cameras)
        self.check_none_coherency(self.focals)
        self.check_none_coherency(self.bounding_boxes)
        self.check_none_coherency(self.bounding_boxes_validity)

        if self.actions[0] is None:
            self.actions = [0] * len(self.actions)
        if self.rewards[0] is None:
            self.rewards = [0.0] * len(self.rewards)
        if self.metadata[0] is None:
            self.metadata = [{}] * len(self.metadata)
        if self.dones[0] is None:
            self.dones = [False] * len(self.dones)
        if self.cameras[0] is None:
            self.cameras = [PoseParametersNumpy([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) for _ in range(len(self.cameras))]
        if self.focals[0] is None:
            self.focals = [0.0] * len(self.focals)
        if self.bounding_boxes[0] is None:
            self.bounding_boxes = [np.zeros((4, 1)) for _ in range(len(self.bounding_boxes))]
        if self.bounding_boxes_validity[0] is None:
            self.bounding_boxes_validity = [np.asarray([False]) for _ in range(len(self.bounding_boxes_validity))]

    def discover_extension(self, path: str) -> str:
        '''
        Discovers the extension of dataset images at the given path
        :param path:
        :return:
        '''

        # Discover extension of videos
        padded_index = self._index_to_filename(0)
        results = glob.glob(os.path.join(path, f'{padded_index}.*'))
        if len(results) != 1:
            raise Exception("Could not find first video frame")
        extension = results[0].split(".")[-1]
        return extension

    def load(self, path):

        if not os.path.isdir(path):
            raise Exception(f"Cannot load video: '{path}' is not a directory")

        # Load LazyImages to avoid loading frames into memory
        self.frames_path = path
        self.frames = self.load_lazy_images_from_path(path)
        self.optical_flow_path = os.path.join(self.frames_path, self.optical_flow_directory)

        # Load data as pickle objects
        if os.path.isfile(os.path.join(path, Video.actions_filename)):
            with open(os.path.join(path, Video.actions_filename), 'rb') as f:
                self.actions = pickle.load(f)
        else:
            self.actions = [0] * len(self.frames)
            print("Warning: actions not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.rewards_filename)):
            with open(os.path.join(path, Video.rewards_filename), 'rb') as f:
                self.rewards = pickle.load(f)
        else:
            self.rewards = [0.0] * len(self.frames)
            print("Warning: rewards not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.metadata_filename)):
            with open(os.path.join(path, Video.metadata_filename), 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = [{} for _ in range(len(self.frames))]
            print("Warning: metadata not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.dones_filename)):
            with open(os.path.join(path, Video.dones_filename), 'rb') as f:
                self.dones = pickle.load(f)
        else:
            self.dones = [False] * len(self.frames)
            print("Warning: dones not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.cameras_filename)):
            with open(os.path.join(path, Video.cameras_filename), 'rb') as f:
                self.cameras = pickle.load(f)
        else:
            self.cameras = [PoseParametersNumpy([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) for _ in range(len(self.frames))]
            print("Warning: cameras not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.focals_filename)):
            with open(os.path.join(path, Video.focals_filename), 'rb') as f:
                self.focals = pickle.load(f)
        else:
            self.focals = [1.0] * len(self.frames)
            print("Warning: focals not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.bounding_boxes_filename)):
            with open(os.path.join(path, Video.bounding_boxes_filename), 'rb') as f:
                self.bounding_boxes = pickle.load(f)
        else:
            self.bounding_boxes = [np.zeros((4, 1), dtype=np.float) for _ in range(len(self.frames))]
            print("Warning: bounding boxes not found during video loading. Should be treated as an error if happening outside of data processing scripts.")
        if os.path.isfile(os.path.join(path, Video.bounding_boxes_validity_filename)):
            with open(os.path.join(path, Video.bounding_boxes_validity_filename), 'rb') as f:
                self.bounding_boxes_validity = pickle.load(f)
        else:
            self.bounding_boxes_validity = [np.zeros((1,), dtype=np.bool) for _ in range(len(self.frames))]
            print("Warning: bounding boxes validity not found during video loading. Should be treated as an error if happening outside of data processing scripts.")

        # Loads the keypoints if they are present
        if self.has_keypoints():
            with open(os.path.join(path, Video.keypoints_filename), 'rb') as f:
                self.keypoints = pickle.load(f)
            with open(os.path.join(path, Video.keypoints_validity_filename), 'rb') as f:
                self.keypoints_validity = pickle.load(f)

        # Loads object poses if they are present
        if self.has_object_poses():
            with open(os.path.join(path, Video.object_poses_filename), 'rb') as f:
                self.object_poses = pickle.load(f)

        # Loads crop reigon if present
        if self.has_crop_region():
            with open(os.path.join(path, Video.crop_region_filename), 'rb') as f:
                self.crop_region = pickle.load(f)

        # Old data format detected, conversion is needed
        if isinstance(self.cameras[0], PoseParameters):
            self.cameras = [PoseParametersNumpy(*current_element.get_rotation_translation()) for current_element in self.cameras]
        if torch.is_tensor(self.bounding_boxes[0]):
            self.bounding_boxes = [current_element.cpu().numpy() for current_element in self.bounding_boxes]
        if torch.is_tensor(self.bounding_boxes_validity[0]):
            self.bounding_boxes_validity = [current_element.cpu().numpy() for current_element in self.bounding_boxes_validity]

        #print("Warning: loading fake data")
        #self.cameras = [PoseParameters([0.0,0,0], [0.0,0,0]) for _ in range(len(self.actions))]
        #self.bounding_boxes = [torch.zeros((4, 2), dtype=torch.float32) for _ in range(len(self.actions))]
        #self.bounding_boxes_validity = [torch.zeros((2), dtype=torch.bool) for _ in range(len(self.actions))]

        frames_count = len(self.actions)
        if frames_count != len(self.rewards) or frames_count != len(self.metadata) or frames_count != len(self.dones)\
                or frames_count != len(self.cameras) or frames_count != len(self.focals)\
                or frames_count != len(self.bounding_boxes) or frames_count != len(self.bounding_boxes_validity):
            raise Exception("Read data have inconsistent number of frames")
        # Checks the integrity of keypoints
        if self.has_keypoints() and (frames_count != len(self.keypoints) or frames_count != len(self.keypoints_validity)):
            raise Exception("Read data have inconsistent number of frames")
        if self.has_object_poses() and frames_count != len(self.object_poses):
            raise Exception("Read data have inconsistent number of frames")

        # Sets detault values in the metadata if needed
        self.check_metadata_and_set_defaults()

        self.extension = self.discover_extension(path)

        # Checks integrity of the optical flow if present
        if self.has_flow():
            integrity_preserved = self.check_flow_integrity()
            if not integrity_preserved:
                raise Exception(f"Optical Flow data in '{self.frames_path}' is corrupt. Is an optical flow present for each object and each frame?")

        if self.has_crop_region() and len(self.crop_region) != 4:
            raise Exception(f"Crop region must have length 4, but is {self.crop_region}")

    def check_flow_integrity(self) -> bool:
        '''
        Check whether optical flow data is correct
        :return: True if the integrity of optical flow data is correct
        '''

        objects_count = self.get_objects_count()
        frames_count = self.get_frames_count()

        # Checks flow for each object
        for object_idx in range(objects_count):
            current_optical_flow_directory = os.path.join(self.optical_flow_path, self._index_to_filename(object_idx))
            # Checks flow for each frame
            for frame_idx in range(frames_count):
                current_filename = os.path.join(current_optical_flow_directory, f"{self._index_to_filename(frame_idx)}.{self.optical_flow_extension}")
                # Return false if optical flow is not present
                if not os.path.isfile(current_filename):
                    return False

        return True

    def get_image_size(self) -> Tuple[int, int]:
        '''
        Returns the dimensions of the image
        :return: (width, height) of the image
        '''

        # Lazy computation of image size
        if self.cached_image_size is None:
            first_frame = self.get_frame_at(0)
            width, height = first_frame.size
            self.cached_image_size = (width, height)

        return self.cached_image_size

    def has_keypoints(self) -> bool:
        '''
        Check whether keypoints data is available
        :return: True if keypoints data is present
        '''

        # Checks if they are stored in memory or whether they are available on disk
        return self.keypoints is not None or (self.frames_path is not None and os.path.isfile(os.path.join(self.frames_path, Video.keypoints_filename)))

    def has_flow(self) -> bool:
        '''
        Check whether optical flow data is available
        :return: True if optical flow data is present
        '''

        return os.path.isdir(self.optical_flow_path)

    def has_object_poses(self) -> bool:
        '''
        Check whether object poses are available
        :return: True if object poses are available
        '''

        # Checks if they are stored in memory or whether they are available on disk
        return self.object_poses is not None or (self.frames_path is not None and os.path.isfile(os.path.join(self.frames_path, Video.object_poses_filename)))

    def has_crop_region(self) -> bool:
        '''
        Check whether crop region data is available
        :return: True if crop region is present
        '''

        # Checks if they are stored in memory or whether they are available on disk
        return self.crop_region is not None or (self.frames_path is not None and os.path.isfile(os.path.join(self.frames_path, Video.crop_region_filename)))

    def get_objects_count(self) -> int:
        '''
        Obtains the number of objects in the video
        :return:
        '''

        if self.actions is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")

        return self.bounding_boxes[0].shape[-1]

    def get_frames_count(self) -> int:
        if self.actions is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")

        return len(self.actions)

    def get_frame_at(self, idx: int, translate_lazy=True) -> Union[Image.Image, LazyImage]:
        '''
        Returns the frame corresponding to the specified index

        :param idx: index of the frame to retrieve in [0, frames_count-1]
        :param translate_lazy: if True, LazyImage instances are loaded into memory before being returned, otherwise LazyImage instances are returned as such
        :return: The frame at the specified index
        '''
        if self.actions is None or self.frames is None:
            raise Exception("Video has not been initialized. Did you forget to call load()?")
        if idx < 0 or idx >= len(self.actions):
            raise Exception(f"Index {idx} is out of range")

        image = self.frames[idx]
        # If the image is lazy and must be loaded
        if isinstance(image, LazyImage) and translate_lazy:
            image = image.get_image()
        # If the image is lazy and must not be loaded
        elif isinstance(image, LazyImage):
            return image  # Directly returns the LazyImage

        # It must be an Image
        image = self.remove_transparency(image)
        return image

    def get_frame_path_at(self, idx: int) -> str:
        '''
        Returns the path for the frame at the specified index
        :param idx: Index of the frame path to retrieve
        :return: The path of the frame at the specified index, None if the frame is not on disk
        '''

        image = self.frames[idx]

        # There is a path only if the frame is loaded in a lazy image
        if isinstance(image, LazyImage):
            return image.path
        else:
            return None

    def get_flow_crop(self, frame_idx: int, object_idx: int):
        '''
        Gets the flow crop corresponding to the given object

        :param frame_idx: index of the optical flow to retrieve in [0, frames_count-1]
        :param object_idx: index of the object for which retrieve optical flow [0, objects_count]
        :return: (height, width, 2) normalized optical flow in the range [0, 1] at the specified index
                                    the last two dimensions are in y, x order instead of the usual x, y order for optical flow
        '''

        crop_filename = os.path.join(self.optical_flow_path, self._index_to_filename(object_idx), f"{self._index_to_filename(frame_idx)}.{self.optical_flow_extension}")
        flow_crop = np.load(crop_filename)

        # Flow crop is in x, y order on disk
        # Invert to y, x
        flow_crop = flow_crop[..., [1, 0]]
        # Flows may be stored in different precision formats, load as single precision
        flow_crop = flow_crop.astype(np.float32)
        return flow_crop

    def rescale_bounding_box(self, bounding_box: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        '''

        :param bounding_boxes: bounding box in range [0, 1] (left, top, right, bottom) coordinates
        :param image_size: (height, width) size of the image
        :return: bounding boxes rescaled in image coordinates expressed as integers
        '''

        height, width = image_size

        rescaled_box = bounding_box.copy()
        rescaled_box[0] *= width
        rescaled_box[1] *= height
        rescaled_box[2] *= width
        rescaled_box[3] *= height

        rescaled_box = np.rint(rescaled_box).astype(np.int)
        return rescaled_box

    def get_center(self, bounding_box: np.ndarray):

        return (bounding_box[:2] + bounding_box[2:]) / 2

    def clamp_flow(self, flow: np.ndarray, bounding_box: np.ndarray, image_size: Tuple[int, int]):
        '''
        Clamps the flow and the bounding box such that they do not exceed the boundaries of the image

        :param flow: (height, width, 2) optical flow
        :param bounding_box: (left, top, right, bottom) bounding box
        :param image_size: (height, width) of the image
        :return: flow and bounding box modified such that they do not exceed the boundaries of the image
        '''

        bounding_box = bounding_box.copy()

        image_height, image_width = image_size

        flow_height = flow.shape[0]
        flow_width = flow.shape[1]

        box_width = bounding_box[2] - bounding_box[0]
        box_height = bounding_box[3] - bounding_box[1]

        if flow_height != box_height or flow_width != box_width:
            raise Exception(f"Flow and box dimensions do not match. Flow: ({flow_height}, {flow_width}) - Box: ({box_height}, {box_width})")

        if bounding_box[0] < 0:
            flow = flow[:, int(-bounding_box[0]):]
            bounding_box[0] = 0
        # Last bounding box pixel is exclusive, so just check if the box is strictly larger than the image
        if bounding_box[2] > image_width:
            flow = flow[:, :-int(bounding_box[2] - image_width)]
            bounding_box[2] = image_width
        if bounding_box[1] < 0:
            flow = flow[int(-bounding_box[1]):]
            bounding_box[1] = 0
        if bounding_box[3] > image_height:
            flow = flow[:-int(bounding_box[3] - image_height)]
            bounding_box[3] = image_height

        return flow, bounding_box

    def get_flow_at(self, frame_idx: int) -> np.ndarray:
        '''
        Returns the optical flow corresponding to the specified index

        :param frame_idx: index of the optical flow to retrieve in [0, frames_count-1]
        :return: The normalized optical flow in the range [0, 1] at the specified index
                 (2, height, width) the two channel dimensions are in y, x order instead of the usual x, y order for optical flow
        '''

        # Computes dimensions of full flows
        full_height, full_width = self.get_image_size()
        # Creates empty optical flows
        full_flow = np.zeros((full_height, full_width, 2), np.float32)

        # Inserts the flow of each object into the full flow
        objects_count = self.get_objects_count()
        for object_idx in range(objects_count):

            current_flow_crop = self.get_flow_crop(frame_idx, object_idx)

            # Computes dimensions of flow crops
            crop_height = current_flow_crop.shape[0]
            crop_width = current_flow_crop.shape[1]
            box_half_size = np.asarray([crop_width // 2, crop_height // 2], dtype=np.int)

            # Gets the current bounding box
            current_bounding_box = self.bounding_boxes[frame_idx][..., object_idx]

            # Rescales the bounding boxes
            current_bounding_box = self.rescale_bounding_box(current_bounding_box, (full_height, full_width))

            # Computes the center as an integer coordinate
            bounding_box_center = np.rint(self.get_center(current_bounding_box)).astype(np.int)
            # Computes the box on which to perform cropping
            crop_box = np.concatenate([bounding_box_center - box_half_size, bounding_box_center + box_half_size])

            # Clamps bounding boxes and flow so that they stay within the image
            current_flow_crop, current_crop_box = self.clamp_flow(current_flow_crop, crop_box, (full_height, full_width))

            # Copies the flow into the full flow
            left = current_crop_box[0]
            top = current_crop_box[1]
            right = current_crop_box[2]
            bottom = current_crop_box[3]
            full_flow[top:bottom, left:right] = current_flow_crop

        # Transforms HWC to CHW
        full_flow = np.moveaxis(full_flow, -1, 0)
        return full_flow

    def remove_transparency(self, image, bg_colour=(255, 255, 255)):

        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):

            # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
            alpha = image.convert('RGBA').split()[-1]

            # Create a new background image of our matt color.
            # Must be RGBA because paste requires both images have the same format
            # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
            bg = Image.new("RGBA", image.size, bg_colour + (255,))
            bg.paste(image, mask=alpha)
            bg = bg.convert("RGB")
            return bg
        else:
            return image

    def subsample_split_resize(self, frame_skip: int, output_sequence_length: int, crop_size: Tuple[int]=None, target_size: Tuple[int]=None, min_sequence_length=None, legacy_splitting_compat=False) -> List:
        '''
        Splits the current sequence into a number of sequences of the specified length, skipping the specified number
        of frames in the source sequence between successive frames in the target sequence
        Resizes the output sequence to the target_size

        Optical flow is not preserved in the returned videos

        :param frame_skip: frames to skip in the source sequence between successive frames in the target sequence
        :param output_sequence_length: number of frames in each output sequence. -1 if length should not be modified
        :param crop_size: (left_index, upper_index, right_index, lower_index) size of the crop to take before resizing
        :param target_size: (width, height) size of the frames in the output sequence
        :param min_sequence_length: Minimum length of the sequences to retain. If None, sequences shorter than output_sequence_length are discarded
        :param legacy_splitting_compat: if True, splits the videos exactly as PVG did which sometimes discarded sequences at the end of the video of the correct length
        :return: List of videos representing the split and subsampled source video
        '''

        if min_sequence_length is None:
            min_sequence_length = output_sequence_length

        # Warns in case of loss of optical flow
        if self.has_flow():
            print("Warning: optical flow is present and will be discarded in the subsample_split_resize operation", file=sys.stderr)

        # Subsamples the video. Do not load lazy instances to save memory
        all_frames = [self.get_frame_at(idx, translate_lazy=False) for idx in range(0, self.get_frames_count(), frame_skip + 1)]
        all_actions = self.actions[::frame_skip + 1]
        all_rewards = self.rewards[::frame_skip + 1]
        all_metadata = self.metadata[::frame_skip + 1]
        all_dones = self.dones[::frame_skip + 1]
        all_cameras = self.cameras[::frame_skip + 1]
        all_focals = self.focals[::frame_skip + 1]
        all_bounding_boxes = self.bounding_boxes[::frame_skip + 1]
        all_bounding_boxes_validity = self.bounding_boxes_validity[::frame_skip + 1]

        if self.has_keypoints():
            all_keypoints = self.keypoints[::frame_skip + 1]
            all_keypoints_validity = self.keypoints_validity[::frame_skip + 1]

        if self.has_object_poses():
            all_object_poses = self.object_poses[::frame_skip + 1]

        # Defines cropping and resizing. If frames are lazy, create new lazy frames with given cropping and resizing
        if isinstance(all_frames[0], LazyImage):
            all_frames = copy.deepcopy(all_frames)
            for frame in all_frames:
                frame.target_size = target_size
                frame.crop_size = crop_size
        # If frames are real directly apply the transformations
        else:
            # Crops the frames
            if crop_size is not None:
                all_frames = [frame.crop(crop_size) for frame in all_frames]

            # Resizes the video if needed
            original_width, original_height = all_frames[0].size
            if target_size is not None:
                if original_width != target_size[0] or original_height != target_size[1]:
                    all_frames = [frame.resize(target_size, Image.BICUBIC) for frame in all_frames]

        split_videos = []

        # Applies length splitting if needed
        if output_sequence_length > 0:
            # Splits the subsampled video in constant length sequences
            total_frames = len(all_frames)
            for current_idx in range(0, total_frames, output_sequence_length):

                end_idx = min(current_idx + output_sequence_length, total_frames)

                # If the video is long enough to be retained
                # In PVG a bug discarded the last sequence if the number of frames to build the sequence was exact.
                # So in compat mode we discard the last sequence if its length is exact
                if (end_idx - current_idx >= min_sequence_length) and (not legacy_splitting_compat or end_idx < total_frames):
                    current_frames = all_frames[current_idx:end_idx]
                    current_actions = all_actions[current_idx:end_idx]
                    current_rewards = all_rewards[current_idx:end_idx]
                    current_metadata = all_metadata[current_idx:end_idx]
                    current_dones = all_dones[current_idx:end_idx]
                    current_cameras = all_cameras[current_idx:end_idx]
                    current_focals = all_focals[current_idx:end_idx]
                    current_bounding_boxes = all_bounding_boxes[current_idx:end_idx]
                    current_bounding_boxes_validity = all_bounding_boxes_validity[current_idx:end_idx]

                    current_keypoints = None
                    current_keypoints_validity = None
                    if self.has_keypoints():
                        current_keypoints = all_keypoints[current_idx:end_idx]
                        current_keypoints_validity = all_keypoints_validity[current_idx:end_idx]

                    current_object_poses = None
                    if self.has_object_poses():
                        current_object_poses = all_object_poses[current_idx:end_idx]

                    current_video = Video()
                    current_video.add_content(current_frames, current_actions, current_rewards, current_metadata, current_dones, current_cameras, current_focals, current_bounding_boxes, current_bounding_boxes_validity,
                                              keypoints=current_keypoints, keypoints_validity=current_keypoints_validity, object_poses=current_object_poses, crop_region=self.crop_region)
                    split_videos.append(current_video)

        # Otherwise return the video in original length
        else:
            current_video = Video()
            current_video.add_content(all_frames, all_actions, all_rewards, all_metadata, all_dones, all_cameras, all_focals, all_bounding_boxes, all_bounding_boxes_validity,
                                      keypoints=all_keypoints, keypoints_validity=all_keypoints_validity, object_poses=all_object_poses, crop_region=self.crop_region)
            split_videos.append(current_video)

        return split_videos

    def save_moco(self, path: str, extension="png", target_size=None):
        '''
        Saves a video to the moco format. The video must already be present on disk
        :param path: The location where to save the video in moco format
        :param extension: The extension to use for image files
        :param target_size: (witdh, height) size for the dataset
        :return:
        '''

        if os.path.exists(path):
            raise Exception(f"A directory at '{path}' already exists")

        all_frames = [self.get_frame_at(idx) for idx in range(self.get_frames_count())]

        # Resizes the images if needed
        if target_size is not None:
            all_frames = [current_frame.resize(target_size) for current_frame in all_frames]

        widths, heights = zip(*(i.size for i in all_frames))

        total_width = sum(widths)
        max_height = max(heights)

        concatenated_frame = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for current_image in all_frames:
            concatenated_frame.paste(current_image, (x_offset, 0))
            x_offset += current_image.size[0]

        concatenated_frame.save(f"{path}.{extension}")

    def save(self, path: str, extension="png", exists_ok=False):
        if self.actions is None or self.frames is None:
            raise Exception("Video has not been initialized. Did you forget to call add_content()?")
        if exists_ok == False and os.path.isdir(path):
            raise Exception(f"A directory at '{path}' already exists")

        # Creates the directory
        Path(path).mkdir(exist_ok=True)

        # Saves all the frames
        for idx in range(self.get_frames_count()):
            frame = self.get_frame_at(idx)
            padded_index = self._index_to_filename(idx)
            filename = os.path.join(path, f'{padded_index}.{extension}')
            frame.save(filename)

        # Saves other data as pickle objects
        with open(os.path.join(path, Video.actions_filename), 'wb') as f:
            pickle.dump(self.actions, f)
        with open(os.path.join(path, Video.rewards_filename), 'wb') as f:
            pickle.dump(self.rewards, f)
        with open(os.path.join(path, Video.metadata_filename), 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(os.path.join(path, Video.dones_filename), 'wb') as f:
            pickle.dump(self.dones, f)
        with open(os.path.join(path, Video.cameras_filename), 'wb') as f:
            pickle.dump(self.cameras, f)
        with open(os.path.join(path, Video.focals_filename), 'wb') as f:
            pickle.dump(self.focals, f)
        with open(os.path.join(path, Video.bounding_boxes_filename), 'wb') as f:
            pickle.dump(self.bounding_boxes, f)
        with open(os.path.join(path, Video.bounding_boxes_validity_filename), 'wb') as f:
            pickle.dump(self.bounding_boxes_validity, f)

        if self.has_keypoints():
            # Saves the keypoints
            with open(os.path.join(path, Video.keypoints_filename), 'wb') as f:
                pickle.dump(self.keypoints, f)
            with open(os.path.join(path, Video.keypoints_validity_filename), 'wb') as f:
                pickle.dump(self.keypoints_validity, f)

        if self.has_object_poses():
            # Saves the object poses
            with open(os.path.join(path, Video.object_poses_filename), 'wb') as f:
                pickle.dump(self.object_poses, f)

        if self.has_crop_region():
            # Saves the crop region
            with open(os.path.join(path, Video.crop_region_filename), 'wb') as f:
                pickle.dump(self.crop_region, f)

