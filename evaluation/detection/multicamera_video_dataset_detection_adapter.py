import os
import sys
import time
from typing import Set, List, Dict, Callable, Tuple

import torch
import torchvision

import evaluation.detection.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataset.batching import BatchElement, single_batch_elements_collate_fn, multiple_batch_elements_collate_fn
from dataset.multicamera_video import MulticameraVideo
from dataset.transforms import TransformsGenerator
from dataset.video import Video

from torch.utils.data import Dataset

from dataset.video_dataset import MulticameraVideoDataset


class MulticameraVideoDatasetDetectionAdapter(Dataset):

    def __init__(self, path: str, transforms: Callable, size: Tuple[int], boxes_expansion_factor=(1.0, 1.0)):
        '''
        Builds an adapter for detection training for the dataset at the specified path

        :param path: path of the dataset
        :param size: (height, width) to which resize the frames
        :param frame_transform: transformation to apply to dataset frames
        :param boxes_expansion_factor: (rows, columns) factor with which to multiply the height and width of the bounding boxes
        '''

        self.path = path
        self.transforms = transforms
        self.size = size
        self.expansion_factor_rows = boxes_expansion_factor[0]
        self.expansion_factor_columns = boxes_expansion_factor[1]

        batching_config = {
            "batch_size": 0, # Unused here

            # Indexes of the camera to use. null to use all cameras
            "allowed_cameras": None,
            # Just one observation per video for detection
            "observations_count": 1,
            # Unused since we just use one observation per video
            "skip_frames": 0,
            # Just one frame
            "observation_stacking": 1,

            "num_workers": 0  # Unused here
        }

        # Need to resize here to also resize the boudning boxes afterwards
        original_transforms = torchvision.transforms.Resize(size)
        self.dataset = MulticameraVideoDataset(path, batching_config, original_transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        original_element = self.dataset[idx]

        # Gets the observations
        assert(len(original_element.observations) == 1)
        assert(len(original_element.observations[0]) == 1)
        image = original_element.observations[0][0]
        # Unique frame identifier
        assert(len(original_element.global_frame_indexes) == 1)
        image_id = torch.tensor(idx)

        image_width, image_height = image.size

        # Gets the bounding boxes. Makes a copy to avoid modifying the original ones
        bounding_boxes = original_element.bounding_boxes[0][0].clone()
        bounding_boxes_validity = original_element.bounding_boxes_validity[0][0]

        # Expands the bounding boxes
        bounding_boxes_dimensions = bounding_boxes[2:, :] - bounding_boxes[..., :2, :]
        bounding_boxes[0, :] -= bounding_boxes_dimensions[0, :] * self.expansion_factor_columns
        bounding_boxes[2, :] += bounding_boxes_dimensions[0, :] * self.expansion_factor_columns
        bounding_boxes[1, :] -= bounding_boxes_dimensions[1, :] * self.expansion_factor_rows

        # Computes the new dimensions
        bounding_boxes_dimensions[0, :] *= self.expansion_factor_columns * image_width
        bounding_boxes_dimensions[1, :] *= self.expansion_factor_rows * image_height

        bounding_boxes_area = bounding_boxes_dimensions[0, :] * bounding_boxes_dimensions[1, :]

        # Do not expand the bounding boxes to the bottom
        # Clamp values of the bounding boxes
        bounding_boxes = torch.clamp(bounding_boxes, min=0.0, max=1.0)

        # Expresses the bounding boxes in pixels
        bounding_box_denormalization_tensor = torch.tensor([image_width, image_height, image_width, image_height]).unsqueeze(-1)
        bounding_boxes = bounding_boxes * bounding_box_denormalization_tensor
        # Puts the boxes in (dynamic_objects_count, 4) format
        bounding_boxes = torch.transpose(bounding_boxes, 0, 1)

        # Filters only valid boxes
        bounding_boxes = bounding_boxes[bounding_boxes_validity, :]

        is_crowd = torch.zeros((bounding_boxes.size(0), ), dtype=torch.int64)

        box_labels = torch.as_tensor([1] * bounding_boxes.size(0), dtype=torch.int64)  # 0=background, 1=dynamic object

        annotations = {
            "boxes": bounding_boxes,
            "labels": box_labels,
            "area": bounding_boxes_area,
            "iscrowd": is_crowd,
            "image_id": image_id
        }

        if self.transforms is not None:
            image, annotations = self.transforms(image, annotations)

        return image, annotations


if __name__ == "__main__":

    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms = T.Compose(transforms)

    dataset = MulticameraVideoDatasetDetectionAdapter("data/tennis_v7_reduced/train", transforms)

    element = dataset[0]
    print(element)