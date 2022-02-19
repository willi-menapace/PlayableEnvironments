import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.layers.residual_block import ResidualBlock
from utils.tensor_folder import TensorFolder


class ObjectEncoderV4(nn.Module):
    '''
    Encodes the style and pose of an object using an attention mechanism
    Accounts for the position from which observations are provided
    '''

    def __init__(self, config: Dict, model_config: Dict):
        '''
        Initializes the model representing the style of an object

        :param config: the configuration file
        :param model_config: the configuration for the specific object
        '''
        super(ObjectEncoderV4, self).__init__()

        self.config = config

        self.input_size = model_config["input_size"]
        self.deformation_features = model_config["deformation_features"]
        self.style_features = model_config["style_features"]

        self.expansion_factor_rows = 0.0
        self.expansion_factor_cols = 0.0
        if "expansion_factor" in model_config:
            self.expansion_factor_rows = model_config["expansion_factor"]["rows"]
            self.expansion_factor_cols = model_config["expansion_factor"]["cols"]

        # Takes as input rgb channels + camera rotations and camera translations
        self.conv1 = nn.Conv2d(3 + 6, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Backbone before feature merging. 1 layer for attention. Attention and other features require different
        # activations so it is dropped from the layer
        self.initial_backbone = nn.Sequential(ResidualBlock(16, 16 + 1, downsample_factor=1, drop_final_activation=True))  # res / 2

        # Backbone after feature merging
        final_backbone_blocks = [
            ResidualBlock(16, 32, downsample_factor=2),  # res / 4
            ResidualBlock(32, 32, downsample_factor=1),  # res / 4
            ResidualBlock(32, 64, downsample_factor=2),  # res / 8
            ResidualBlock(64, 64, downsample_factor=1),  # res / 8
        ]
        self.final_backbone = nn.Sequential(*final_backbone_blocks)

        # Heads for latent code outputs
        self.style_head = nn.Linear(64, self.style_features)
        self.deformation_head = nn.Linear(64, self.deformation_features)

    def expand_bounding_boxes(self, bounding_boxes: torch.Tensor):
        '''
        Expands the bounding boxes to ensure the object is not cut. Original bounding boxes are not altered
        :param bounding_boxes: see forward
        :return:
        '''

        # Avoids modification to be reflected to the original tensor
        bounding_boxes = bounding_boxes.clone()

        bounding_boxes_dimensions = bounding_boxes[..., 2:] - bounding_boxes[..., :2]
        bounding_boxes[..., 0] -= bounding_boxes_dimensions[..., 0] * self.expansion_factor_cols
        bounding_boxes[..., 2] += bounding_boxes_dimensions[..., 0] * self.expansion_factor_cols
        bounding_boxes[..., 1] -= bounding_boxes_dimensions[..., 1] * self.expansion_factor_rows
        # Do not expand the bounding boxes to the bottom

        bounding_boxes = torch.clamp(bounding_boxes, min=0.0, max=1.0)
        return bounding_boxes

    def forward(self, observations: torch.Tensor, bounding_boxes: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                global_frame_indexes: torch.Tensor, video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor) -> torch.Tensor:
        '''
        Obtains the translations o2w of each scene object represented in the observations

        :param observations: (..., cameras_count, 3, height, width) observations
        :param bounding_boxes: (..., cameras_count, 4) tensor with left, top, right, bottom bounding box coordinates in [0, 1]
        :param camera_rotations: (..., cameras_count, 3) tensor with camera rotations
        :param camera_translations: (..., cameras_count, 3) tensor with camera translations
        :param global_frame_indexes: (...) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (...) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset

        :return: (..., style_features_count) tensor with style encoding for each position
                 (..., deformation_features_count) tensor with deformation encoding for each position
                 (..., 1, 1, features_height, features_width) tensor with attention map. Refers only to the first camera
                 (..., 1, 3, crop_height, crop_width) tensor with cropped image. Refers only to the first camera
        '''

        observations = observations[..., :1, :, :, :]
        bounding_boxes = bounding_boxes[..., :1, :]
        camera_rotations = camera_rotations[..., :1, :]
        camera_translations = camera_translations[..., :1, :]

        # Makes sure the bounding boxes contain the whole object
        bounding_boxes = self.expand_bounding_boxes(bounding_boxes)

        if random.randint(0, 100) == 0:
            print("Warning: using only the first camera for extracting object style")

        observations_height = observations.size(-2)
        observations_width = observations.size(-1)

        # Avoids modifications from propagating
        bounding_boxes = bounding_boxes.clone()
        # Denormalizes the bounding boxes
        bounding_boxes[..., 0] *= observations_width
        bounding_boxes[..., 2] *= observations_width
        bounding_boxes[..., 1] *= observations_height
        bounding_boxes[..., 3] *= observations_height

        # Removes the dimensions before cameras_count
        flat_observations, initial_observations_dimensions = TensorFolder.flatten(observations, -3)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -1)
        flat_rotations, _ = TensorFolder.flatten(camera_rotations, -1)
        flat_translations, _ = TensorFolder.flatten(camera_translations, -1)

        # Since multiple bounding boxes may be requested for each image, we must specify which batch index each box refers to
        batch_indexes = torch.arange(0, flat_bounding_boxes.size(0), device=flat_bounding_boxes.device).unsqueeze(-1)
        flat_bounding_boxes = torch.cat([batch_indexes, flat_bounding_boxes], dim=-1)
        flat_cropped_observations = torchvision.ops.roi_pool(flat_observations, flat_bounding_boxes, self.input_size)
        cropped_observations = TensorFolder.fold(flat_cropped_observations, initial_observations_dimensions)

        cameras_count = initial_observations_dimensions[-1]

        # Adds the spatial dimensions to allow concatenation with cropped observation
        dimensions_count = len(flat_rotations.size())
        repeat_size = [1] * dimensions_count + [self.input_size[0], self.input_size[1]]
        flat_rotations = flat_rotations.unsqueeze(-1).unsqueeze(-1).repeat(repeat_size)
        flat_translations = flat_translations.unsqueeze(-1).unsqueeze(-1).repeat(repeat_size)
        # Concatenates the cropped observations with rotations and translations
        flat_inputs = torch.cat([flat_cropped_observations, flat_rotations, flat_translations], dim=-3)

        # Forwards through the first convolution
        x = self.conv1(flat_inputs)
        x = F.avg_pool2d(x, 2)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)

        # Forwards through the initial features extractor common to each camera
        initial_output = self.initial_backbone(x)
        # Separates features from attention and applies activations
        attention = initial_output[:, -1:]
        common_features = initial_output[:, :-1]
        attention = torch.sigmoid(attention)
        common_features = F.leaky_relu(common_features, 0.2)
        folded_attention = TensorFolder.fold(attention, initial_observations_dimensions)

        # Applies the attention
        common_features = common_features * attention

        # Extracts back the camera dimensions and average features over the cameras
        common_features = TensorFolder.fold(common_features, [TensorFolder.prod(initial_observations_dimensions[:-1]), cameras_count])
        common_features = common_features.sum(dim=1) / cameras_count

        # Forwards the averaged features on the common backbone
        x = self.final_backbone(common_features)
        flat_pooled_output = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        flat_style = self.style_head(flat_pooled_output)
        flat_deformation = self.deformation_head(flat_pooled_output)

        # Reintroduces the initial dimensions apart from the camera
        style = TensorFolder.fold(flat_style, initial_observations_dimensions[:-1])
        deformation = TensorFolder.fold(flat_deformation, initial_observations_dimensions[:-1])

        return style, deformation, folded_attention, cropped_observations


def model(config, model_config):
    '''
    Istantiates a style encoder with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return ObjectEncoderV4(config, model_config)
