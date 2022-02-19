import math
import random
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.layers.residual_block import ResidualBlock
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_folder import TensorFolder


class ObjectParametersEncoderV4(nn.Module):
    '''
    Computes an object rotation and translation parameters
    Assumes the object lies on the xz plane to compute translation
    Similar to V3 but uses a sin/cos encoding for the output rotation instead of direct rotation prediction
    Assumes rotation happens only around the y axis
    '''

    def __init__(self, config: Dict, model_config: Dict):
        '''
        Initializes the model representing the style of an object

        :param config: the configuration file
        :param model_config: the configuration for the specific object
        '''
        super(ObjectParametersEncoderV4, self).__init__()

        self.config = config

        self.input_size = model_config["input_size"]
        # Distance from the edge of the bounding box to its center
        self.edge_to_center_distance = model_config["edge_to_center_distance"]

        self.expansion_factor_rows = 0.0
        self.expansion_factor_cols = 0.0
        if "expansion_factor" in model_config:
            self.expansion_factor_rows = model_config["expansion_factor"]["rows"]
            self.expansion_factor_cols = model_config["expansion_factor"]["cols"]

        # TODO initialize network so that outputs are close to 0

        # Takes as input rgb channels + camera rotations and camera translations
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Backbone before feature merging.
        self.initial_backbone = nn.Sequential(ResidualBlock(64, 64, downsample_factor=2),
                                              ResidualBlock(64, 64, downsample_factor=1))  # res / 4

        # Backbone after feature merging
        final_backbone_blocks = [
            ResidualBlock(64, 128, downsample_factor=2),  # res / 8
            ResidualBlock(128, 128, downsample_factor=1),  # res / 8
            ResidualBlock(128, 256, downsample_factor=2),  # res / 16
            ResidualBlock(256, 256, downsample_factor=1),  # res / 16
            ResidualBlock(256, 512, downsample_factor=2),  # res / 32
            ResidualBlock(512, 512, downsample_factor=1),  # res / 32
        ]
        self.final_backbone = nn.Sequential(*final_backbone_blocks)

        # Outputs a direction vector representing the rotation along the axis
        self.rotation_head = nn.Linear(512, 2)

        # Initializes the weight so that initial output is close to 0
        self.init_weights()

    def init_weights(self):
        '''
        Initializes the weights of the newtork so that the last layer initially outputs values close to 0
        :return:
        '''

        # Zero initialization so that no rotations are predicted initially
        torch.nn.init.uniform_(self.rotation_head.weight, a=-1e-5, b=1e-5)
        self.rotation_head.bias.data *= 0.0

    def expand_bounding_boxes(self, bounding_boxes: torch.Tensor):
        '''
        Expands the bounding boxes to ensure the object is not cut. Original bounding boxes are not altered
        :param bounding_boxes: see compute_rotations
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

    def compute_rotations(self, observations: torch.Tensor, camera_rotations: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Obtains the rotations o2w of each scene object represented in the observations.
        Assumes rotation is possible only around the y axis

        :param observations: (..., cameras_count, 3, height, width) observations
        :param camera_rotations: (..., cameras_count, 3) tensor with camera rotations
        :param focals: (..., cameras_count) tensor with focal length associated to each camera
        :param bounding_boxes: (..., cameras_count, 4) tensor with normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., cameras_count) tensor with True is the corresponding dynamic object instance is present in the scene

        :return: (..., 3) tensor with rotation parameters from object to world for each object
                 (..., 3) tensor with translation parameters from object to world for each object
        '''

        observations = observations[..., :1, :, :, :]
        bounding_boxes = bounding_boxes[..., :1, :]
        # Removes the camera dimensions
        bounding_boxes_validity = bounding_boxes_validity[..., 0]
        camera_rotations = camera_rotations[..., 0, :]

        # Expand the bounding boxes to make sure no part of the object is cut
        bounding_boxes = self.expand_bounding_boxes(bounding_boxes)

        if random.randint(0, 100) == 0:
            print("Warning: using only the first camera for extracting object parameters")

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

        # Since multiple bounding boxes may be requested for each image, we must specify which batch index each box refers to
        batch_indexes = torch.arange(0, flat_bounding_boxes.size(0), device=flat_bounding_boxes.device).unsqueeze(-1)
        flat_bounding_boxes = torch.cat([batch_indexes, flat_bounding_boxes], dim=-1)
        flat_cropped_observations = torchvision.ops.roi_pool(flat_observations, flat_bounding_boxes, self.input_size)

        cameras_count = initial_observations_dimensions[-1]

        # Concatenates the cropped observations with rotations and translations
        flat_inputs = flat_cropped_observations

        # Forwards through the first convolution
        x = self.conv1(flat_inputs)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)

        # Forwards through the initial features extractor common to each camera
        initial_output = self.initial_backbone(x)

        common_features = initial_output

        # Extracts back the camera dimensions and average features over the cameras
        common_features = TensorFolder.fold(common_features, [TensorFolder.prod(initial_observations_dimensions[:-1]), cameras_count])
        common_features = common_features.sum(dim=1) / cameras_count

        # Forwards the averaged features on the common backbone
        x = self.final_backbone(common_features)
        flat_pooled_output = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)

        flat_c2o_rotation_offset_vector = self.rotation_head(flat_pooled_output)
        # Applies the activation function. Scales to ensure 0, 90, 180 and 270 rotations do not fall in the saturation range of the tanh function
        flat_c2o_rotation_offset_vector = torch.tanh(flat_c2o_rotation_offset_vector) * 1.4

        flat_c2o_rotation_offset = torch.atan2(flat_c2o_rotation_offset_vector[..., 1], flat_c2o_rotation_offset_vector[..., 0])

        # Adds x and z dimensions. Rotation can be present only along the y axis
        flat_c2o_rotation_offset = flat_c2o_rotation_offset.unsqueeze(-1).repeat([1] * len(flat_c2o_rotation_offset.size()) + [3])
        flat_c2o_rotation_offset[..., [0, 2]] *= 0.0

        # Reintroduces the initial dimensions apart from the camera
        folded_c2o_rotation_offset = TensorFolder.fold(flat_c2o_rotation_offset, initial_observations_dimensions[:-1])
        # Adds the offsets. Considers only the y direction
        camera_rotations = camera_rotations.clone()
        camera_rotations[..., [0, 2]] = 0.0
        rotations_o2w = camera_rotations + folded_c2o_rotation_offset

        # No rotation is computed if the object is not visible
        rotations_o2w[bounding_boxes_validity == False, :] *= 0
        folded_c2o_rotation_offset[bounding_boxes_validity == False, :] *= 0

        return rotations_o2w, folded_c2o_rotation_offset

    def normalize_range(self, tensor: torch.Tensor, min: float, max: float):
        '''
        Brings all values in the tensor between min and max.
        Each value is updated in steps of (max - min) towards the interval
        :param tensor: (...) tensor to normalize
        :param min:
        :param max:
        :return:
        '''

        tensor = tensor.clone()

        delta = max - min
        while True:
            mask = tensor > max
            if mask.any():
                tensor[mask] -= delta
            else:
                break
        while True:
            mask = tensor < min
            if mask.any():
                tensor[mask] += delta
            else:
                break

        return tensor

    def compute_translations(self, observations: torch.Tensor, transformation_matrix_w2c: torch.Tensor, c2o_rotation_offset: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Obtains the translations o2w of each scene object represented in the observations.
        Applies correction to account for non-tight bounding boxes

        :param observations: (..., cameras_count, 3, height, width) observations
        :param transformation_matrix_w2c: (..., cameras_count, 4, 4) tensor with transformation
                                          from world to camera coordinates
        :param c2o_rotation_offset: (..., 3, dynamic_objects_count) tensor with rotation offsets from camera to object
        :param focals: (..., cameras_count) tensor with focal length associated to each camera
        :param bounding_boxes: (..., cameras_count, 4, dynamic_objects_count) tensor with normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., cameras_count, dynamic_objects_count) tensor with True is the corresponding dynamic object instance is present in the scene

        :return: (..., 3, objects_count) tensor with rotation parameters from object to world for each object
                 (..., 3, objects_count) tensor with translation parameters from object to world for each object
        '''

        eps = 1e-6

        observations = observations[..., :1, :, :, :]
        transformation_matrix_w2c = transformation_matrix_w2c[..., :1, :, :]
        focals = focals[..., :1]
        bounding_boxes = bounding_boxes[..., :1, :, :]
        bounding_boxes_validity = bounding_boxes_validity[..., :1, :]
        if random.randint(0, 100) == 0:
            print("Warning: using only the first camera for extracting object parameters")

        # Retrieves dimensions of the observations
        observations_height = observations.size(-2)
        observations_width = observations.size(-1)

        # Expresses the bounding boxes in pixels
        bounding_boxes = bounding_boxes.clone()
        bounding_boxes[..., [0, 2], :] *= observations_width
        bounding_boxes[..., [1, 3], :] *= observations_height

        # Removes the dimensions before cameras_count
        flat_observations, initial_observations_dimensions = TensorFolder.flatten(observations, -4)
        flat_transformation_matrix_w2c, _ = TensorFolder.flatten(transformation_matrix_w2c, -3)
        flat_focals, _ = TensorFolder.flatten(focals, -1)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -3)
        flat_bounding_boxes_validity, _ = TensorFolder.flatten(bounding_boxes_validity, -2)
        flat_c2o_rotation_offset, _ = TensorFolder.flatten(c2o_rotation_offset, -2)

        # Computes the transformation matrix from camera to world
        flat_transformation_matrix_c2w = flat_transformation_matrix_w2c[:, 0].inverse()

        objects_count = flat_bounding_boxes.size(-1)

        all_object_translations = []
        for object_idx in range(objects_count):
            elements_count = flat_bounding_boxes.size(0)
            current_bounding_boxes = flat_bounding_boxes[..., 0, :, object_idx]

            # Computes the validity of an object. An object is valid if it has been detected in at least one camera
            current_bounding_boxes_validity = flat_bounding_boxes_validity[..., 0, object_idx]

            ray_origins = torch.zeros((elements_count, 3), dtype=torch.float, device=observations.device)
            # Center of the bounding box minus the image center
            feet_image_x_coordinates = ((current_bounding_boxes[:, 0] + current_bounding_boxes[:, 2]) / 2) - (observations_width / 2)
            # Bottom of the boudning box minus the image center. - accounts for y axis going up in camera coordinates
            feet_image_y_coordinates = -(current_bounding_boxes[:, 3] - (observations_height / 2))
            # Position of the camera plane. - accounts for the camera facing the -z direction
            feet_image_z_coordinates = -flat_focals[:, 0]

            ray_directions = torch.stack([feet_image_x_coordinates, feet_image_y_coordinates, feet_image_z_coordinates], dim=-1)

            # Transforms origins and directions in camera coordinates
            ray_origins = RayHelper.transform_points(ray_origins, flat_transformation_matrix_c2w, rotation=True, translation=True)
            ray_directions = RayHelper.transform_points(ray_directions, flat_transformation_matrix_c2w, rotation=True, translation=False)

            # Number of times the direction vector must be repeated to touch the ground (Y=0)
            n = -ray_origins[:, 1] / (ray_directions[:, 1] + eps)
            # Repeats the vector n times to find the position on the ground
            current_object_translations = ray_origins + n.unsqueeze(-1) * ray_directions
            current_object_translations[..., 1] *= 0.0  # Forces y to be 0

            # Computes the vector on the xz plane that looks in the camera direction
            # Normalizes it such that is has norm 1
            ray_directions_correction_vector = ray_directions.clone()
            ray_directions_correction_vector[:, 1] = 0
            ray_directions_correction_vector = ray_directions_correction_vector / torch.sqrt(ray_directions_correction_vector.pow(2).sum(-1, keepdims=True))

            # Gets the current rotation component and normalizes the rotation in the range [-pi/4 and pi/4]
            current_c2o_y_rotation_offset = flat_c2o_rotation_offset[:, 1, object_idx]
            current_c2o_y_rotation_offset = self.normalize_range(current_c2o_y_rotation_offset, min=-(math.pi / 4), max=+(math.pi / 4))

            # The distance from the box center to the point on the box edge where the camera ray intersects the box
            sloped_edge_center_distance = self.edge_to_center_distance / torch.cos(current_c2o_y_rotation_offset)
            # Correction to apply to the translations to account for the fact that bounding boxes may not be tight
            translation_correction_factor = ray_directions_correction_vector * sloped_edge_center_distance.unsqueeze(-1)

            current_object_translations = current_object_translations + translation_correction_factor

            # Parameters for objects not present in the scene are 0
            current_object_translations[current_bounding_boxes_validity == False, :] *= 0.0

            all_object_translations.append(current_object_translations)

        all_object_translations = torch.stack(all_object_translations, dim=-1)

        # Reintroduces the original flattened dimensions
        all_object_translations = TensorFolder.fold(all_object_translations, initial_observations_dimensions)

        return all_object_translations

    def forward(self, observations: torch.Tensor, transformation_matrix_w2c: torch.Tensor, camera_rotations: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, apply_ranges=True) -> Tuple[torch.Tensor]:
        '''
        Obtains the rotations o2w and translations o2w of each scene object represented in the observations

        :param observations: (..., cameras_count, 3, height, width) observations
        :param transformation_matrix_w2c: (..., cameras_count, 4, 4) tensor with transformation
                                          from world to camera coordinates
        :param camera_rotations: (..., cameras_count, 3) tensor with camera rotations
        :param focals: (..., cameras_count) tensor with focal length associated to each camera
        :param bounding_boxes: (..., cameras_count, 4, dynamic_objects_count) tensor with normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., cameras_count, dynamic_objects_count) tensor with True is the corresponding dynamic object instance is present in the scene
        :param apply_ranges: unused. For compatibility with encoders using ranges

        :return: (..., 3, objects_count) tensor with rotation parameters from object to world for each object
                 (..., 3, objects_count) tensor with translation parameters from object to world for each object
        '''

        # Computes the rotations for each object
        rotations_o2w = []
        c2o_rotation_offset = []
        objects_count = bounding_boxes.size(-1)
        for object_idx in range(objects_count):
            current_rotations_o2w, current_c2o_rotation_offset = self.compute_rotations(observations, camera_rotations, bounding_boxes[..., object_idx], bounding_boxes_validity[..., object_idx])
            rotations_o2w.append(current_rotations_o2w)
            c2o_rotation_offset.append(current_c2o_rotation_offset)

        # Transforms the list into a tensor
        rotations_o2w = torch.stack(rotations_o2w, dim=-1)
        c2o_rotation_offset = torch.stack(c2o_rotation_offset, dim=-1)

        # Computes the translations for each object
        translations_o2w = self.compute_translations(observations, transformation_matrix_w2c, c2o_rotation_offset, focals, bounding_boxes, bounding_boxes_validity)

        return rotations_o2w, translations_o2w

def model(config, model_config):
    '''
    Istantiates a style encoder with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return ObjectParametersEncoderV4(config, model_config)
