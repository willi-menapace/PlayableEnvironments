import importlib
from typing import Dict, Tuple

import torch
import torch.nn as nn

from utils.lib_3d.bounding_box import BoundingBox
from utils.tensor_broadcaster import TensorBroadcaster
from utils.tensor_folder import TensorFolder


class RayBendingStyleNerfModel(nn.Module):
    '''
    A NERF rendering model with ray bending to model deformations and adain to model style
    '''

    def __init__(self, config: Dict, model_config: Dict):
        super(RayBendingStyleNerfModel, self).__init__()

        self.config = config
        self.model_config = model_config

        self.empty_space_alpha = model_config["empty_space_alpha"]
        self.bounding_box = BoundingBox(model_config["bounding_box"])

        # Number of features to use as style and as deformations
        self.style_features = model_config["style_features"]
        self.deformation_features = model_config["deformation_features"]

        self.nerf_model_config = self.model_config["nerf_model"]
        self.ray_bender_model_config = self.model_config["ray_bender_model"]

        self.transfer_attributes_to_submodels_configs()

        # Builds the nerf model and the ray bender
        self.nerf_model = getattr(importlib.import_module(self.nerf_model_config["architecture"]), 'model')(config, self.nerf_model_config)
        self.ray_bender = getattr(importlib.import_module(self.ray_bender_model_config["architecture"]), 'model')(config, self.ray_bender_model_config)

    def transfer_attributes_to_submodels_configs(self):
        '''
        Transfers common configuration attributes to the configuration files of submodels
        :return:
        '''

        # Transfer attributes to both the nerf model and the ray bender
        for current_config in [self.nerf_model_config, self.ray_bender_model_config]:
            current_config["bounding_box"] = self.model_config["bounding_box"]
            current_config["empty_space_alpha"] = self.model_config["empty_space_alpha"]
            current_config["style_features"] = self.model_config["style_features"]
            current_config["deformation_features"] = self.model_config["deformation_features"]

    def set_step(self, current_step: int):
        '''
        Sets the current step to the specified value
        :param current_step:
        :return:
        '''

        # Sets the step in the deformation model
        self.ray_bender.set_step(current_step)

    def compute_bounding_box_filtering_mask(self, flat_ray_positions: torch.Tensor) -> torch.Tensor:
        '''
        Checks which positions fall in the bounding box
        :param flat_ray_positions: (elements_count, 3) tensor with positions to check
        :return: (elements_count) boolean tensor with True to the elements that fall in the bounding box
        '''

        # Dimensions of the bounding box
        bounding_box_dimensions = self.bounding_box.dimensions

        all_dimensions_bounds_respected = None
        for dimension_idx in range(3):
            # Checks if the current dimensions respects the bounding box
            lower_bound_respected = flat_ray_positions[..., dimension_idx] >= bounding_box_dimensions[dimension_idx, 0]
            upper_bound_respected = flat_ray_positions[..., dimension_idx] <= bounding_box_dimensions[dimension_idx, 1]
            current_dimension_bounds_respected = torch.logical_and(lower_bound_respected, upper_bound_respected)

            # Merges the results with the ones for the other dimensions
            if all_dimensions_bounds_respected is None:
                all_dimensions_bounds_respected = current_dimension_bounds_respected
            else:
                all_dimensions_bounds_respected = torch.logical_and(all_dimensions_bounds_respected, current_dimension_bounds_respected)

        return all_dimensions_bounds_respected

    def compute_network_pass(self, ray_positions: torch.Tensor, ray_origins: torch.Tensor, ray_directions: torch.Tensor, style: torch.Tensor, deformation: torch.Tensor, video_indexes: torch.Tensor=None, canonical_pose: bool=False):
        '''
        Computes network outputs for the given position and directions pairs

        :param ray_positions: (elements_count, 3) tensor with ray positions
        :param ray_origins: (elements_count, 3) tensor with ray origins
        :param ray_directions: (elements_count, 3) tensor with ray directions
        :param style: (elements_count, style_features_count) tensor with style for each position
        :param deformation: (elements_count, deformation_features_count) tensor with deformation encoding for each position
        :param video_indexes: (elements_count) tensor of integers representing indexes of each video in the dataset. Indexes are inferred if None
        :param canonical_pose: if True renders the object in the canonical pose
        :return: (elements_count, output_features_count) tensor with features
                 (elements_count) tensor with alpha value
                 (elements_count, 3) tensor with style associated with each position
                 dictionary with extra output values in the form
                     (elements_count, dim)
        '''

        # Obtains the ray displacement for each position and bends the rays
        ray_displacements = self.ray_bender(ray_positions, deformation, video_indexes=video_indexes)
        # If the canonical pose is required, zero the displacements
        if canonical_pose:
            ray_displacements = ray_displacements * 0.0

        bent_ray_positions = ray_positions + ray_displacements

        # Evaluates the model along the bent rays applying the style
        features_output, alpha_output, extra_outputs = self.nerf_model(bent_ray_positions, ray_origins, ray_directions, style, video_indexes=video_indexes)

        return features_output, alpha_output, ray_displacements, extra_outputs

    def expand_latent_code(self, code: torch.Tensor, reference_tensor: torch.Tensor):
        '''
        Broadcasts the code tensor to the size of the reference tensor
        :param code:
        :param reference_tensor:
        :return:
        '''

        code_repeats = list(code.size())
        reference_size = list(reference_tensor.size())
        for idx in range(len(code_repeats)):
            if code_repeats[idx] == 1:
                code_repeats[idx] = reference_size[idx]
            else:
                code_repeats[idx] = 1
        code = code.repeat(code_repeats)

        return code

    def forward(self, ray_positions: torch.Tensor, ray_origins: torch.Tensor, ray_directions: torch.Tensor, style: torch.Tensor, deformation: torch.Tensor, video_indexes: torch.Tensor=None, canonical_pose: bool=False) -> Tuple[torch.Tensor]:
        '''

        :param ray_positions: (..., positions_count, 3) tensor with ray positions
        :param ray_origins: (..., 3) tensor with ray_origins
        :param ray_directions: (..., 3) tensor with ray directions
        :param style: (..., style_features_count) tensor with style for each position
        :param deformation: (..., deformation_features_count) tensor with deformation encodings for each position
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset. Indexes are inferred if None
        :param canonical_pose: if True renders the object in the canonical pose
        :return: (..., positions_count, output_features_count) tensor with output features
                 (..., positions_count) tensor with output alphas
                 (..., positions_count, 3) tensor with ray displacement associated to each position
                 dictionary with extra output values in the form
                     (..., positions_count, dim)
        '''

        # Makes ray directions the same size as ray positions
        positions_count = ray_positions.size(-2)
        ray_origins = TensorBroadcaster.add_dimension(ray_origins, positions_count, dim=-2)
        ray_directions = TensorBroadcaster.add_dimension(ray_directions, positions_count, dim=-2)

        # Makes ray positions, style and deformations of the same size
        style = style.unsqueeze(-2)
        style = self.expand_latent_code(style, ray_positions)
        deformation = deformation.unsqueeze(-2)
        deformation = self.expand_latent_code(deformation, ray_positions)

        # Flattens the inputs
        flat_ray_positions, flattened_ray_position_dimensions = TensorFolder.flatten(ray_positions, -1)
        flat_ray_origins, _ = TensorFolder.flatten(ray_origins, -1)
        flat_ray_directions, _ = TensorFolder.flatten(ray_directions, -1)
        flat_style, _ = TensorFolder.flatten(style, -1)
        flat_deformation, _ = TensorFolder.flatten(deformation, -1)

        # Creates empty tensors for the output
        device = ray_positions.device
        first_dimension = flat_ray_positions.size(0)
        flat_output_features = torch.zeros((first_dimension, self.nerf_model.output_features), dtype=torch.float32, device=device)
        flat_output_alphas = torch.ones((first_dimension), dtype=torch.float32, device=device) * self.empty_space_alpha
        flat_output_ray_displacements = torch.zeros((first_dimension, 3), dtype=torch.float32, device=device)

        # Excludes from the computation the points which are outside the bounding box of the model
        bounding_box_mask = self.compute_bounding_box_filtering_mask(flat_ray_positions)

        filtered_ray_positions = flat_ray_positions[bounding_box_mask, :]
        filtered_ray_origins = flat_ray_origins[bounding_box_mask, :]
        filtered_ray_directions = flat_ray_directions[bounding_box_mask, :]
        filtered_style = flat_style[bounding_box_mask, :]
        filtered_deformation = flat_deformation[bounding_box_mask, :]

        # Prepares and filters the video indexes if they are present
        # Uses the same processing as style and deformation
        filtered_video_indexes = None
        if video_indexes is not None:
            video_indexes = video_indexes.unsqueeze(-1)
            video_indexes = self.expand_latent_code(video_indexes, ray_positions)
            flat_video_indexes, _ = TensorFolder.flatten(video_indexes, 0)
            filtered_video_indexes = flat_video_indexes[bounding_box_mask]

        # Passes the inputs through the network
        filtered_output_features, filtered_output_alpha, filtered_output_ray_displacements, filtered_extra_outputs = self.compute_network_pass(filtered_ray_positions, filtered_ray_origins, filtered_ray_directions, filtered_style, filtered_deformation, filtered_video_indexes, canonical_pose=canonical_pose)

        flat_output_features[bounding_box_mask, :] = filtered_output_features
        flat_output_alphas[bounding_box_mask] = filtered_output_alpha
        flat_output_ray_displacements[bounding_box_mask, :] = filtered_output_ray_displacements

        extra_outputs = {}
        for current_key, current_value in filtered_extra_outputs.items():
            unfiltered_size = list(current_value.size())
            unfiltered_size[0] = first_dimension
            current_flat_output = torch.zeros(unfiltered_size, dtype=current_value.dtype, device=current_value.device)
            # Expands the returned value to the initial dimensions
            current_flat_output[bounding_box_mask, :] = current_value
            current_output = TensorFolder.fold(current_flat_output, flattened_ray_position_dimensions)
            extra_outputs[current_key] = current_output

        # Folds the results to the original shape
        output_features = TensorFolder.fold(flat_output_features, flattened_ray_position_dimensions)
        output_alphas = TensorFolder.fold(flat_output_alphas, flattened_ray_position_dimensions)
        output_ray_displacements = TensorFolder.fold(flat_output_ray_displacements, flattened_ray_position_dimensions)

        return output_features, output_alphas, output_ray_displacements, extra_outputs


def model(config, model_config):
    '''
    Instantiates a nerf model with the given parameters

    :param config:
    :param model_config:
    :return:
    '''
    return RayBendingStyleNerfModel(config, model_config)
