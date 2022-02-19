from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.adain import AffineTransformAdaIn
from model.layers.adain_sequential import AdaInSequential
from model.positional_encoder import PositionalEncoder
from utils.lib_3d.bounding_box import BoundingBox
from utils.tensor_folder import TensorFolder


class AdaInStyleNerfModel(nn.Module):
    '''
    A NERF rendering model with style encodings through AdaIn layers
    '''

    def __init__(self, config: Dict, model_config: Dict):
        super(AdaInStyleNerfModel, self).__init__()

        self.config = config
        self.model_config = model_config

        self.layers_width = model_config["layers_width"]
        self.backbone_layers_count = model_config["backbone_layers_count"]
        self.output_features = model_config["output_features"]
        self.skip_layer_idx = model_config["skip_layer_idx"]
        self.style_features = model_config["style_features"]

        self.empty_space_alpha = model_config["empty_space_alpha"]

        if self.skip_layer_idx >= self.backbone_layers_count:
            raise Exception("Skip layer must refer to a valid backbone layer idx")

        # Builds the positional encoders
        self.position_encoder = PositionalEncoder(3, model_config["position_encoder"]["octaves"], model_config["position_encoder"]["append_original"])

        self.bounding_box = BoundingBox(model_config["bounding_box"])

        # Instantiates the backbone layers
        self.backbone_layers = nn.ModuleList()
        current_input_size = self.position_encoder.get_encoding_size()
        for layer_idx in range(self.backbone_layers_count):
            # Accounts for the skip connection
            if layer_idx == self.skip_layer_idx:
                current_input_size += self.position_encoder.get_encoding_size()

            self.backbone_layers.append(nn.Linear(current_input_size, self.layers_width))
            current_input_size = self.layers_width

        # The head outputting the alpha value
        self.alpha_head = nn.Linear(self.layers_width, 1)
        # The first part of the head for the features
        self.features_head = self.get_features_head()

    def get_features_head(self):
        '''
        Returns the module to use as the feature head
        :return:
        '''

        return AdaInSequential(
            nn.Linear(self.layers_width, self.layers_width, bias=False),
            self.get_style_embedding_layer_class()(self.layers_width, self.style_features),
            nn.ReLU(),
            nn.Linear(self.layers_width, self.layers_width // 2, bias=False),
            self.get_style_embedding_layer_class()(self.layers_width // 2, self.style_features),
            nn.ReLU(),
            nn.Linear(self.layers_width // 2, self.output_features)
        )

    def get_style_embedding_layer_class(self):
        '''
        Gets the class to use for style embedding
        :return:
        '''

        return AffineTransformAdaIn

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

    def compute_network_pass(self, ray_positions: torch.Tensor, ray_origins: torch.Tensor, ray_directions: torch.Tensor, style: torch.Tensor):
        '''
        Computes network outputs for the given position and directions pairs

        :param ray_positions: (elements_count, 3) tensor with ray positions
        :param ray_origins: (elements_count, 3) tensor with ray origins
        :param ray_directions: (elements_count, 3) tensor with ray directions
        :param style: (elements_count, style_features_count) tensor with style for each position
        :return: (elements_count, output_features_count) tensor with features
                 (elements_count, 1) tensor with alpha value
                 dictionary with extra output values
        '''

        # Normalizes the input positions
        bounding_box_size = self.bounding_box.get_size()
        normalized_ray_positions = ray_positions / bounding_box_size
        normalized_ray_origins = ray_origins / bounding_box_size

        encoded_ray_positions = self.position_encoder(normalized_ray_positions)

        # passes positions through the backbone
        current_output = encoded_ray_positions
        for layer_idx, current_layer in enumerate(self.backbone_layers):
            # Accounts for the skip connection
            if layer_idx == self.skip_layer_idx:
                current_output = torch.cat([current_output, encoded_ray_positions], dim=-1)

            # Applies the layer and the activation
            current_output = current_layer(current_output)
            current_output = F.relu(current_output)

        # Obtains the alpha
        alpha_output = self.alpha_head(current_output)

        # Continues the computation of the features
        features_output = self.features_head(current_output, style)

        extra_outputs = {}

        return features_output, alpha_output, extra_outputs

    def forward(self, ray_positions: torch.Tensor, ray_origins: torch.Tensor, ray_directions: torch.Tensor, style: torch.Tensor, video_indexes: torch.Tensor=None) -> Tuple[torch.Tensor]:
        '''

        :param ray_positions: (..., 3) tensor with ray positions
        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., 3) tensor with ray directions
        :param style: (..., style_features_count) tensor with style for each position
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset.
        :return: (..., output_features_count) tensor with output features
                 (...) tensor with output alphas
                 dictionary with extra output values
        '''

        # Flattens the inputs
        flat_ray_positions, flattened_ray_position_dimensions = TensorFolder.flatten(ray_positions, -1)
        flat_ray_origins, _ = TensorFolder.flatten(ray_origins, -1)
        flat_ray_directions, _ = TensorFolder.flatten(ray_directions, -1)
        flat_style, _ = TensorFolder.flatten(style, -1)

        # Creates empty tensors for the output
        device = self.alpha_head.weight.data.device
        first_dimension = flat_ray_positions.size(0)
        flat_output_features = torch.zeros((first_dimension, self.output_features), dtype=torch.float32, device=device)
        flat_output_alphas = torch.ones((first_dimension, 1), dtype=torch.float32, device=device) * self.empty_space_alpha

        # Excludes from the computation the points which are outside the bounding box of the model
        bounding_box_mask = self.compute_bounding_box_filtering_mask(flat_ray_positions)

        filtered_ray_positions = flat_ray_positions[bounding_box_mask, :]
        filtered_ray_origins = flat_ray_origins[bounding_box_mask, :]
        filtered_ray_directions = flat_ray_directions[bounding_box_mask, :]
        filtered_style = flat_style[bounding_box_mask, :]

        # Passes the inputs through the network
        filtered_output_features, filtered_output_alpha, filtered_extra_outputs = self.compute_network_pass(filtered_ray_positions, filtered_ray_origins, filtered_ray_directions, filtered_style)

        flat_output_features[bounding_box_mask, :] = filtered_output_features
        flat_output_alphas[bounding_box_mask, :] = filtered_output_alpha
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

        return output_features, output_alphas.squeeze(-1), extra_outputs

def model(config, model_config):
    '''
    Istantiates a nerf model with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return AdaInStyleNerfModel(config, model_config)

