from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.annealable_positional_encoder import AnnealablePositionalEncoder
from utils.lib_3d.bounding_box import BoundingBox
from utils.tensor_folder import TensorFolder


class PositionalRayBender(nn.Module):
    '''
    A model for bending rays
    '''

    def __init__(self, config: Dict, model_config: Dict):
        super(PositionalRayBender, self).__init__()

        self.config = config
        self.model_config = model_config

        self.layers_width = model_config["layers_width"]
        self.layers_count = model_config["layers_count"]
        self.skip_layer_idx = model_config["skip_layer_idx"]
        self.deformation_features = model_config["deformation_features"]

        # Builds the annealable positional encoder
        octaves_count = model_config["position_encoder"]["octaves"]
        append_original = model_config["position_encoder"]["append_original"]
        positional_annealing_steps = model_config["position_encoder"]["num_steps"]
        self.positional_encoder = AnnealablePositionalEncoder(3, octaves_count, append_original, positional_annealing_steps)

        self.bounding_box = BoundingBox(model_config["bounding_box"])

        # Do not use bias in the last layer. Displacements should have zero mean
        self.last_layer_bias = False

        # Instantiates the backbone layers
        self.backbone_layers = nn.ModuleList()
        current_input_size = self.positional_encoder.get_encoding_size() + self.deformation_features  # Inject also deformations
        for layer_idx in range(self.layers_count):

            # Accounts for the skip connection
            if layer_idx == self.skip_layer_idx:
                current_input_size += self.positional_encoder.get_encoding_size() + self.deformation_features

            self.backbone_layers.append(nn.Linear(current_input_size, self.layers_width))
            current_input_size = self.layers_width

        # The head outputting the displacements value
        self.output_head = nn.Linear(self.layers_width, 3, bias=self.last_layer_bias)

        # Initializes the weights
        self.init_weights()

    def set_step(self, current_step: int):
        '''
        Sets the current step to the specified value
        :param current_step:
        :return:
        '''

        self.positional_encoder.set_step(current_step)

    def init_weights(self):
        '''
        Initializes the weights of the newtork
        :return:
        '''

        for current_layer in self.backbone_layers:
            torch.nn.init.kaiming_uniform_(current_layer.weight, a=0, mode="fan_in", nonlinearity="relu")
            torch.nn.init.zeros_(current_layer.bias)

        # Zero initialization so that no displacements are present initially
        torch.nn.init.uniform_(self.backbone_layers[-1].weight, a=-1e-5, b=1e-5)
        if self.last_layer_bias:
            self.backbone_layers[-1].bias.data *= 0.0

    def compute_network_pass(self, ray_positions, deformation):
        '''
        Computes network outputs for the given position and deformation

        :param ray_positions: (elements_count, 3) tensor with ray positions
        :param deformation: (elements_count, deformation_features_count) tensor with deformation encoding for each position
        :return: (elements_count, 3) tensor with position displacements
        '''

        # Normalizes the input positions
        bounding_box_size = self.bounding_box.get_size()
        normalized_ray_positions = ray_positions / bounding_box_size

        encoded_ray_positions = self.positional_encoder(normalized_ray_positions)

        # passes positions and deformation through the backbone
        current_output = torch.cat([encoded_ray_positions, deformation], dim=-1)
        for layer_idx, current_layer in enumerate(self.backbone_layers):

            # Accounts for the skip connection
            if layer_idx == self.skip_layer_idx:
                current_output = torch.cat([current_output, encoded_ray_positions, deformation], dim=-1)

            # Applies layer and activation
            current_output = current_layer(current_output)
            current_output = F.relu(current_output)

        # Obtains the alpha
        displacements = self.output_head(current_output)

        # Denormalizes the output displacements
        displacements = displacements * bounding_box_size

        return displacements

    def clamp_output(self, flat_ray_positions: torch.Tensor, flat_displacements: torch.Tensor) -> torch.Tensor:
        '''
        Clamps the displacements so that they fall within the bounding box

        :param flat_ray_positions: (elements_count, 3) tensor with positions
        :param flat_displacements: (elements_count, 3) tensor with displacements to clamp
        :return: (elements_count, 3) tensor with clamped displacements
        '''

        # Dimensions of the bounding box
        bounding_box_dimensions = self.bounding_box.dimensions

        # Computes upper and lower bounding box bounds
        lower_bounds = bounding_box_dimensions[..., 0].unsqueeze(0)
        upper_bounds = bounding_box_dimensions[..., 1].unsqueeze(0)

        # Computes maximum and minimum values allowed for each displacement
        min_displacement = lower_bounds - flat_ray_positions
        max_displacement = upper_bounds - flat_ray_positions

        # Clamps the displacements
        clamped_displacements = torch.maximum(flat_displacements, min_displacement)
        clamped_displacements = torch.minimum(clamped_displacements, max_displacement)

        return clamped_displacements

    def forward(self, ray_positions: torch.Tensor, deformation: torch.Tensor, video_indexes: torch.Tensor=None) -> Tuple[torch.Tensor]:
        '''

        :param ray_positions: (..., 3) tensor with ray positions
        :param deformation: (..., deformation_features_count) tensor with deformation encodings for each position
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset.
        :return: (..., 3) tensor with position displacements
        '''

        # Flattens the inputs
        flat_ray_positions, flattened_ray_position_dimensions = TensorFolder.flatten(ray_positions, -1)
        flat_deformation, _ = TensorFolder.flatten(deformation, -1)

        # Passes the inputs through the network
        flat_displacements = self.compute_network_pass(flat_ray_positions, flat_deformation)
        # Clamps the displacements so that the bent rays are inside the bounding box
        flat_displacements = self.clamp_output(flat_ray_positions, flat_displacements)

        # Folds the results to the original shape
        displacements = TensorFolder.fold(flat_displacements, flattened_ray_position_dimensions)

        return displacements


def model(config, model_config):
    '''
    Instantiates a nerf model with the given parameters

    :param config:
    :param model_config:
    :return:
    '''
    return PositionalRayBender(config, model_config)

