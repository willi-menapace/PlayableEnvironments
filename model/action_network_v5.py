from typing import List

import torch
import torch.nn as nn

from model.layers.masked_batch_norm import MaskedBatchNorm1d
from model.layers.masked_sequential import MaskedSequential
from model.layers.rotation_encoder import RotationEncoder
from utils.lib_3d.bounding_box import BoundingBox
from utils.tensor_folder import TensorFolder
from utils.tensor_splitter import TensorSplitter


class ActionNetworkV5(nn.Module):
    '''
    Model that infers the action associated to the transition between two states
    Makes use of an internal sin, cos encoding for representing rotations
    Like V4 but uses masked batch normalization
    '''

    def __init__(self, config, model_config):
        super(ActionNetworkV5, self).__init__()

        self.config = config

        # Computes the number of input features
        self.style_features = model_config["style_features"]
        self.deformation_features = model_config["deformation_features"]
        # Checks whether to use deformation to predict actions
        self.use_deformation = False
        if "use_deformation" in model_config:
            self.use_deformation = model_config["use_deformation"]

        # Bounding box used for input translation normalization
        self.bounding_box = BoundingBox(model_config["bounding_box"])

        self.features_count = [6, 3]  # Do not use style or deformations to predict actions. Rotations are encoded into (sin,cos)
        if self.use_deformation:
            self.features_count.append(self.deformation_features)
        self.input_features = sum(self.features_count)

        # Parameters for the mlp
        self.mlp_features = model_config["layers_width"]
        self.layers_count = model_config["layers_count"]

        # Dimensions of the actions
        self.actions_count = model_config["actions_count"]
        self.action_space_dimension = model_config["action_space_dimension"]

        # Builds the mlp backbone
        mlp_layers = []
        current_features_count = self.input_features
        for i in range(self.layers_count):
            mlp_layers.append(nn.Linear(current_features_count, self.mlp_features))
            mlp_layers.append(MaskedBatchNorm1d(self.mlp_features))
            mlp_layers.append(nn.ReLU())
            current_features_count = self.mlp_features
        self.mlp_backbone = MaskedSequential(*mlp_layers)

        # Linear layers for the prediction of the posterior parameters in the action space
        self.mean_fc = nn.Linear(self.mlp_features, self.action_space_dimension)
        self.log_variance_fc = nn.Linear(self.mlp_features, self.action_space_dimension)

        # Classifies the vector
        self.final_fc = nn.Linear(self.action_space_dimension, self.actions_count)

    def sample(self, mean: torch.Tensor, log_variance: torch.Tensor):
        '''
        Samples from the posterior distribution with given mean and log variance

        :param mean: (..., action_space_dimension) tensor with posterior mean
        :param log_variance: (..., action_space_dimension) tensor with posterior log variance
        :return: (..., action_space_dimension) tensor with points sampled from the posterior
        '''

        noise = torch.randn(mean.size(), dtype=torch.float32).cuda()
        sampled_points = noise * torch.exp(log_variance * 0.5) + mean

        return sampled_points

    def sample_variance(self, mean: torch.Tensor, variance: torch.Tensor):
        '''
        Samples from the posterior distribution with given mean and variance

        :param mean: (..., action_space_dimension) tensor with posterior mean
        :param variance: (..., action_space_dimension) tensor with posterior variance
        :return: (..., action_space_dimension) tensor with points sampled from the posterior
        '''

        noise = torch.randn(mean.size(), dtype=torch.float32).cuda()
        sampled_points = noise * torch.sqrt(variance) + mean

        return sampled_points

    def forward(self, *inputs: List[torch.Tensor]) -> torch.Tensor:
        '''
        Computes actions corresponding to the state transition from predecessor to successor state

        :param inputs: (bs, observations_count, 3) tensors with rotations around each axis
                       (bs, observations_count, 3) tensors with translations
                       (bs, observations_count, deformation_features) tensors with deformation latent codes if use_deformations is True
                       (bs, observations_count) tensor with boolean indicating whether the sequence is valid at a certain point
                       (bs, observations_count) tensor with true if the object is in scene

        :return: action_logits, action_directions_distribution, sampled_action_directions,
                 action_states_distribution, sampled_action_states
                 (bs, observations_count - 1, actions_count) tensor with logits of probabilities for each action
                 (bs, observations_count - 1, 2, action_space_dimension) tensor posterior mean and log variance for action directions
                 (bs, observations_count - 1, action_space_dimension) tensor with sampled action directions
                 (bs, observations_count, 2, action_space_dimension) tensor posterior mean and log variance for action states
                 (bs, observations_count, action_space_dimension) tensor with sampled action states

        '''

        rotations = inputs[0]
        translations = inputs[1]
        encoded_rotations = RotationEncoder.encode(rotations, dim=-1)
        bounding_box_size = self.bounding_box.get_size()
        normalized_translations = translations / bounding_box_size

        # Concatenates the inputs in a single tensor
        concatenated_inputs = torch.cat([encoded_rotations, normalized_translations], dim=-1)
        # If also deformations are present, retrieves and concatenates the deformations
        if self.use_deformation:
            if len(inputs) != 3:
                raise Exception("The use of deformations was requested, but no deformation input is present")
            deformations = inputs[2]
            concatenated_inputs = torch.cat([concatenated_inputs, deformations], dim=-1)
        object_in_scene = inputs[-1]
        
        flat_inputs, initial_input_dimensions = TensorFolder.flatten(concatenated_inputs)
        flat_object_in_scene, _ = TensorFolder.flatten(object_in_scene, 2)

        # Computes state embeddings mean and log variance
        x = self.mlp_backbone(flat_inputs, flat_object_in_scene)
        flat_states_mean = self.mean_fc(x)
        flat_states_log_variance = self.log_variance_fc(x)

        # Folds the tensors
        folded_states_mean = TensorFolder.fold(flat_states_mean, initial_input_dimensions)
        folded_states_log_variance = TensorFolder.fold(flat_states_log_variance, initial_input_dimensions)
        folded_states_distribution = torch.stack([folded_states_mean, folded_states_log_variance], dim=2)
        folded_sampled_states = self.sample(folded_states_mean, folded_states_log_variance)

        predecessor_mean, successor_mean = TensorSplitter.predecessor_successor_split(folded_states_mean)
        predecessor_log_variance, successor_log_variance = TensorSplitter.predecessor_successor_split(folded_states_log_variance)

        # The distribution of the difference vector is the difference of means and sum of variances
        action_directions_mean = successor_mean - predecessor_mean

        action_directions_variance = torch.exp(successor_log_variance) + torch.exp(predecessor_log_variance)
        action_directions_log_variance = torch.log(action_directions_variance)

        action_directions_distribution = torch.stack([action_directions_mean, action_directions_log_variance], dim=2)
        sampled_action_directions = self.sample_variance(action_directions_mean, action_directions_variance)  # Directly samples with variance to avoid additional operations

        flat_sampled_action_directions, initial_action_dimensions = TensorFolder.flatten(sampled_action_directions)
        # Computes the final action probabilities
        flat_action_logits = self.final_fc(flat_sampled_action_directions)
        folded_action_logits = TensorFolder.fold(flat_action_logits, initial_action_dimensions)

        return folded_action_logits, action_directions_distribution, sampled_action_directions, \
               folded_states_distribution, folded_sampled_states


def model(config, model_config):
    '''
    Instantiates an action network with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return ActionNetworkV5(config, model_config)


