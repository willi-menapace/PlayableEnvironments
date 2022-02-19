import importlib
from typing import Tuple, Dict, List

import torch
import torch.nn as nn

from model.layers.centroid_estimator import CentroidEstimator
from model.layers.gumbel_softmax import GumbelSoftmax
from utils.tensor_folder import TensorFolder


class ObjectAnimationModel(nn.Module):
    '''
    Model for the dynamics of the environment
    '''

    def __init__(self, config: Dict, model_config: Dict):
        '''
        Initializes the model representing the dynamics of the environment

        :param config: the configuration file
        :param model_config: the configuration for the specific object
        '''
        super(ObjectAnimationModel, self).__init__()

        self.config = config

        # Transfers configuration parameters to submodels
        self.propagate_configurations(model_config)

        # Instantiates the dynamics network
        model_class = model_config["dynamics_network"]["architecture"]
        self.dynamics_network = getattr(importlib.import_module(model_class), 'model')(self.config, model_config["dynamics_network"])
        # Instantiates the action network
        model_class = model_config["action_network"]["architecture"]
        self.action_network = getattr(importlib.import_module(model_class), 'model')(self.config, model_config["action_network"])

        # Computes whether deformation is used to compute actions
        self.use_deformation = self.action_network.use_deformation

        # Instantiates the centroid estimator
        self.actions_count = model_config["actions_count"]
        self.action_space_dimension = model_config["action_space_dimension"]
        self.centroid_estimation_alpha = model_config["centroid_estimator"]["alpha"]
        self.centroid_estimator = CentroidEstimator(self.actions_count, self.action_space_dimension, self.centroid_estimation_alpha)

        self.gumbel_temperature = model_config["gumbel_temperature"]
        self.hard_gumbel = model_config["hard_gumbel"]
        self.gumbel_softmax = GumbelSoftmax(self.gumbel_temperature, self.hard_gumbel)

    def propagate_configurations(self, model_config: Dict):

        for current_model_config in [model_config["action_network"], model_config["dynamics_network"]]:
            current_model_config["style_features"] = model_config["style_features"]
            current_model_config["deformation_features"] = model_config["deformation_features"]
            current_model_config["actions_count"] = model_config["actions_count"]
            current_model_config["action_space_dimension"] = model_config["action_space_dimension"]

    def reinit_memory(self):
        '''
        Initializes the state of the recurrent model
        :return:
        '''

        self.dynamics_network.reinit_memory()

    def get_memory_state(self):
        '''
        Obtains the current state of the memory

        :return: see dynamics_network
        '''

        return self.dynamics_network.get_memory_state()

    def set_memory_state(self, memory_state: Tuple[torch.Tensor]):
        '''
        Sets the state of the memory

        :param memory_state: see dynamics_network
        :return:
        '''

        self.dynamics_network.set_memory_state(memory_state)

    def forward(self, object_rotations: torch.Tensor, object_translations: torch.Tensor,
                object_style: torch.Tensor, object_deformation: torch.Tensor, object_in_scene: torch.Tensor,
                ground_truth_observations: int, action_modifier=None) -> Dict:
        '''
        Generates the next state of the environment starting from the current state and the actions for the current step

        :param object_rotations: (bs, observations_count, 3) tensor with rotations of the object
        :param object_translations: (bs, observations_count, 3) tensor with translations of the object
        :param object_style: (bs, observations_count, style_features_count) tensor with style of the object
        :param object_deformation: (bs, observations_count, deformation_features_count) tensor with deformations of the object
        :param object_in_scene: (bs, observations_count) boolean tensor with True if the object is in the scene at the given time
        :param ground_truth_observations: number of observations to use as ground truth
        :param action_modifier: action modifier object. If provided, sampled actions and action variations are modified according to this modifier

        :return: Dictionary with reconstruction of the input parameters and action information
                 (bs, observations_count, 3) tensor with reconstructed rotations of the object
                 (bs, observations_count, 3) tensor with reconstructed translations of the object
                 (bs, observations_count, style_features_count) tensor with reconstructed style of the object
                 (bs, observations_count, deformation_features_count) tensor with reconstructed deformations of the object
                 (bs, observations_count - 1, actions_count) tensor with sampled actions
                 (bs, observations_count - 1, actions_count) tensor with action logits
                 (bs, observations_count - 1, 2, action_space_dimension) tensor with distribution of action directions
                 (bs, observations_count - 1, action_space_dimension) tensor with sampled action directions
                 (bs, observations_count, 2, action_space_dimension) tensor with distribution of action states
                 (bs, observations_count, action_space_dimension) tensor with sampled action states
                 (bs, observations_count - 1, action_space_dimension) tensor with action variations
                 (bs, observations_count) boolean tensor with True if the results are valid at the given points

                 (actions_count, action_space_dim) tensor with estimated action centroids
        '''

        # Computes the valid parts of the sequence
        sequence_validity = ObjectAnimationModel.compute_sequence_validity(object_in_scene)

        action_inputs = [object_rotations, object_translations]
        if self.use_deformation:  # Adds deformation if needed
            action_inputs.append(object_deformation)
        action_inputs.append(object_in_scene)
        sampled_actions, action_logits, action_directions_distribution, sampled_action_directions, action_states_distribution, sampled_action_states \
            = self.compute_actions(*action_inputs)

        action_probabilities = torch.softmax(action_logits, dim=-1)

        # Updates the centroids for action
        self.update_centroids(action_probabilities, action_directions_distribution, sequence_validity[:, :-1])

        # Computes the variations of each action using the sampled directions
        action_variations = self.centroid_estimator.compute_variations(sampled_action_directions, sampled_actions)
        estimated_action_centroids = self.centroid_estimator.get_estimated_centroids()

        # If a modifier is specified, apply its modification
        if action_modifier is not None:
            sampled_actions, action_variations = action_modifier.apply(sampled_actions, action_variations)

        reconstructed_object_rotations, reconstructed_object_translations, reconstructed_object_style, reconstructed_object_deformation \
            = self.forward_through_dynamics(object_rotations, object_translations, object_style, object_deformation,
                                            sampled_actions, action_variations, ground_truth_observations)

        reconstructed_action_inputs = [reconstructed_object_rotations, reconstructed_object_translations]
        if self.use_deformation:  # Adds deformation if needed
            reconstructed_action_inputs.append(reconstructed_object_deformation)
        reconstructed_action_inputs.append(object_in_scene)
        reconstructed_sampled_actions, reconstructed_action_logits, reconstructed_action_directions_distribution, \
        reconstructed_sampled_action_directions, reconstructed_action_states_distribution, reconstructed_sampled_action_states \
            = self.compute_actions(*reconstructed_action_inputs)

        # Builds and returns the results
        results = {
            # Reconstructed states of the environment
            "reconstructed_object_rotations": reconstructed_object_rotations,
            "reconstructed_object_translations": reconstructed_object_translations,
            "reconstructed_object_style": reconstructed_object_style,
            "reconstructed_object_deformation": reconstructed_object_deformation,

            # Actions
            "sampled_actions": sampled_actions,
            "action_logits": action_logits,
            "action_directions_distribution": action_directions_distribution,
            "sampled_action_directions": sampled_action_directions,
            "action_states_distribution": action_states_distribution,
            "sampled_action_states": sampled_action_states,
            "action_variations": action_variations,

            # Reconstructed actions
            "reconstructed_action_logits": reconstructed_action_logits,
            "reconstructed_action_directions_distribution": reconstructed_action_directions_distribution,
            "reconstructed_sampled_action_directions": reconstructed_sampled_action_directions,
            "reconstructed_action_states_distribution": reconstructed_action_states_distribution,
            "reconstructed_sampled_action_states": reconstructed_sampled_action_states,

            "sequence_validity": sequence_validity,
            "estimated_action_centroids": estimated_action_centroids,
        }

        return results

    @staticmethod
    def compute_sequence_validity(object_in_scene: torch.Tensor) -> torch.Tensor:
        '''
        Computes whether the points in the sequences where obtained results are valid

        :param object_in_scene: (bs, observations_count) boolean tensor with True if the object is in the scene at the given time
        :return: (bs, observations_count) boolean tensor with True if the results are valid at the given points
        '''

        observations_count = object_in_scene.size(1)
        sequence_validity = object_in_scene.clone()
        for observation_idx in range(observations_count - 1):
            # Propagates Falses to the column on the right
            false_mask = sequence_validity[:, observation_idx] == False
            sequence_validity[:, observation_idx + 1][false_mask] = False

        return sequence_validity

    def compute_actions(self, *inputs: List[torch.Tensor]):
        '''
        Computes actions associated to the given inputs

        :param inputs: list of (bs, observations_count, features_count) tensors representing the input on which to compute actions
                       last list element is (bs, observations_count) tensor with true if the object is in scene
        :return: (bs, observations_count - 1, actions_count) tensor with sampled actions
                 (bs, observations_count - 1, actions_count) tensor with logits of probabilities for each action
                 (bs, observations_count - 1, 2, action_space_dimension) tensor posterior mean and log variance for action directions
                 (bs, observations_count - 1, action_space_dimension) tensor with sampled action directions
                 (bs, observations_count, 2, action_space_dimension) tensor posterior mean and log variance for action states
                 (bs, observations_count, action_space_dimension) tensor with sampled action states

        '''

        # Computes actions
        action_logits, *other_results = self.action_network(*inputs)

        # Samples the actions with gumbel softmax
        log_action_probabilities = torch.log_softmax(action_logits, dim=-1)
        sampled_actions = self.gumbel_softmax(log_action_probabilities)

        return tuple([sampled_actions, action_logits] + list(other_results))

    def update_centroids(self, action_probabilities: torch.Tensor, action_directions_distribution: torch.Tensor, sequence_validity: torch.Tensor):
        '''
        Updates the centroids corresponding to the inferred actions

        :param action_probabilities: (..., actions_count) tensor with probabilities for each action in [0, 1]
        :param action_directions_distribution: (..., 2, action_space_dimension) tensor posterior mean and log variance for action directions
        :param sequence_validity: (...) boolean tensor with True if the results are valid at the given points
        :return:
        '''

        flat_action_probability, _ = TensorFolder.flatten(action_probabilities, -1)
        flat_action_directions_distribution, _ = TensorFolder.flatten(action_directions_distribution, -2)
        flat_sequence_validity, _ = TensorFolder.flatten(sequence_validity, 0)  # Completely flattens it

        # Filters only valid entries in the sequence
        flat_action_probability = flat_action_probability[flat_sequence_validity, ...]
        flat_action_directions_distribution = flat_action_directions_distribution[flat_sequence_validity, ...]

        # Updates the centroids
        self.centroid_estimator.update_centroids(flat_action_directions_distribution, flat_action_probability)

    def forward_through_dynamics(self, object_rotations: torch.Tensor, object_translations: torch.Tensor,
                object_style: torch.Tensor, object_deformation: torch.Tensor,
                actions: torch.Tensor, action_variations: torch.Tensor, ground_truth_observations: int) -> torch.Tensor:
        '''
        Generates the next state of the environment starting from the current state and the actions for the current step

        :param object_rotations: (bs, observations_count, 3) tensor with rotations of the object
        :param object_translations: (bs, observations_count, 3) tensor with translations of the object
        :param object_style: (bs, observations_count, style_features_count) tensor with style of the object
        :param object_deformation: (bs, observations_count, deformation_features_count) tensor with deformations of the object
        :param actions: (bs, observations_count - 1, actions_count) tensor with current actions
        :param action_variations: (bs, observations_count - 1, action_space_dimension) tensor with variation for the current action
        :param ground_truth_observations: number of observations to use as ground truth

        :return: (bs, observations_count, 3) tensor with reconstructed rotations of the object
                 (bs, observations_count, 3) tensor with reconstructed translations of the object
                 (bs, observations_count, style_features_count) tensor with reconstructed style of the object
                 (bs, observations_count, deformation_features_count) tensor with reconstructed deformations of the object
        '''

        observations_count = object_rotations.size(1)

        reconstructed_object_rotations = [object_rotations[:, 0]]
        reconstructed_object_translations = [object_translations[:, 0]]
        reconstructed_object_style = [object_style[:, 0]]
        reconstructed_object_deformation = [object_deformation[:, 0]]

        # Reconstructs each step of the sequence
        for observation_idx in range(observations_count - 1):
            # Use ground truth information for the specified number of initial steps
            if observation_idx < ground_truth_observations:
                current_object_rotation = object_rotations[:, observation_idx]
                current_object_translations = object_translations[:, observation_idx]
                current_object_style = object_style[:, observation_idx]
                current_object_deformation = object_deformation[:, observation_idx]
            # Otherwise use the reconstructed results
            else:
                current_object_rotation = reconstructed_object_rotations[-1]
                current_object_translations = reconstructed_object_translations[-1]
                current_object_style = reconstructed_object_style[-1]
                current_object_deformation = reconstructed_object_deformation[-1]

            # Gets the current actions
            current_action = actions[:, observation_idx]
            current_action_variation = action_variations[:, observation_idx]

            # Predicts the next step
            next_object_rotations, next_object_translation, next_object_style, next_object_deformation \
                = self.dynamics_network(current_object_rotation, current_object_translations, current_object_style,
                                        current_object_deformation, current_action, current_action_variation)

            reconstructed_object_rotations.append(next_object_rotations)
            reconstructed_object_translations.append(next_object_translation)
            reconstructed_object_style.append(next_object_style)
            reconstructed_object_deformation.append(next_object_deformation)

        # Creates output tensors
        reconstructed_object_rotations = torch.stack(reconstructed_object_rotations, dim=1)
        reconstructed_object_translations = torch.stack(reconstructed_object_translations, dim=1)
        reconstructed_object_style = torch.stack(reconstructed_object_style, dim=1)
        reconstructed_object_deformation = torch.stack(reconstructed_object_deformation, dim=1)

        return reconstructed_object_rotations, reconstructed_object_translations, reconstructed_object_style, reconstructed_object_deformation


def model(config, model_config):
    '''
    Istantiates a dynamics network with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return ObjectAnimationModel(config, model_config)


if __name__ == "__main__":

    object_in_scene = torch.as_tensor([[True, False, True], [False, True, True]])

    result = ObjectAnimationModel.compute_sequence_validity(object_in_scene)
    pass
