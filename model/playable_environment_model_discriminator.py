import importlib
from typing import Dict, List, Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter

from model.environment_model import EnvironmentModel
from model.playable_environment_model_v2 import PlayableEnvironmentModelV2
from utils.lib_3d.bounding_box import BoundingBox


class PlayableEnvironmentModelDiscriminator(PlayableEnvironmentModelV2):  # Inherits from the version without the environment model parameters bug

    def __init__(self, config, environment_model: EnvironmentModel):
        '''
        Initializes the environment model

        :param config: the configuration file
        :param environment_model: the model for the environment. Must be in evaluation mode
        '''
        super(PlayableEnvironmentModelDiscriminator, self).__init__(config, environment_model)

        # Enables anomaly detection
        torch.autograd.set_detect_anomaly(self.config["playable_model"]["detect_anomaly"])
        if self.config["playable_model"]["detect_anomaly"]:
            self.register_nan_hooks()

        # Creates the discriminators
        discriminator_models = self.create_discriminator_models()
        self.discriminator_models = nn.ModuleList(discriminator_models)

        # Registers the forward functions
        self.register_forward_function(self.forward_vanilla_plus_discriminator, "vanilla_plus_discriminator")
        self.register_forward_function(self.forward_only_discriminator, "only_discriminator")

        # Computes which codes are to be fed to the discriminator
        if not "discriminator_input" in self.config["playable_model"]:
            self.config["playable_model"]["discriminator_input"] = ["deformation"]
        self.discriminator_input = self.config["playable_model"]["discriminator_input"]
        self.detach_translation = self.config["playable_model"]["detach_translation"]
        self.discriminator_bounding_box = BoundingBox(self.config["playable_model"]["discriminator_bounding_box"])

        # Map from input names to their key in the scene encoding dictionary
        self.scene_encoding_fields_map = {
            "rotation": "object_rotation_parameters",
            "translation": "object_translation_parameters",
            "style": "object_style",
            "deformation": "object_deformation",
        }

        # Map from input names to their key in the object_results dictionary
        self.object_results_fields_map = {
            "rotation": "reconstructed_object_rotations",
            "translation": "reconstructed_object_translations",
            "style": "reconstructed_object_style",
            "deformation": "reconstructed_object_deformation",
        }

    def register_nan_hooks(self):
        '''
        Registers hooks for finding nan values
        :return:
        '''

        for name, submodule in self.named_modules():
            submodule.__hook_name = name
            def nan_hook(self, inp, output):
                if isinstance(output, dict):
                    for value in output.values():
                        nan_hook(self, inp, value)
                        return
                elif isinstance(output, tuple) or isinstance(output, list):
                    for out in output:
                        nan_hook(self, inp, out)
                        return
                else:
                    if torch.is_tensor(output):
                        nan_mask = torch.isnan(output)
                        if nan_mask.any():
                            print("In", self.__class__.__name__)
                            raise RuntimeError(f"Found NAN in submodule {self.__hook_name} on tensor with size {list(output.size())} {output} at indices: ", nan_mask.nonzero())

            submodule.register_forward_hook(nan_hook)

    def non_environment_parameters(self, recurse: bool = True, additional_excluded_names=None) -> Iterator[Parameter]:
        '''
        Returns only non environment model parameters
        :param recurse:
        :param additional_excluded_names: Additional names not to include in the returned ones
        :return:
        '''

        # Gathers the names of parameters for the environment model
        excluded_names = set()
        for name, _ in self.discriminator_models.named_parameters():
            excluded_names.add("discriminator_models." + name)

        if additional_excluded_names is not None:
            additional_excluded_names = set(additional_excluded_names)
            excluded_names = excluded_names.union(additional_excluded_names)

        # Yields each parameter that is not in the environment model
        return super(PlayableEnvironmentModelDiscriminator, self).non_environment_parameters(recurse=recurse, additional_excluded_names=excluded_names)

    def discriminator_parameters(self, recurse: bool = True):
        '''
        Returns parameters for the discriminators
        :param recurse:
        :return:
        '''
        return self.discriminator_models.parameters(recurse=recurse)

    def create_discriminator_models(self) -> List[nn.Module]:
        '''
        Creates discriminator models for each dynamic object class
        :return: list of created models
        '''

        discriminator_models = []
        # Creates the model for each object as specified in the configuration
        for current_model_config in self.config["playable_model"]["discriminator_models"]:
            model_class = current_model_config["architecture"]

            current_model = getattr(importlib.import_module(model_class), 'model')(self.config, current_model_config)
            discriminator_models.append(current_model)

        return discriminator_models

    def set_discriminator_requires_grad(self, requires_grad: bool):
        '''
        Set the requires_grad parameter for the discriminator models
        :param requires_grad:
        :return:
        '''

        self.discriminator_models.requires_grad_(requires_grad=requires_grad)

    def forward_vanilla_plus_discriminator(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, ground_truth_observations: int, shuffle_style: bool = False,
                action_modifier=None) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param observations: see forward_vanilla
        :param camera_rotations: see forward_vanilla
        :param camera_translations: see forward_vanilla
        :param focals: see forward_vanilla
        :param bounding_boxes: see forward_vanilla
        :param bounding_boxes_validity: see forward_vanilla
        :param global_frame_indexes: see forward_vanilla
        :param video_frame_indexes: see forward_vanilla
        :param video_indexes: see forward_vanilla
        :param shuffle_style: see forward_vanilla
        :param ground_truth_observations: see forward_vanilla
        :param action_modifier: see forward_vanilla

        :return: scene_encoding, object_results
                 scene_encoding: see forward_vanilla
                                 Each object result possesses a field "discriminator_output" of size (batch_size)
                                 contains output of the discriminator for the fake output
                 object_results: see forward_vanilla
        '''

        # Makes the first part of the forward as in vanilla
        scene_encoding, object_results = self.forward(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity,
                             global_frame_indexes, video_frame_indexes, video_indexes, ground_truth_observations,
                             shuffle_style=shuffle_style, action_modifier=action_modifier, mode="vanilla")

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Computes losses for each dynamic object
        for dynamic_object_idx in range(dynamic_objects_count):

            # Gets animation results for the current object
            current_object_results = object_results[dynamic_object_idx]

            # Gets the valid entries in the sequence
            current_sequence_validity = current_object_results["sequence_validity"]
            current_sampled_actions = current_object_results["sampled_actions"]
            current_sampled_action_directions = current_object_results["sampled_action_directions"]

            # Gets the discriminator model
            discriminator_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_discriminator = self.discriminator_models[discriminator_model_idx]

            # Obtains the input sequence for the discriminator
            discriminator_input = self.get_discriminator_sequence_from_object_results(object_results, current_sampled_actions, current_sampled_action_directions, current_sequence_validity, dynamic_object_idx, detach=False)

            discriminator_output = current_discriminator(discriminator_input, current_sequence_validity)
            current_object_results["discriminator_output"] = discriminator_output

        return scene_encoding, object_results

    def get_discriminator_sequence_from_scene_encoding(self, scene_encoding, sampled_actions: torch.Tensor, sampled_action_directions: torch.Tensor, sequence_validity: torch.Tensor, dynamic_object_idx: int, detach: bool):
        '''
        Obtains an input sequence for the discriminator from the scene encoding

        :param scene_encoding: scene encoding as returned by the forward method
        :param sampled_actions: (batch_size, observations_count - 1, actions_count) tensor with sampled actions
        :param sampled_action_directions: (batch_size, observations_count - 1, action_space_dimension) tensor with sampled action directions
        :param sequence_validity: (batch_size, observations_count) boolean tensor with True if the entries at the given positions are valid
        :param dynamic_object_idx: id of the dynamic object for which to retrieve the sequence
        :param detach: if True detaches the tensors extracted from the scene encoding
        :return: (batch_size, observations_count, features_count) tensor with input sequence for the discriminator.
                 the number of features is the sum of all selected features for the discriminator
        '''

        object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)

        input_tensors = []
        for input_type in self.discriminator_input:

            if input_type == "action":
                # Replicates the last action to have observations_count actions. Actions are always detached
                current_tensor = torch.cat([sampled_actions, sampled_actions[:, -1:]], dim=-2).detach()
            elif input_type == "action_direction":
                # Replicates the last action direction to have observations_count actions. Actions are always detached
                current_tensor = torch.cat([sampled_action_directions, sampled_action_directions[:, -1:]], dim=-2).detach()
            else:
                scene_encoding_key = self.scene_encoding_fields_map[input_type]
                current_tensor = scene_encoding[scene_encoding_key][..., object_idx]

            # Detaches if requested
            if detach:
                current_tensor = current_tensor.detach()
            # If detach of translations is requested
            elif input_type == "translation" and self.detach_translation:
                current_tensor = current_tensor.detach()

            # Normalizes translation input
            if input_type == "translation":
                bounding_box_size = self.discriminator_bounding_box.get_size()
                current_tensor = current_tensor / bounding_box_size
            # Avoids modifications to propagate to the original tensor
            current_tensor = current_tensor.clone()
            input_tensors.append(current_tensor)

        # Concatenates into a single tensor and zeroes the entries that correspond to invalid locations
        input_tensors = torch.cat(input_tensors, dim=-1)
        input_tensors[sequence_validity == False] *= 0

        return input_tensors

    def get_discriminator_sequence_from_object_results(self, object_results: Dict, sampled_actions: torch.Tensor, sampled_action_directions: torch.Tensor, sequence_validity: torch.Tensor, dynamic_object_idx: int, detach: bool):
        '''
        Obtains an input sequence for the discriminator from the scene encoding

        :param object_results: object_results as returned by the forward method
        :param sampled_actions: (batch_size, observations_count - 1, actions_count) tensor with sampled actions
        :param sampled_action_directions: (batch_size, observations_count - 1, action_space_dimension) tensor with sampled action directions
        :param sequence_validity: (batch_size, observations_count) boolean tensor with True if the entries at the given positions are valid
        :param dynamic_object_idx: id of the dynamic object for which to retrieve the sequence
        :param detach: if True detaches the tensors extracted from the scene encoding
        :return: (batch_size, observations_count, features_count) tensor with input sequence for the discriminator.
                 the number of features is the sum of all selected features for the discriminator
        '''

        current_object_results = object_results[dynamic_object_idx]

        input_tensors = []
        for input_type in self.discriminator_input:

            if input_type == "action":
                # Replicates the last action to have observations_count actions. Actions are always detached
                current_tensor = torch.cat([sampled_actions, sampled_actions[:, -1:]], dim=-2).detach()
            elif input_type == "action_direction":
                # Replicates the last action direction to have observations_count actions. Actions are always detached
                current_tensor = torch.cat([sampled_action_directions, sampled_action_directions[:, -1:]], dim=-2).detach()
            else:
                scene_encoding_key = self.object_results_fields_map[input_type]
                current_tensor = current_object_results[scene_encoding_key]

            # Detaches if requested
            if detach:
                current_tensor = current_tensor.detach()
            # If detach of translations is requested
            elif input_type == "translation" and self.detach_translation:
                current_tensor = current_tensor.detach()

            # Normalizes translation input
            if input_type == "translation":
                bounding_box_size = self.discriminator_bounding_box.get_size()
                current_tensor = current_tensor / bounding_box_size
            # Avoids modifications to propagate to the original tensor
            current_tensor = current_tensor.clone()
            input_tensors.append(current_tensor)

        # Concatenates into a single tensor and zeroes the entries that correspond to invalid locations
        input_tensors = torch.cat(input_tensors, dim=-1)
        input_tensors[sequence_validity == False] *= 0

        return input_tensors

    def forward_only_discriminator(self, scene_encoding, object_results: Dict):
        '''
        Forwards real and reconstructed sequences through the discriminator.
        Inputs are detached to avoid backpropagation outside the discriminator.

        :param scene_encoding: scene encoding as returned by the forward method
        :param object_results: object results as returned by the forward method
        :return: dictionary with results for each dynamic object with the following fields:
                 "discriminator_output_real" (batch_size) discriminator output for the real images
                 "discriminator_output_fake" (batch_size) discriminator output for the fake images
        '''

        results = {}

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Computes losses for each dynamic object
        for dynamic_object_idx in range(dynamic_objects_count):

            # Gets animation results for the current object
            current_object_results = object_results[dynamic_object_idx]

            # Gets the valid entries in the sequence
            current_sequence_validity = current_object_results["sequence_validity"]
            current_sampled_actions = current_object_results["sampled_actions"]
            current_sampled_action_directions = current_object_results["sampled_action_directions"]

            # Gets the discriminator model
            discriminator_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_discriminator = self.discriminator_models[discriminator_model_idx]

            # Obtains input for the discriminator. Detaches to avoid backpropagation beyond the discriminator
            discriminator_input_real = self.get_discriminator_sequence_from_scene_encoding(scene_encoding, current_sampled_actions, current_sampled_action_directions, current_sequence_validity, dynamic_object_idx, detach=True)
            discriminator_input_fake = self.get_discriminator_sequence_from_object_results(object_results, current_sampled_actions, current_sampled_action_directions, current_sequence_validity, dynamic_object_idx, detach=True)

            current_discriminator_results = {}
            discriminator_output_fake = current_discriminator(discriminator_input_fake, current_sequence_validity)
            current_discriminator_results["discriminator_output_fake"] = discriminator_output_fake

            discriminator_output_real = current_discriminator(discriminator_input_real, current_sequence_validity)
            current_discriminator_results["discriminator_output_real"] = discriminator_output_real

            results[dynamic_object_idx] = current_discriminator_results

            # Pytorch requires one of the first level distionary outputs to be a tensor for hook registration. Use an arbitrary tensor
            results["pytorch_hook"] = discriminator_output_real

        return results


def model(config, environment_model: EnvironmentModel):
    return PlayableEnvironmentModelDiscriminator(config, environment_model)
