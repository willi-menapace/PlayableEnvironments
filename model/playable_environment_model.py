import importlib
from typing import Tuple, Dict, List, Iterator, Callable

import torch
import torch.nn as nn
from torch.nn import Parameter

from model.environment_model import EnvironmentModel
from model.utils.object_ids_helper import ObjectIDsHelper
from utils.drawing.bounding_box_drawer import BoundingBoxDrawer


class PlayableEnvironmentModel(nn.Module):

    def __init__(self, config, environment_model: EnvironmentModel):
        '''
        Initializes the environment model. The received model will be frozen

        :param config: the configuration file
        :param environment_model: the model for the environment. Must be in evaluation mode and will be frozen
        '''
        super(PlayableEnvironmentModel, self).__init__()

        self.config = config
        if environment_model.training:
            raise Exception("The environment model must be in evaluation mode")
        environment_model.requires_grad_(False)  # Freezes the model
        self.environment_model = environment_model

        current_models = self.create_object_animation_models()
        self.object_animation_models = nn.ModuleList(current_models)

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        # Map containing forward function names and function objects
        self.forward_functions_map = {}
        self.register_forward_function(self.forward_vanilla, "vanilla")

    def train(self, mode: bool = True):
        '''
        Overrides train not to affect the environment model which always remains in evaluation mode
        :param mode:
        :return:
        '''
        super(PlayableEnvironmentModel, self).train(mode)

        # The environment model is always in evaluation mode
        self.environment_model.eval()
        return self

    def non_environment_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        '''
        Returns only non environment model parameters
        :param recurse:
        :return:
        '''

        # Gathers the names of parameters for the environment model
        excluded_names = set()
        for name, _ in self.environment_model.named_parameters():
            excluded_names.add(name)

        # Yields each parameter that is not in the environment model
        for name, parameter in super(PlayableEnvironmentModel, self).named_parameters(recurse=recurse):
            if name not in excluded_names:
                yield parameter

    def create_object_animation_models(self) -> List[nn.Module]:
        '''
        Creates object animation models for each dynamic object class
        :return: list of created models
        '''

        object_models = []
        # Creates the model for each object as specified in the configuration
        for current_object_config in self.config["playable_model"]["object_animation_models"]:
            model_class = current_object_config["architecture"]

            current_model = getattr(importlib.import_module(model_class), 'model')(self.config, current_object_config)
            object_models.append(current_model)

        return object_models

    def get_object_scene_encoding(self, scene_encoding: Dict, dynamic_object_idx: int):
        '''
        Gets the scene encoding corresponding to the given dynamic object id.

        :param scene_encoding: dictionary representing the scene encoding
        :param dynamic_object_idx: Id of the dynamic object
        :return: The scene encoding object with the object_count dimension eliminated in its tensors
        '''

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count
        if dynamic_object_idx > dynamic_objects_count:
            raise Exception("Requested scene encoding for object {dynamic_object_idx}, but only {dynamic_objects_count} dynamic objects are present")

        if len(scene_encoding.keys()) != 8:
            raise Exception("The passed scene encoding does not have the expected number of keys. Did you change its fields but did not update this function?")

        object_scene_encoding = {}
        # Transfers tensors that need not be modified
        for key in ["camera_rotations", "camera_translations", "focals"]:
            object_scene_encoding[key] = scene_encoding[key]
        # Transfers the tensors that need to be indexed by the object id first
        for key in ["object_rotation_parameters", "object_translation_parameters", "object_style", "object_deformation", "object_in_scene"]:
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)
            object_scene_encoding[key] = scene_encoding[key][..., object_idx]

        return object_scene_encoding

    def register_forward_function(self, function: Callable, function_name: str, override=False):
        '''
        Registers a forward function
        :param function: the function to register
        :param function_name: the name with which to register the function
        :param override: whether the function is substituting or not a function with the same name
        :return:
        '''
        if function_name in self.forward_functions_map and not override:
            raise Exception(f"Function {function_name} already registered and override is not specified. Did you want to override?")
        if override:
            if function_name in self.forward_functions_map:
                self.forward_functions_map[function_name] = function
            else:
                raise Exception(f"Override of function {function_name} was required, but the function to override was not found")

        self.forward_functions_map[function_name] = function

    def forward(self, *args, mode="vanilla", **kwargs):
        '''

        :param args: Positional arguments to pass to the target forward mode
        :param mode: Mode to use for the forward
                     "vanilla" extracts and reconstructs the input latent space
        :param kwargs: Keyword arguments to pass to the target forward mode
        :return: Output of the chosen forward mode
        '''

        if mode not in self.forward_functions_map:
            raise Exception(f"Unknown forward mode '{mode}'")

        return_value = self.forward_functions_map[mode](*args, **kwargs)

        return return_value

    def forward_vanilla(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, ground_truth_observations: int, shuffle_style: bool = False,
                action_modifier=None) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param observations: (bs, observations_count, cameras_count, 3, height, width) observations
        :param camera_rotations: (bs, observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (bs, observations_count, cameras_count, 3) measured camera_translation
        :param focals: (bs, observations_count, cameras_count) measured camera focal lengths
        :param bounding_boxes: (bs, observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (bs, observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the dynamic object is present in the scene
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param shuffle_style: True if style codes should be shuffled between observations at different temporal points
        :param ground_truth_observations: number of observations to use as ground truth
        :param action_modifier: action modifier object. If provided, sampled actions and action variations are modified according to this modifier

        :return: scene_encoding, object_results
                 scene_encoding: Dictionary with scene encoding as returned by the environment model
                 object_results: Dictionary with integer dynamic_object_id keys
                                 Each dynamic_object_id key is associated to the results returned by the corresponding object animation model
        '''

        with torch.no_grad():
            # Computes scene encodings using the environment model. No gradient is required to freeze the network
            scene_encoding = self.environment_model(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, shuffle_style, mode="observations_scene_encoding_only")

        # Computes reconstructions of the scene encoding for each object
        object_results = {}
        for dynamic_object_idx in range(self.object_id_helper.dynamic_objects_count):

            # Gets the encodings for the current object
            object_scene_encoding = self.get_object_scene_encoding(scene_encoding, dynamic_object_idx)

            # Extracts the needed fields
            object_animation_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_object_animation_model = self.object_animation_models[object_animation_model_idx]
            object_rotations = object_scene_encoding["object_rotation_parameters"]
            object_translations = object_scene_encoding["object_translation_parameters"]
            object_style = object_scene_encoding["object_style"]
            object_deformation = object_scene_encoding["object_deformation"]
            object_in_scene = object_scene_encoding["object_in_scene"]

            # Reinits the memory of the model
            current_object_animation_model.reinit_memory()
            # Computes results for the current sequence
            current_results = current_object_animation_model(object_rotations, object_translations, object_style, object_deformation, object_in_scene, ground_truth_observations, action_modifier)
            object_results[dynamic_object_idx] = current_results

        return scene_encoding, object_results

    def get_reconstructed_observations_from_render_results(self, render_results: Dict) -> torch.Tensor:
        '''
        Gets the reconstructed observations from the current render results

        :param render_results: Render results as returned by the environment model
        :return: (batch_size, observations_count, cameras_count, height, width, 3) tensor with reconstructed observations
        '''

        current_render_results = render_results["coarse"]["global"]

        # Autoencoder rendering
        if "reconstructed_observations" in current_render_results:
            reconstructed_observations = current_render_results["reconstructed_observations"]
            # Puts observations in HWC order
            reconstructed_observations = reconstructed_observations.permute([0, 1, 2, 4, 5, 3])
        # NeRF rendering
        else:
            reconstructed_observations = current_render_results["integrated_features"]

        return reconstructed_observations

    def initialize_interactive_generation(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, batch_idx: int, observation_idx: int) -> Dict[str, torch.Tensor]:
        '''
        Initializes the model for interactive generation from the element at the specified batch and observation idx
        Returns the current observation and scene encoding

        :param observations: (bs, observations_count, cameras_count, 3, height, width) observations
        :param camera_rotations: (bs, observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (bs, observations_count, cameras_count, 3) measured camera_translation
        :param focals: (bs, observations_count, cameras_count) measured camera focal lengths
        :param bounding_boxes: (bs, observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (bs, observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the dynamic object is present in the scene
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param batch_idx: id of the batch to use for initialization
        :param observation_idx: id of the observation to use for initialization


        :return: (height, width, 3) tensor with the current observation
                 scene_encoding: Dictionary with scene encoding as returned by the environment model
                                 contains entry "animation_models_memory_state": List with memory states for object animation models of each dynamic object
        '''

        # Reinit the memory of the models
        for current_object_animation_model in self.object_animation_models:
            current_object_animation_model.reinit_memory()

        # Gets the current state of the memory for each dynamic object
        all_animation_model_memory_states = []
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count
        for dynamic_object_idx in range(dynamic_objects_count):
            object_animation_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_object_animation_model = self.object_animation_models[object_animation_model_idx]
            all_animation_model_memory_states.append(current_object_animation_model.get_memory_state())

        # Extracts only the element of interest
        observations = observations[batch_idx, observation_idx].unsqueeze(0).unsqueeze(0)
        camera_rotations = camera_rotations[batch_idx, observation_idx].unsqueeze(0).unsqueeze(0)
        camera_translations = camera_translations[batch_idx, observation_idx].unsqueeze(0).unsqueeze(0)
        focals = focals[batch_idx, observation_idx].unsqueeze(0).unsqueeze(0)
        bounding_boxes = bounding_boxes[batch_idx, observation_idx].unsqueeze(0).unsqueeze(0)
        bounding_boxes_validity = bounding_boxes_validity[batch_idx, observation_idx].unsqueeze(0).unsqueeze(0)

        image_size = (observations.size(-2), observations.size(-1))

        with torch.no_grad():
            # Computes scene encodings using the environment model. No gradient is required to freeze the network
            scene_encoding = self.environment_model(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, shuffle_style=False, mode="observations_scene_encoding_only")

            # Extracts the relevant parameters from the scene encodings
            object_rotations = scene_encoding["object_rotation_parameters"]
            object_translations = scene_encoding["object_translation_parameters"]
            object_style = scene_encoding["object_style"]
            object_deformation = scene_encoding["object_deformation"]
            object_in_scene = scene_encoding["object_in_scene"]

            # Renders the current results
            render_results = self.environment_model.render_full_frame_from_scene_encoding(camera_rotations, camera_translations,
                                                                                 focals, image_size, object_rotations,
                                                                                 object_translations, object_style,
                                                                                 object_deformation, object_in_scene,
                                                                                 perturb=False, samples_per_image_batching=1200)

            # Gets the reconstruction results
            reconstructed_observations = self.get_reconstructed_observations_from_render_results(render_results)
            reconstructed_observations = reconstructed_observations[0, 0, 0]

        scene_encoding["animation_models_memory_state"] = all_animation_model_memory_states

        return reconstructed_observations, scene_encoding

    def generate_next(self, actions: List[int], state: Dict, image_size: Tuple[int, int], sample_action_variations=False, draw_axes: bool=False, use_initial_style: bool=False):
        '''
        Generates the next observation given the action for each dynamic object and the state of the environment.
        Current state may be overwritten

        :param actions: List with one action for each dynamic object
        :param state: Current state as returned by the prededing call or by the initialize_interactive_generation method.
                      May be overwritten
        :param image_size: (height, width) size of the image to generate
        :param sample_action_variations: If true samples action variations, otherwise sets them to 0
        :param draw_axes: whether to draw axes on the images
        :param use_initial_style: whether to use the initial style rather than recomputing it
        :return: (height, width, 3) tensor with the current observation
                 scene_encoding: Dictionary with scene encoding as returned by the environment model
        '''

        objects_count = self.object_id_helper.objects_count
        static_objects_count = self.object_id_helper.static_objects_count
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Extracts the relevant parameters from the scene encodings
        camera_rotations = state["camera_rotations"]
        camera_translations = state["camera_translations"]
        focals = state["focals"]
        object_rotations = state["object_rotation_parameters"]
        object_translations = state["object_translation_parameters"]
        object_style = state["object_style"]
        object_deformation = state["object_deformation"]
        object_in_scene = state["object_in_scene"]

        all_animation_model_memory_states = state["animation_models_memory_state"]

        # Updates each dynamic object according to the current action
        for dynamic_object_idx in range(dynamic_objects_count):
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)

            object_animation_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_action = actions[dynamic_object_idx]
            current_object_animation_model = self.object_animation_models[object_animation_model_idx]

            # Reinits the memory for the current animation model to the last value.
            # Necessary if the same model is used for multiple object instances
            current_object_animation_model_memory_state = all_animation_model_memory_states[dynamic_object_idx]
            current_object_animation_model.set_memory_state(current_object_animation_model_memory_state)

            # Extracts the information for the current object.
            # The returned elements have only one observation, so we select that observation
            current_object_rotations = object_rotations[:, 0, ..., object_idx]
            current_object_translations = object_translations[:, 0, ..., object_idx]
            current_object_style = object_style[:, 0, ..., object_idx]
            current_object_deformation = object_deformation[:, 0, ..., object_idx]

            current_actions_count = current_object_animation_model.actions_count
            current_action_space_dimension = current_object_animation_model.action_space_dimension

            # Creates a one hot representation of the current action
            current_object_action = torch.zeros((1, current_actions_count), dtype=torch.float32, device=object_rotations.device)
            current_object_action[0, current_action] = 1.0
            # If action variations are to be sampled, sample them
            if sample_action_variations:
                current_object_action_variation = torch.randn((1, current_action_space_dimension), dtype=torch.float32, device=object_rotations.device)
            else:
                current_object_action_variation = torch.zeros((1, current_action_space_dimension), dtype=torch.float32, device=object_rotations.device)

            # Uses the dynamics network to predict the next state
            next_results = current_object_animation_model.dynamics_network(current_object_rotations, current_object_translations, current_object_style, current_object_deformation, current_object_action, current_object_action_variation)
            next_object_rotations, next_object_translations, next_object_style, next_object_deformation = next_results

            # Saves the current memory state of the object animation model.
            # Value is updated also in the original dictionary
            all_animation_model_memory_states[dynamic_object_idx] = current_object_animation_model.get_memory_state()

            # Updates the state with the new values
            state["object_rotation_parameters"][:, 0, ..., object_idx] = next_object_rotations
            state["object_translation_parameters"][:, 0, ..., object_idx] = next_object_translations
            if not use_initial_style:
                state["object_style"][:, 0, ..., object_idx] = next_object_style
            state["object_deformation"][:, 0, ..., object_idx] = next_object_deformation

        # Renders the current results
        render_results = self.environment_model.render_full_frame_from_scene_encoding(camera_rotations, camera_translations,
                                                                                      focals, image_size, object_rotations,
                                                                                      object_translations, object_style,
                                                                                      object_deformation, object_in_scene,
                                                                                      perturb=False, samples_per_image_batching=1200)

        # Gets the reconstruction results
        reconstructed_observations = self.get_reconstructed_observations_from_render_results(render_results)
        # Draws the axes if requested
        if draw_axes:
            projected_axes = render_results["projected_axes"]
            reconstructed_observations = reconstructed_observations.permute([0, 1, 2, 5, 3, 4])
            # Draws the axes of each dynamic object
            for object_idx in range(static_objects_count, objects_count):
                current_projected_axes = projected_axes[..., object_idx]
                reconstructed_observations = BoundingBoxDrawer.draw_axes(reconstructed_observations, current_projected_axes)

            reconstructed_observations = reconstructed_observations.permute([0, 1, 2, 4, 5, 3])

        reconstructed_observations = reconstructed_observations[0, 0, 0]

        return reconstructed_observations, state


def model(config, environment_model: EnvironmentModel):
    return PlayableEnvironmentModel(config, environment_model)
