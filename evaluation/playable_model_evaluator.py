import os
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset.batching import single_batch_elements_collate_fn, Batch
from dataset.video_dataset import MulticameraVideoDataset
from evaluation.action_modifiers.zero_variation_action_modifier import ZeroVariationActionModifier
from model.playable_environment_model_discriminator import PlayableEnvironmentModelDiscriminator
from model.utils.object_ids_helper import ObjectIDsHelper
from training.playable_model_trainer import PlayableModelTrainer
from utils.average_meter import AverageMeter
from utils.drawing.image_helper import ImageHelper
from utils.drawing.video_saver import VideoSaver


class PlayableModelEvaluator:
    '''
    Helper class for model evaluation
    '''

    def __init__(self, config, trainer: PlayableModelTrainer, dataset: MulticameraVideoDataset, logger, logger_prefix="playable_model_test"):
        self.config = config
        self.trainer = trainer
        self.logger = logger
        self.logger_prefix = logger_prefix
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.config["playable_model_evaluation"]["batching"]["batch_size"], shuffle=True, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)
        self.amt_playability_dataloader = DataLoader(dataset, batch_size=self.config["playable_model_evaluation"]["batching"]["batch_size"], shuffle=False, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)
        if "imaging_batch_size" not in self.config["playable_model_evaluation"]["batching"]:
            self.config["playable_model_evaluation"]["batching"]["imaging_batch_size"] = 1
        self.imaging_dataloader = DataLoader(dataset, batch_size=self.config["playable_model_evaluation"]["batching"]["imaging_batch_size"], shuffle=True, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)

        # The maximum number of batches to use for evaluation
        self.max_batches = self.config["playable_model_evaluation"]["max_evaluation_batches"]

        self.bounding_box_color = (255, 0, 0)
        self.ground_truth_bounding_box_color = (0, 0, 255)

        # Parameters for extra cameras for evaluation
        self.extra_cameras_rotations = torch.as_tensor(self.config["playable_model_evaluation"]["extra_cameras"]["camera_rotations"], dtype=torch.float, device="cuda")
        self.extra_cameras_translations = torch.as_tensor(self.config["playable_model_evaluation"]["extra_cameras"]["camera_translations"], dtype=torch.float, device="cuda")
        self.extra_cameras_focals = torch.as_tensor(self.config["playable_model_evaluation"]["extra_cameras"]["camera_focals"], dtype=torch.float, device="cuda")

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        # Helper for loggin the images
        self.image_helper = ImageHelper(self.config, logger, logger_prefix)

        # Creates the action modifiers to use for producing sequences
        self.action_modifiers = [None, ZeroVariationActionModifier()]

    def evaluate(self, playable_environment_model, environment_model, step: int):
        '''
        Evaluates the performances of the given model

        :param playable_environment_model: The model for the playable environment
        :param environment_model: The model for the environment
        :param step: The current step
        :param action_modifier: action modifier object. If provided, sampled actions and action variations are modified according to this modifier
        :return:
        '''

        self.logger.print(f'== Evaluation [{self.trainer.global_step}] ==')

        # Saves sample images
        with torch.no_grad():
            for idx, batch in enumerate(self.imaging_dataloader):

                # Saves reconstruction results for the current batch
                self.save_reconstruction_from_batch(playable_environment_model, environment_model, batch, step, self.action_modifiers)

                # Saves a sequence for each learned action for the first batch element
                self.save_action_videos_from_batch(playable_environment_model, batch, step)

                break

        self.logger.print(f"- Saving losses")
        self.save_losses(playable_environment_model)

        return

    def save_losses(self, playable_environment_model):

        forward_mode = "vanilla"
        if isinstance(playable_environment_model.module, PlayableEnvironmentModelDiscriminator):
            forward_mode = "vanilla_plus_discriminator"

        average_meter = AverageMeter()
        # Saves sample images
        with torch.no_grad():
            for idx, batch in enumerate(self.dataloader):

                # Breaks if the maximum number of batches is reached
                if self.max_batches is not None and idx > self.max_batches:
                    break

                loss, loss_info, additional_info, _ = self.trainer.compute_losses(playable_environment_model, batch, ground_truth_observations_count_override=1, forward_mode=forward_mode)

                # Accumulates loss information
                average_meter.add(loss_info)

        self.logger.print(f'step: {self.trainer.global_step}/{self.trainer.max_steps}', end=" ")

        average_values = {description: average_meter.pop(description) for description in average_meter.keys()}
        for description, value in average_values.items():
            self.logger.print("{}:{:.3f}".format(description, value), end=" ")
        self.logger.print("")

        # Logs on wandb
        wandb = self.logger.get_wandb()
        logged_map = {f"{self.logger_prefix}/" + f"{description}": item for description, item in average_values.items()}
        logged_map["step"] = self.trainer.global_step
        wandb.log(logged_map, step=self.trainer.global_step)
        additional_info["step"] = self.trainer.global_step
        wandb.log(additional_info, step=self.trainer.global_step)

    def save_action_videos_from_batch(self, playable_environment_model, batch: Batch, step: int, frames_per_action=15, framerate=10):
        '''

        :param playable_environment_model: The model for the playable environment
        :param environment_model: The model for the environment
        :param batch: The data to reconstruct
        :param step: The current step
        :param frames_per_action: Number of frames to produce for each action
        :param framerate: Framerate for the output video
        :return:
        '''

        # If the model is DataParallel, unwraps it
        if isinstance(playable_environment_model, DataParallel):
            playable_environment_model = playable_environment_model.module

        # Creates a video saver
        video_saver = VideoSaver()

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Gets the number of actions for each object
        actions_count = []
        for dynamic_object_idx in range(dynamic_objects_count):
            object_animation_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            object_animation_model = playable_environment_model.object_animation_models[object_animation_model_idx]
            current_actions_count = object_animation_model.action_network.actions_count
            actions_count.append(current_actions_count)

        max_actions_count = max(actions_count)

        # Performs inference
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple
        image_size = (observations.size(-2), observations.size(-1))

        # Generates each video sequence
        for current_action_idx in range(max_actions_count):
            # Computes names for the output sequences
            base_filename = f"{step:09d}_action_video_{current_action_idx}"
            base_filename = os.path.join(self.config["logging"]["output_images_directory"], base_filename)
            video_filename = base_filename + ".mp4"
            gif_filename = base_filename + ".gif"

            # Initializes the model
            with torch.no_grad():
                current_observation, current_state = playable_environment_model.initialize_interactive_generation(
                    observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity,
                    global_frame_indexes, video_frame_indexes, video_indexes, 0, 0)

            # Generates all observations for the current sequence
            current_observation_idx = 0
            all_observations = []
            all_pil_observations = []
            while True:
                pil_image = transforms.ToPILImage()(current_observation.permute((2, 0, 1)))
                int_observation = (current_observation.cpu().numpy() * 255).astype(np.uint8)
                all_observations.append(int_observation)
                all_pil_observations.append(pil_image)

                # Collects action inputs for the current step
                action_inputs = []
                for current_actions_count in actions_count:
                    # If the current object has a maximum number of actions smaller than another object, clip the action value
                    current_action = min(current_actions_count, current_action_idx)
                    action_inputs.append(current_action)

                # Terminates the current sequence and saves the videos
                if current_observation_idx == frames_per_action:
                    all_observations = np.stack(all_observations, axis=0)
                    video_saver.save_video(all_observations, video_filename, framerate)
                    video_saver.video_to_gif(video_filename, gif_filename, framerate, image_size[-1])

                    # Saves all pil images
                    for image_idx, current_pil_image in enumerate(all_pil_observations):
                        current_pil_image.save(f"{base_filename}_{image_idx:05d}.png")

                    break

                with torch.no_grad():
                    current_observation, current_state = playable_environment_model.generate_next(action_inputs, current_state, image_size, sample_action_variations=False, draw_axes=False)

                # Next observation
                current_observation_idx += 1

    def save_reconstruction_from_batch(self, playable_environment_model, environment_model, batch: Batch, step: int, action_modifiers: List=None):
        '''
        Reconstruct sequences in the current batch of data

        :param playable_environment_model: The model for the playable environment
        :param environment_model: The model for the environment
        :param batch: The data to reconstruct
        :param step: The current step
        :param action_modifiers: list of action modifier objects. A sequence is reconstructed using each of the action modifiers
        :return:
        '''

        # By default use the None action modifier which does not alter the actions
        if action_modifiers is None or len(action_modifiers) == 0:
            action_modifiers=[None]

        for current_action_modifier in action_modifiers:

            # Performs inference
            batch_tuple = batch.to_tuple()
            observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

            # Forwards the batch through the model
            scene_encoding, object_results = playable_environment_model(observations, camera_rotations, camera_translations,
                                                                        focals,
                                                                        bounding_boxes, bounding_boxes_validity,
                                                                        global_frame_indexes, video_frame_indexes,
                                                                        video_indexes,
                                                                        ground_truth_observations=1, shuffle_style=False,
                                                                        action_modifier=current_action_modifier)

            # Extracts the scene encodings
            camera_rotations = scene_encoding["camera_rotations"]
            camera_translations = scene_encoding["camera_translations"]
            focals = scene_encoding["focals"]
            object_rotations = scene_encoding["object_rotation_parameters"]
            object_translations = scene_encoding["object_translation_parameters"]
            object_style = scene_encoding["object_style"]
            object_deformation = scene_encoding["object_deformation"]

            # Computes the number of objects
            static_objects_count = self.object_id_helper.static_objects_count
            dynamic_objects_count = self.object_id_helper.dynamic_objects_count
            objects_count = self.object_id_helper.objects_count

            # Substitutes the values for camera and static object parameters with the ones of the first observation
            camera_rotations = self.broadcast_first_observation(camera_rotations)
            camera_translations = self.broadcast_first_observation(camera_translations)
            focals = self.broadcast_first_observation(focals)
            static_object_rotations = self.broadcast_first_observation(object_rotations[..., :static_objects_count])
            static_object_translations = self.broadcast_first_observation(object_translations[..., :static_objects_count])
            static_object_style = self.broadcast_first_observation(object_style[..., :static_objects_count])
            static_object_deformation = self.broadcast_first_observation(object_deformation[..., :static_objects_count])

            # Computes the values for the reconstructed dynamic objectss
            dynamic_object_rotations, dynamic_object_translations, dynamic_object_style, dynamic_object_deformation = \
                self.compute_dynamic_object_parameters(object_results)

            # Concatenates static and dynamic object reconstructed values together
            object_rotations = torch.cat([static_object_rotations, dynamic_object_rotations], dim=-1)
            object_translations = torch.cat([static_object_translations, dynamic_object_translations], dim=-1)
            object_style = torch.cat([static_object_style, dynamic_object_style], dim=-1)
            object_deformation = torch.cat([static_object_deformation, dynamic_object_deformation], dim=-1)

            # Object is set to always be in the scene
            object_in_scene = torch.ones((object_rotations.size(0), object_rotations.size(1), object_rotations.size(-1)),
                                         dtype=torch.bool, device=object_rotations.device)

            image_size = (observations.size(-2), observations.size(-1))

            render_results = environment_model.render_full_frame_from_scene_encoding(camera_rotations, camera_translations,
                                                                                     focals, image_size, object_rotations,
                                                                                     object_translations, object_style,
                                                                                     object_deformation, object_in_scene,
                                                                                     perturb=False,
                                                                                     samples_per_image_batching=60,
                                                                                     upsample_factor=1.0)

            normalized_observations = ImageHelper.normalize(observations, (-1, +1))
            # Computes prefix to assign to saved images
            images_prefix = "playable_model"
            if current_action_modifier is not None:
                images_prefix = f"{current_action_modifier.name}_{images_prefix}"

            # Gets a textual representation of the selected actions
            action_text = self.get_action_text(object_results)

            reconstructed_bounding_boxes = render_results["reconstructed_bounding_boxes"]
            reconstructed_3d_bounding_boxes = render_results["reconstructed_3d_bounding_boxes"]
            projected_axes = render_results["projected_axes"]
            self.image_helper.save_images_from_results(render_results, step, bounding_boxes=bounding_boxes, reconstructed_bounding_boxes=reconstructed_bounding_boxes, reconstructed_3d_bounding_boxes=reconstructed_3d_bounding_boxes, projected_axes=projected_axes, prefix=images_prefix, ground_truth_observations=normalized_observations, text=action_text)

    def get_action_text(self, object_results: Dict):
        '''
        Extracts a textual descriptions of the actions selected for each frame

        :param object_results: object results output of the playable environment model
        :return: List of size batch_size with lists of size observations_count with a textual description of the actions
                 selected for each frame
        '''

        batch_size = object_results[0]["sampled_actions"].size(0)
        observations_count = object_results[0]["sampled_actions"].size(1) + 1
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Builds the text to impress on the images
        action_text = []
        for batch_idx in range(batch_size):
            current_batch_text = []
            for observation_idx in range(observations_count - 1):
                current_observation_text = ""
                for dynamic_object_idx in range(dynamic_objects_count):
                    current_observation_text += str(object_results[dynamic_object_idx]["sampled_actions"][batch_idx, observation_idx].argmax().item())
                    if dynamic_object_idx != dynamic_objects_count - 1:
                        current_observation_text += ", "
                current_batch_text.append(current_observation_text)
            current_batch_text.append("")  # No action is present for the last image
            action_text.append(current_batch_text)

        return action_text

    def compute_dynamic_object_parameters(self, object_results: Dict) -> Tuple:
        '''
        Computes dynamic object rotations, translations, style and deformation starting from the results output by the
        playable environment model
        :param object_results: object results output of the playable environment model
        :return: (bs, observations_count, 3, dynamic_objects_count) tensor with object rotations
                 (bs, observations_count, 3, dynamic_objects_count) tensor with object translations
                 (bs, observations_count, style_features_count, dynamic_objects_count) tensor with object style features
                 (bs, observations_count, deformation_features_count, dynamic_objects_count) tensor with object deformation features
        '''

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        all_rotations = []
        all_translations = []
        all_style_features = []
        all_deformation_features = []
        # Extracts the reconstructed values for each object
        for dynamic_object_idx in range(dynamic_objects_count):
            current_results = object_results[dynamic_object_idx]

            all_rotations.append(current_results["reconstructed_object_rotations"])
            all_translations.append(current_results["reconstructed_object_translations"])
            all_style_features.append(current_results["reconstructed_object_style"])
            all_deformation_features.append(current_results["reconstructed_object_deformation"])

        # Stacks the values along the object dimension
        all_rotations = torch.stack(all_rotations, dim=-1)
        all_translations = torch.stack(all_translations, dim=-1)
        all_style_features = torch.stack(all_style_features, dim=-1)
        all_deformation_features = torch.stack(all_deformation_features, dim=-1)

        return all_rotations, all_translations, all_style_features, all_deformation_features

    def broadcast_first_observation(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Replaces all the observations in the tensor with the first observation
        :param tensor: (bs, observations_count, ...) tensor
        :return: (bs, observations_count, ...) tensor where all the values in the observations_count dimension are constant
                                               and correspond to the first one in the input tensor
        '''

        first_observation = tensor[:, :1]
        result = first_observation.expand(tensor.size())

        return result


def evaluator(config, trainer: PlayableModelTrainer, dataset: MulticameraVideoDataset, logger, logger_prefix="playable_model_test"):
    return PlayableModelEvaluator(config, trainer, dataset, logger, logger_prefix)
