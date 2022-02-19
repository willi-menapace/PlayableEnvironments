import math
import os
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn, Batch
from dataset.video_dataset import MulticameraVideoDataset
from model.layers.rotation_encoder import RotationEncoder
from model.utils.object_ids_helper import ObjectIDsHelper
from training.losses import SmoothMutualInformationLoss, EntropyLogitLoss, EntropyProbabilityLoss, \
    KLGaussianDivergenceLoss, ACMV
from utils.average_meter import AverageMeter
from utils.lib_3d.transformations_3d import Transformations3D
from utils.logger import Logger
from utils.time_meter import TimeMeter
from utils.torch_time_meter import TorchTimeMeter


class PlayableModelTrainer:
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: MulticameraVideoDataset, logger: Logger):

        self.config = config
        self.dataset = dataset
        self.logger = logger

        self.optimizer = self.get_optimizer(model)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config["playable_model_training"]["lr_gamma"])
        self.lr_decay_iterations = self.config["playable_model_training"]["lr_decay_iterations"]

        # Number of steps between each logging
        self.log_interval_steps = config["playable_model_training"]["log_interval_steps"]

        self.dataloader = DataLoader(dataset, batch_size=self.config["playable_model_training"]["batching"]["batch_size"], drop_last=True,
                                     shuffle=True, collate_fn=single_batch_elements_collate_fn,
                                     num_workers=self.config["playable_model_training"]["batching"]["num_workers"], pin_memory=True)

        self.average_meter = AverageMeter()
        self.time_meter = TimeMeter()
        self.global_step = 0
        self.max_steps = self.config["playable_model_training"]["max_steps"]

        # Observations count annealing parameters
        self.observations_count_start = self.config["playable_model_training"]["batching"]["observations_count_start"]
        self.observations_count_end = self.config["playable_model_training"]["batching"]["observations_count"]
        self.observations_count_steps = self.config["playable_model_training"]["batching"]["observations_count_steps"]

        # Real observations annealing parameters
        self.real_observations_start = self.config["playable_model_training"]["ground_truth_observations_start"]
        self.real_observations_end = self.config["playable_model_training"]["ground_truth_observations_end"]
        self.real_observations_steps = self.config["playable_model_training"]["ground_truth_observations_steps"]

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        self.mutual_information_estimation_alpha = self.config["playable_model_training"]["mutual_information_estimation_alpha"]
        self.mutual_infromation_entropy_lambda = config["playable_model_training"]["mutual_information_entropy_lambda"]

        # Whether to translate movements to the camera space for acmv loss computation
        self.use_camera_relative_acmv = config["playable_model_training"]["use_camera_relative_acmv"]
        self.acmv_rotation_axis = config["playable_model_training"]["acmv_rotation_axis"]

        # Creates for each object its module for the estimation of mutual information
        current_modules = self.create_mutual_information_loss_terms(model)
        self.mutual_information_loss_modules = nn.ModuleList(current_modules).cuda()

        self.entropy_logit_loss = EntropyLogitLoss()
        self.entropy_probability_loss = EntropyProbabilityLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_gaussian_divergence_loss = KLGaussianDivergenceLoss()
        self.acmv_loss = ACMV()

        self.torch_time_meter = TorchTimeMeter()

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def get_optimizer(self, model):
        '''
        Creates the optimizer

        :param model:
        :return:
        '''

        betas = self.config["playable_model_training"]["betas"]

        return torch.optim.Adam(model.non_environment_parameters(), lr=self.config["playable_model_training"]["learning_rate"], weight_decay=self.config["playable_model_training"]["weight_decay"], betas=betas)

    def create_mutual_information_loss_terms(self, model) -> List[nn.Module]:
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        all_modules = []
        for dynamic_object_idx in range(dynamic_objects_count):
            animation_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_action_count = model.object_animation_models[animation_model_idx].actions_count
            all_modules.append(SmoothMutualInformationLoss(current_action_count, self.mutual_information_estimation_alpha))

        return all_modules

    def save_checkpoint(self, model, name=None):
        '''
        Saves the current training state
        :param model: the model to save
        :param name: the name to give to the checkopoint. If None the default name is used
        :return:
        '''

        if name is None:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], f"{name}_.pth.tar")

        torch.save({"model": model.module.state_dict(), "optimizer": self.optimizer.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(), "mi_estimator": self.mutual_information_loss_modules.state_dict(), "step": self.global_step}, filename)

    def load_checkpoint(self, model, name=None):
        """
        Loads the model from a saved state
        :param model: The model to load
        :param name: Name of the checkpoint to use. If None the default name is used
        :return:
        """

        if name is None:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], f"{name}.pth.tar")

        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        loaded_state = torch.load(filename)
        model.load_state_dict(loaded_state["model"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        self.mutual_information_loss_modules.load_state_dict(loaded_state["mi_estimator"])
        self.global_step = loaded_state["step"]

    def get_ground_truth_observations_count(self) -> int:
        '''
        Computes the number of ground truth observations to use for the current training step according to the annealing
        parameters
        :return: number of ground truth observations to use in the training sequence at the current step
        '''

        ground_truth_observations_count = self.real_observations_start - \
                                  (self.real_observations_start - self.real_observations_end) * \
                                  self.global_step / self.real_observations_steps
        ground_truth_observations_count = math.ceil(ground_truth_observations_count)
        ground_truth_observations_count = max(self.real_observations_end, ground_truth_observations_count)

        return ground_truth_observations_count

    def get_observations_count(self):
        '''
        Computes the number of observations to use for the sequence at the current training step according to
        the annealing parameters
        :return: Number of observations to use in each training sequence at the current step
        '''

        observations_count = self.observations_count_start + \
                                  (self.observations_count_end - self.observations_count_start) * \
                                  self.global_step / self.observations_count_steps
        observations_count = math.floor(observations_count)
        observations_count = min(self.observations_count_end, observations_count)

        return observations_count

    def compute_average_centroid_distance(self, centroids: torch.Tensor):
        '''
        Computes the average distance between centroids

        :param centroids: (centroids_count, space_dimensions) tensor with centroids
        :return: Average L2 distance between centroids
        '''

        centroids_count = centroids.size(0)

        centroids_1 = centroids.unsqueeze(0)  # (1, centroids_count, space_dimensions)
        centroids_2 = centroids.unsqueeze(1)  # (centroids_count, 1, space_dimensions)
        centroids_sum = (centroids_1 - centroids_2).pow(2).sum(2).sqrt().sum()
        average_centroid_distance = centroids_sum / (centroids_count * (centroids_count - 1))

        return average_centroid_distance

    def filter_valids(self, tensor: torch.Tensor, sequence_validity: torch.Tensor):
        '''
        Filters the tensor retaining only entries that are flagged as valid

        :param tensor: (bs, observations_count | observations_count-1, ...) tensor with data to filter
        :param sequence_validity: (bs, observations_count) boolean tensor with True in the positions where the sequence if valid
        :return: (elements_count, ...) tensor with filtered data. The first two dimensions are flattened due to filtering
        '''

        observations_count = sequence_validity.size(1)
        tensor_observations_count = tensor.size(1)
        if observations_count != tensor_observations_count:
            if tensor_observations_count == observations_count - 1:
                # Drops the last column from the sequence validity
                sequence_validity = sequence_validity[:, :-1]
            else:
                raise Exception("Tensor must have the same size or exactly one less for the observations_count dimension")

        tensor = tensor[sequence_validity == True]
        return tensor

    def get_rotation_matrix(self, rotations: torch.Tensor, rotation_axis: int) -> torch.Tensor:
        '''

        :param rotations: (..., 3) tensor with rotations
        :return: (..., 3, 3) tensor of rotation matrices around the rotation_axis
        '''

        if rotation_axis == 0:
            rotation_function = Transformations3D.rotation_matrix_x
        elif rotation_axis == 1:
            rotation_function = Transformations3D.rotation_matrix_y
        elif rotation_axis == 2:
            rotation_function = Transformations3D.rotation_matrix_z
        else:
            raise Exception(f"Invalid rotation axis {rotation_axis}")

        # Transforms in a matrix the rotation around the rotation axis
        rotation_matrices = rotation_function(rotations[..., rotation_axis])
        return rotation_matrices

    def get_camera_relative_movements(self, movements: torch.Tensor, camera_rotations: torch.Tensor, rotation_axis: int):
        '''

        :param movements: (bs, observations_count - 1, 3) tensor with object movements expressed in world coordinates
        :param camera_rotations: (bs, observations_count, cameras_count, 3) tensor with camera rotations. Only a single camera must be present
        :param rotation_axis: Rotation axis to align to the camera. Should be the axis normal to the ground
        :return: (bs, observations_count - 1, 3) tensor with object movements expressed relative to the camera. Only a rotation along rotation_axis is applied
        '''

        observations_count = camera_rotations.size(1)
        cameras_count = camera_rotations.size(-2)

        if cameras_count != 1:
            raise Exception(f"A single camera can be used for movement alignment to camera, but {cameras_count} cameras are present")
        if observations_count != movements.size(1) + 1:
            raise Exception(f"The number of observations in the movements should be 1 less than in the cameras, but got respectively {movements.size(1)} and {observations_count}.")
        if rotation_axis is None:
            raise Exception("Rotation axis is None, but a valid axis is required")

        # Removes the last element and removes the camera dimension
        camera_rotations = camera_rotations[:, :-1, 0]
        # (bs, observations_count - 1, 3, 3)
        rotation_matrices = self.get_rotation_matrix(-camera_rotations, rotation_axis)  # Rotations are negated to get teh transformations from world to camera

        # (bs, observations_count - 1, 3, 1) Translations expressed in the object coordinate system
        movements = movements.unsqueeze(-1)
        # (bs, observations_count - 1, 3). Translations expresesd in the world coordinate system
        camera_relative_movements = torch.matmul(rotation_matrices, movements).squeeze(-1)

        return camera_relative_movements

    def compute_losses(self, model, batch: Batch, ground_truth_observations_count_override: int=0, forward_mode="vanilla") -> Tuple:
        '''
        Computes losses using the full model

        :param model: The network model
        :param batch: Batch of data
        :param ground_truth_observations_count_override: If = 0, let the encoder schedule the number of ground truth observations to use,
                                                         If > 0, overrides the number of ground truth observations to use
        :param forward_mode: which mode to use for the forward pass
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
                  nonlogged_additional_info: Dict with additional nonloggable information
        '''

        # Extracts the current batch
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple
        observations_count = observations.size(1)

        # Ground truth observations to use at the current step inferred by the trainer
        if ground_truth_observations_count_override <= 0:
            ground_truth_observations_count = self.get_ground_truth_observations_count()
        # Ground truth observations to use manually specified
        else:
            ground_truth_observations_count = ground_truth_observations_count_override

        # Since the annealing of the ground truth observations to use may produce a number greater than the number of
        # observations in the sequence, we cap it to the maximum value for the current sequence length
        if ground_truth_observations_count >= observations_count:
            ground_truth_observations_count = observations_count - 1

        # Forwards the batch through the model
        scene_encoding, object_results = model(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, ground_truth_observations_count, shuffle_style=False, mode=forward_mode)

        total_loss, loss_info, additional_info, nonlogged_additional_info = self.compute_losses_from_forward_results(scene_encoding, object_results)

        return total_loss, loss_info, additional_info, nonlogged_additional_info

    def compute_losses_from_forward_results(self, *forward_results):
        '''
        Computes losses using the full model

        :param forward_results: results of calling forward on the model for which to compute the losses
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
                  nonlogged_additional_info: Dict with additional nonloggable information
        '''

        scene_encoding, object_results = forward_results

        # Extracts the scene encodings
        camera_rotations = scene_encoding["camera_rotations"]
        camera_translations = scene_encoding["camera_translations"]
        focals = scene_encoding["focals"]
        object_rotations = scene_encoding["object_rotation_parameters"]
        object_translations = scene_encoding["object_translation_parameters"]
        object_style = scene_encoding["object_style"]
        object_deformation = scene_encoding["object_deformation"]

        objects_count = self.object_id_helper.objects_count
        static_object_models_count = self.object_id_helper.static_object_models_count
        static_objects_count = self.object_id_helper.static_objects_count
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Loss information
        loss_info = {}
        total_loss = torch.zeros((), dtype=torch.float32).cuda()

        # Computes losses for each dynamic object
        for dynamic_object_idx in range(dynamic_objects_count):
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_prefix = f"object_{object_idx}_"

            # Gets animation results for the current object
            current_object_results = object_results[dynamic_object_idx]

            sequence_validity = current_object_results["sequence_validity"]
            #sequence_validity[:] = False
            #TranslationsPlotter.plot_translations(object_translations[..., object_idx], current_object_results["reconstructed_object_translations"], 2, "results/reconstructed_translations", prefix=f'{dynamic_object_idx}')

            # Unpacks results filtering for the valid entries only
            current_object_rotations = self.filter_valids(object_rotations[..., object_idx], sequence_validity)
            current_object_translations = self.filter_valids(object_translations[..., object_idx], sequence_validity)
            current_object_style = self.filter_valids(object_style[..., object_idx], sequence_validity)
            current_object_deformation = self.filter_valids(object_deformation[..., object_idx], sequence_validity)

            current_object_reconstructed_rotations = self.filter_valids(current_object_results["reconstructed_object_rotations"], sequence_validity)
            current_object_reconstructed_translations = self.filter_valids(current_object_results["reconstructed_object_translations"], sequence_validity)
            current_object_reconstructed_style = self.filter_valids(current_object_results["reconstructed_object_style"], sequence_validity)
            current_object_reconstructed_deformations = self.filter_valids(current_object_results["reconstructed_object_deformation"], sequence_validity)

            sampled_actions = self.filter_valids(current_object_results["sampled_actions"], sequence_validity)
            action_logits = self.filter_valids(current_object_results["action_logits"], sequence_validity)
            action_directions_distribution = self.filter_valids(current_object_results["action_directions_distribution"], sequence_validity)
            action_states_distribution = self.filter_valids(current_object_results["action_states_distribution"], sequence_validity)
            action_variations = self.filter_valids(current_object_results["action_variations"], sequence_validity)

            reconstructed_action_logits = self.filter_valids(current_object_results["reconstructed_action_logits"], sequence_validity)
            reconstructed_action_directions_distribution = self.filter_valids(current_object_results["reconstructed_action_directions_distribution"], sequence_validity)
            reconstructed_action_states_distribution = self.filter_valids(current_object_results["reconstructed_action_states_distribution"], sequence_validity)

            estimated_action_centroids = current_object_results["estimated_action_centroids"]

            # Computes action entropy losses
            entropy_loss = self.entropy_logit_loss(action_logits)

            # Computes KL loss for the action directions
            action_directions_kl_divergence_loss = self.kl_gaussian_divergence_loss(action_directions_distribution)

            # Computes MI loss
            current_mutual_information_loss = self.mutual_information_loss_modules[dynamic_object_idx]
            action_mutual_information_loss = current_mutual_information_loss(torch.softmax(action_logits, dim=-1),
                                                                             torch.softmax(reconstructed_action_logits, dim=-1),
                                                                             lamb=self.mutual_infromation_entropy_lambda)

            # Computes reconstruction losses for the state space
            # The loss on rotations is computed in the encoded (sin, cos) space
            current_object_encoded_reconstructed_rotations = RotationEncoder.encode(current_object_reconstructed_rotations, dim=-1)
            current_object_encoded_rotations = RotationEncoder.encode(current_object_rotations, dim=-1)
            rotations_reconstruction_loss = self.mse_loss(current_object_encoded_reconstructed_rotations, current_object_encoded_rotations)
            translations_reconstruction_loss = self.mse_loss(current_object_reconstructed_translations, current_object_translations)
            style_reconstruction_loss = self.mse_loss(current_object_reconstructed_style, current_object_style)
            deformation_reconstruction_loss = self.mse_loss(current_object_reconstructed_deformations, current_object_deformation)

            # Additional debug information not used for backpropagation
            with torch.no_grad():
                samples_entropy = self.entropy_probability_loss(sampled_actions)
                action_ditribution_entropy = self.entropy_probability_loss(sampled_actions.mean(dim=0, keepdim=True))
                action_directions_mean_magnitude = torch.mean(torch.abs(action_directions_distribution[:, 0])).item()  # Compute magnitude of the mean
                action_directions_variance = torch.mean(torch.abs(torch.exp(action_directions_distribution[:, 1]))).item()  # Compute magnitude of the variance
                reconstructed_action_directions_mean_magnitude = torch.mean(torch.abs(reconstructed_action_directions_distribution[:, 0])).item()  # Compute magnitude of the mean
                reconstructed_action_directions_variance = torch.mean(torch.abs(torch.exp(reconstructed_action_directions_distribution[:, 1]))).item()  # Compute magnitude of the variance
                action_directions_reconstruction_error = torch.mean((reconstructed_action_directions_distribution[:, 0] - action_directions_distribution[:, 0]).pow(2)).item()  # Compute differences of the mean
                reconstructed_action_directions_kl_divergence_loss = self.kl_gaussian_divergence_loss(reconstructed_action_directions_distribution)
                centroids_mean_magnitude = torch.mean(torch.abs(estimated_action_centroids)).item()
                average_centroids_distance = self.compute_average_centroid_distance(estimated_action_centroids).item()
                average_action_variations_norm_l2 = action_variations.pow(2).sum(-1).sqrt().mean().item()
                action_variations_mean = action_variations.mean().item()

            # Updates the total loss with the statistics for the current object
            current_object_loss = \
                 self.config["playable_model_training"]["loss_weights"]["rotations_rec_lambda"] * rotations_reconstruction_loss + \
                 self.config["playable_model_training"]["loss_weights"]["translations_rec_lambda"] * translations_reconstruction_loss + \
                 self.config["playable_model_training"]["loss_weights"]["style_rec_lambda"] * style_reconstruction_loss + \
                 self.config["playable_model_training"]["loss_weights"]["deformation_rec_lambda"] * deformation_reconstruction_loss + \
                 self.config["playable_model_training"]["loss_weights"]["entropy_lambda"] * entropy_loss + \
                 self.config["playable_model_training"]["loss_weights"]["action_directions_kl_lambda"] * action_directions_kl_divergence_loss + \
                 self.config["playable_model_training"]["loss_weights"]["action_mutual_information_lambda"] * action_mutual_information_loss

            # If required compute the acmv loss
            acmv_lambda = self.config["playable_model_training"]["loss_weights"]["acmv_lambda"]
            if acmv_lambda > 0.0:
                current_acmv_translations = object_translations[..., object_idx][:, :-1]  # Takes the translations excluding the last one
                next_acmv_translations = object_translations[..., object_idx][:, 1:]  # Takes the translations excluding the first one
                acmv_sequence_validity = sequence_validity[:, 1:]  # A position is valid if the current and the next are valid, to take sequence validity of the next observation
                acmv_movements = next_acmv_translations - current_acmv_translations

                # If required, expresses movements in a space oriented in the direction of the camera
                if self.use_camera_relative_acmv:
                    acmv_movements = self.get_camera_relative_movements(acmv_movements, camera_rotations, self.acmv_rotation_axis)

                # Computes the probability distribution of actions
                acmv_actions = current_object_results["action_logits"]
                acmv_actions = torch.softmax(acmv_actions, dim=-1)

                acmv_movements = self.filter_valids(acmv_movements, acmv_sequence_validity)
                acmv_actions = self.filter_valids(acmv_actions, acmv_sequence_validity)

                acmv_loss = self.acmv_loss(acmv_movements, acmv_actions)
                current_object_loss += acmv_lambda * acmv_loss
                loss_info[current_prefix + "acmv_loss"] = acmv_loss.item()

            total_loss += current_object_loss

            # Logs loss information used for backpropagation
            loss_info[current_prefix + "loss"] = current_object_loss.item()
            loss_info[current_prefix + "rotations_reconstruction_loss"] = rotations_reconstruction_loss.item()
            loss_info[current_prefix + "translations_reconstruction_loss"] = translations_reconstruction_loss.item()
            loss_info[current_prefix + "style_reconstruction_loss"] = style_reconstruction_loss.item()
            loss_info[current_prefix + "deformation_reconstruction_loss"] = deformation_reconstruction_loss.item()
            loss_info[current_prefix + "entropy_loss"] = entropy_loss.item()
            loss_info[current_prefix + "action_directions_kl_divergence_loss"] = action_directions_kl_divergence_loss.item()
            loss_info[current_prefix + "action_mutual_information_loss"] = action_mutual_information_loss.item()
            # Logs additional loss information
            loss_info[current_prefix + "samples_entropy"] = samples_entropy.item()
            loss_info[current_prefix + "action_ditribution_entropy"] = action_ditribution_entropy.item()
            loss_info[current_prefix + "action_directions_mean_magnitude"] = action_directions_mean_magnitude
            loss_info[current_prefix + "action_directions_variance"] = action_directions_variance
            loss_info[current_prefix + "reconstructed_action_directions_mean_magnitude"] = reconstructed_action_directions_mean_magnitude
            loss_info[current_prefix + "reconstructed_action_directions_variance"] = reconstructed_action_directions_variance
            loss_info[current_prefix + "action_directions_reconstruction_error"] = action_directions_reconstruction_error
            loss_info[current_prefix + "reconstructed_action_directions_kl_divergence_loss"] = reconstructed_action_directions_kl_divergence_loss.item()
            loss_info[current_prefix + "centroids_mean_magnitude"] = centroids_mean_magnitude
            loss_info[current_prefix + "average_centroids_distance"] = average_centroids_distance
            loss_info[current_prefix + "average_action_variations_norm_l2"] = average_action_variations_norm_l2
            loss_info[current_prefix + "action_variations_mean"] = action_variations_mean

        additional_info = {}
        nonlogged_additional_info = {
            "scene_encoding": scene_encoding,
            "object_results": object_results,
        }

        return total_loss, loss_info, additional_info, nonlogged_additional_info

    def optimization_step(self, model, batch: Batch):
        '''
        Performs an optimization step
        :param model: the model
        :param batch: batch of data
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        # Computes losses
        loss, loss_info, additional_info, nonlogged_additional_info = self.compute_losses(model, batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, loss_info, additional_info

    def log_if_possible(self, additional_info: Dict):
        '''
        Logs the entries in the average meter along with the additional loss information.
        Logging is performed only after log_interval_steps steps from the previous logging

        :param additional_info: dictionary with additional information to log
        :return:
        '''

        # Logs information at regular intervals
        if (self.global_step - 1) % self.log_interval_steps == 0:

            self.logger.print(f'step: {self.global_step}/{self.max_steps}', end=" ")

            average_time = self.time_meter.get_average_time()
            iterations_per_second = 1 / average_time
            self.logger.print(f'ips: {iterations_per_second:.3f}', end=" ")

            average_values = {description: self.average_meter.pop(description) for description in self.average_meter.keys()}
            for description, value in average_values.items():
                self.logger.print("{}:{:.3f}".format(description, value), end=" ")

            current_lr = self._get_current_lr()
            self.logger.print('lr: %.4f' % (current_lr))
            self.torch_time_meter.print_summary()

            # Logs on wandb
            wandb = self.logger.get_wandb()
            logged_map = {"playable_model_train/" + description: item for description, item in average_values.items()}
            logged_map["ips"] = iterations_per_second
            logged_map["step"] = self.global_step
            logged_map["playable_model_train/lr"] = current_lr
            wandb.log(logged_map, step=self.global_step)
            additional_info["step"] = self.global_step
            wandb.log(additional_info, step=self.global_step)

    def lr_scheduler_step(self):

        # Triggers update of the learning rate
        if self.global_step % self.lr_decay_iterations == 0:
            self.lr_scheduler.step()
            self.logger.print(f"- Learning rate updated to {self.lr_scheduler.get_lr()}")

    def train_epoch(self, model):

        self.logger.print(f'== Train [{self.global_step}] ==')

        # Computes the number of observations to use in the current epoch
        observations_count = self.get_observations_count()
        # Modifies the number of observations to return before instantiating the dataloader
        self.dataset.set_observations_count(observations_count)

        self.torch_time_meter.start("load_batch")

        # Number of training steps performed in this epoch
        performed_steps = 0
        for step, batch in enumerate(self.dataloader):
            # If the maximum number of training steps per epoch is exceeded, we interrupt the epoch
            if performed_steps > self.config["training"]["max_steps_per_epoch"]:
                break

            self.global_step += 1
            performed_steps += 1

            # If there is a change in the number of observations to use, we interrupt the epoch
            current_observations_count = self.get_observations_count()
            if current_observations_count != observations_count:
                break

            # Starts times
            self.time_meter.start()
            self.torch_time_meter.end("load_batch")
            self.torch_time_meter.start("optimization_step")

            # Computes losses and performs update
            loss, loss_info, additional_info = self.optimization_step(model, batch)

            # Triggers update of the learning rate
            self.lr_scheduler_step()

            # Stops times
            self.time_meter.end()
            self.torch_time_meter.end("optimization_step")
            self.torch_time_meter.start("logging")

            # Accumulates loss information
            loss_info["loss"] = loss.item()
            self.average_meter.add(loss_info)

            # Performs logging
            self.log_if_possible(additional_info)

            self.torch_time_meter.end("logging")
            self.torch_time_meter.start("load_batch")


def trainer(config, model, dataset, logger):
    return PlayableModelTrainer(config, model, dataset, logger)

