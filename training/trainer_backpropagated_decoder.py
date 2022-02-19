import re
from typing import Tuple

import re
from typing import Tuple

import torch

from dataset.batching import Batch
from dataset.video_dataset import MulticameraVideoDataset
from training.trainer_backpropagated_autoencoder import TrainerBackpropagatedAutoencoder
from utils.lib_3d.ray_helper import RayHelper
from utils.logger import Logger


class TrainerBackpropagatedDecoder(TrainerBackpropagatedAutoencoder):
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: MulticameraVideoDataset, logger: Logger):
        super(TrainerBackpropagatedDecoder, self).__init__(config, model, dataset, logger)

        if self.patch_size == 0:
            raise Exception("Only the use of patches is supported with the multiresolution backpropagated autoencoder")
        if not self.align_grid:
            raise Exception("Only the use of aligned grids is supported with the multiresolution backpropagated autoencoder")
        if not self.crop_to_patch:
            raise Exception("Backpropagated decoder requires a patch to be rendered, so crop_to_patch must be enabled")
        if self.crop_to_patch and self.patch_size == 0:
            raise Exception("Crop to patch option can be used only if patch size is greater than 0")

    def compute_losses(self, model, batch: Batch) -> Tuple:
        '''
        Computes losses using the full model

        :param model: The network model
        :param batch: Batch of data
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
                  scene_encodings: encoding of the scene as returned by the model
        '''

        if self.global_step < self.frozen_autoencoder_steps:
            model.module.set_autoencoder_frozen(True)
        else:
            model.module.set_autoencoder_frozen(False)

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

        # Forwards the batch through the model
        results = model(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, samples_per_image=self.samples_per_image,
                        perturb=self.perturb, shuffle_style=self.shuffle_style, patch_size=self.patch_size, patch_stride=self.autoencoder_downsample_factor, align_grid=self.align_grid)

        scene_encodings = results["scene_encoding"]

        objects_count = self.object_id_helper.objects_count
        static_object_models_count = self.object_id_helper.static_object_models_count
        static_objects_count = self.object_id_helper.static_objects_count

        # The object is in the scene if it is visible by at least one camera
        object_in_scene, _ = bounding_boxes_validity.max(dim=-2)

        # Extracts the observations that correspond to the sampled positions
        sampled_observations = results["observations"]
        sampled_positions = results["positions"]

        # Extracts bouding boxes
        reconstructed_bounding_boxes = results["reconstructed_bounding_boxes"]
        # The initial static objects are not considered in computation of positional losses
        reconstructed_bounding_boxes = reconstructed_bounding_boxes[..., static_objects_count:]

        # Extract translation parameters
        object_rotation_parameters = results["object_rotation_parameters"]
        object_translation_parameters = results["object_translation_parameters"]

        # Extracts crops and attention maps
        object_attention = results["object_attention"]
        object_crops = results["object_crops"]

        # If cropping is to be used, crop the observations before loss computations
        if self.crop_to_patch:
            observations = RayHelper.sample_original_region_from_patch_samples(observations, sampled_positions, self.autoencoder_downsample_factor)

        # Loss information
        loss_info = {}

        total_loss = torch.zeros((), dtype=torch.float32).cuda()
        for result_type in results.keys():
            # Filter away indesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            cropped_reconstructed_observations = results[result_type]["global"]["reconstructed_observations"]
            reconstructed_encoded_observations = results[result_type]["global"]["integrated_features"]
            integrated_displacements_magnitude = results[result_type]["global"]["integrated_displacements_magnitude"]
            integrated_divergence = results[result_type]["global"]["integrated_divergence"]

            # Saves debug images at regular intervals
            if self.global_step % self.config["training"]["image_save_interval"] == 0:
                self.save_reconstructed_observation(cropped_reconstructed_observations[0, 0, 0])

            # Computes the reconstruction loss
            reconstruction_loss = self.reconstruction_loss(observations, cropped_reconstructed_observations)
            loss_info[f"{result_type}_reconstruction_loss"] = reconstruction_loss.item()
            total_loss += self.config["training"]["loss_weights"]["reconstruction_loss_lambda"] * reconstruction_loss

            displacements_magnitude_loss = integrated_displacements_magnitude.mean()
            divergence_loss_annealing = (1.0 / 100.0) ** (1 - (self.global_step / self.max_steps))
            divergence_loss = integrated_divergence.mean()

            loss_info[f"{result_type}_displacements_magnitude_loss"] = displacements_magnitude_loss.item()
            loss_info[f"{result_type}_divergence_loss"] = divergence_loss.item()
            loss_info[f"{result_type}_divergence_loss_annealing"] = divergence_loss_annealing

            total_loss += self.config["training"]["loss_weights"]["displacements_magnitude_loss_lambda"] * displacements_magnitude_loss
            total_loss += self.config["training"]["loss_weights"]["divergence_loss_lambda"] * divergence_loss_annealing * divergence_loss

            # Computes the perceptual loss if used
            if self.perceptual_loss_lambda > 0.0:

                # Brings reconstructed observations in the range [-1, +1]
                normalized_reconstructed_observations = (cropped_reconstructed_observations - 0.5) / 0.5

                perceptual_loss, perceptual_loss_components = self.perceptual_loss(observations, normalized_reconstructed_observations)
                perceptual_loss_term = self.sum_loss_components(perceptual_loss_components, self.perceptual_loss_lambda)

                # Adds the perceptual loss to the total loss
                # Weights are already accounted for in the perceptual loss term
                total_loss += perceptual_loss_term

                loss_info[f"perceptual_loss"] = perceptual_loss.item()
                for loss_component_idx, current_perceptual_loss_component in enumerate(perceptual_loss_components):
                    loss_info[f"perceptual_loss_{loss_component_idx}"] = current_perceptual_loss_component.item()

            # Computes losses specific to the rendering of each object
            for current_object_key in results[result_type]:
                # Skip global results
                if current_object_key == "global":
                    continue
                object_id = int(re.search(r'\d{0,3}$', current_object_key).group())

                current_results = results[result_type][current_object_key]
                # Computes the head selection loss only if needed
                head_selection_cross_entropy_loss_lambda = self.config["training"]["loss_weights"]["head_selection_cross_entropy_loss_lambda"]
                if head_selection_cross_entropy_loss_lambda > 0:
                    head_selection_logits = current_results["extra_outputs"]["head_selection_logits"]
                    head_selection_cross_entropy_loss = self.head_selection_loss(head_selection_logits, video_indexes)

                    loss_info[f"{result_type}_{current_object_key}_cross_entropy_head_selection_loss"] = head_selection_cross_entropy_loss
                    total_loss += head_selection_cross_entropy_loss_lambda * head_selection_cross_entropy_loss

                # Do not apply the rest of the loss to non moving objects
                if object_id < static_objects_count:
                    continue

                sharpness_loss_annealing = min(1.0, (self.global_step / self.max_steps))

                dynamic_object_id = self.object_id_helper.dynamic_object_idx_by_object_idx(object_id)
                current_bounding_boxes_validity = bounding_boxes_validity[..., dynamic_object_id]

                opacity_loss = self.opacity_loss(current_results["opacity"], current_bounding_boxes_validity)
                sharpness_loss = self.sharpness_loss(current_results["opacity"], current_bounding_boxes_validity)

                loss_info[f"{result_type}_{current_object_key}_opacity_loss"] = opacity_loss.item()
                loss_info[f"{result_type}_{current_object_key}_sharpness_loss"] = sharpness_loss.item()
                loss_info[f"{result_type}_{current_object_key}_sharpness_loss_annealing"] = sharpness_loss_annealing

                total_loss += self.config["training"]["loss_weights"]["opacity_loss_lambda"] * opacity_loss
                total_loss += self.config["training"]["loss_weights"]["sharpness_loss_lambda"] * sharpness_loss_annealing * sharpness_loss

        # Computes losses on attention maps. Computes them only on dynamic objects
        for object_idx in range(static_objects_count, objects_count):
            dynamic_object_id = self.object_id_helper.dynamic_object_idx_by_object_idx(object_id)
            current_bounding_boxes_validity = bounding_boxes_validity[..., dynamic_object_id]

            attention_loss = self.attention_loss(object_attention[object_idx], current_bounding_boxes_validity)

            loss_info[f"object_{object_idx}_attention_loss"] = attention_loss.item()

            total_loss += self.config["training"]["loss_weights"]["attention_loss_lambda"] * attention_loss

        # Computes loss term on bounding boxes
        bounding_box_distance_loss, per_object_bounding_box_distance_loss = self.bounding_box_distance_loss(bounding_boxes.detach(), reconstructed_bounding_boxes, bounding_boxes_validity)
        total_loss += self.config["training"]["loss_weights"]["bounding_box_loss_lambda"] * bounding_box_distance_loss
        loss_info[f"bounding_box_loss"] = bounding_box_distance_loss.item()

        # Logs statistics on bounding boxes
        for object_idx, current_object_loss in enumerate(per_object_bounding_box_distance_loss):
            loss_info[f"object_{object_idx}_bounding_box_loss"] = current_object_loss.item()

        with torch.no_grad():

            # Computes additional statistics on the reconstructed observations
            encoded_observations_l2_norm = torch.sqrt(reconstructed_encoded_observations.pow(2).sum(-1)).mean()
            loss_info[f"encoded_observations_l2_norm"] = encoded_observations_l2_norm.item()

            # Computes statistics on rotation and translation parameters
            for params_tensor, param_type in [(object_rotation_parameters, "rotation"), (object_translation_parameters, "translation")]:

                # Computes statistics for each dynamic object
                objects_count = params_tensor.size(-1)
                dimensions_count = params_tensor.size(-2)
                for object_idx in range(static_objects_count, objects_count):

                    dynamic_object_id = self.object_id_helper.dynamic_object_idx_by_object_idx(object_idx)
                    current_object_in_scene = object_in_scene[..., dynamic_object_id]
                    current_object_parameters = params_tensor[..., object_idx]
                    current_object_parameters = current_object_parameters[current_object_in_scene]

                    object_norm = torch.sqrt(torch.norm(current_object_parameters, dim=-2)).mean(dim=0)
                    object_mean = current_object_parameters.mean(dim=0)
                    object_magnitude = torch.abs(current_object_parameters).mean(dim=0)
                    object_var = current_object_parameters.var(dim=0)

                    loss_info[f"object_{object_idx}_{param_type}_norm"] = object_norm.item()

                    for dimension_idx in range(dimensions_count):
                        loss_info[f"object_{object_idx}_{param_type}_mean_{dimension_idx}"] = object_mean[dimension_idx].item()
                        loss_info[f"object_{object_idx}_{param_type}_magnitude_{dimension_idx}"] = object_magnitude[dimension_idx].item()
                        loss_info[f"object_{object_idx}_{param_type}_var_{dimension_idx}"] = object_var[dimension_idx].item()

        # If available, add to the loss information the ground truth rotation and translation reconstruction errors
        self.add_object_pose_loss_information(loss_info, batch, object_rotation_parameters, object_translation_parameters)

        additional_info = {}
        loss_info["loss"] = total_loss.item()

        return total_loss, loss_info, additional_info, scene_encodings


def trainer(config, model, dataset, logger):
    return TrainerBackpropagatedDecoder(config, model, dataset, logger)

