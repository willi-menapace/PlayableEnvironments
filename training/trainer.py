import collections
import os
import re
from typing import Tuple, List, Union, Dict

import torch
from torch.optim import Optimizer
from torch.profiler import record_function
from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn, Batch
from dataset.video_dataset import MulticameraVideoDataset
from model.utils.object_ids_helper import ObjectIDsHelper
from training.losses import ReconstructionLoss, RayObjectDistanceLoss, BoundingBoxDistanceLoss, OpacityLoss, \
    AttentionLoss, SharpnessLoss, ParallelPerceptualLoss, HeadSelectionLoss, PoseConsistencyLoss, \
    KeypointConsistencyLoss, KeypointOpacityLoss
from utils.average_meter import AverageMeter
from utils.logger import Logger
from utils.time_meter import TimeMeter
from utils.torch_time_meter import TorchTimeMeter


class Trainer:
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: MulticameraVideoDataset, logger: Logger):

        self.config = config
        self.dataset = dataset
        self.logger = logger

        self.optimizer = self.get_optimizer(model)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config["training"]["lr_gamma"])
        self.lr_decay_iterations = self.config["training"]["lr_decay_iterations"]

        self.camera_parameters_optimizer = torch.optim.Adam(model.get_camera_offsets_parameters(), lr=config["training"]["camera_parameters_learning_rate"])

        # Number of steps between each logging
        self.log_interval_steps = config["training"]["log_interval_steps"]

        # Number of preceptual features to use in the computation of the loss
        self.perceptual_features_count = config["training"]["perceptual_features"]
        # Weights to use for the perceptual loss
        self.perceptual_loss_lambda = self.config["training"]["loss_weights"]["perceptual_loss_lambda"]
        # Number of rays to sample for each image. Not used if patches_per_image > 0
        self.samples_per_image = config["training"]["samples_per_image"]
        # Number of rays to sample for each image during pose consistency
        self.pose_consistency_samples_per_image = config["training"]["pose_consistency_samples_per_image"]
        # Number of rays to sample for each image during keypoint consistency
        self.keypoint_consistency_samples_per_image = config["training"]["keypoint_consistency_samples_per_image"]
        # Whether to apply perturnations to samples
        self.perturb = config["training"]["perturb"]
        # Whether to shuffle style codes of observations at different temporal points
        self.shuffle_style = config["training"]["shuffle_style"]
        # Whether to use the pose consistency loss
        self.use_pose_consistency = self.config["training"]["loss_weights"]["pose_consistency_loss_lambda"] > 0.0
        # Whether to use the keypoint consistency loss
        self.use_keypoint_consistency = self.config["training"]["loss_weights"]["keypoint_consistency_loss_lambda"] > 0.0
        # Whether to use radial weights for the resonstruction losses
        self.use_radial_weights = self.config["training"]["loss_weights"]["use_radial_weights"]

        # Initializes losses
        self.reconstruction_loss = ReconstructionLoss()
        self.ray_object_distance_loss = RayObjectDistanceLoss()
        self.bounding_box_distance_loss = BoundingBoxDistanceLoss()
        self.opacity_loss = OpacityLoss()
        self.attention_loss = AttentionLoss()
        self.sharpness_loss = SharpnessLoss(config["training"]["loss_weights"]["sharpness_loss_mean"], config["training"]["loss_weights"]["sharpness_loss_std"])
        self.head_selection_loss = HeadSelectionLoss()
        self.pose_consistency_loss = PoseConsistencyLoss()
        self.keypoint_consistency_loss = KeypointConsistencyLoss(config["training"]["loss_weights"]["keypoint_consistency_loss_threshold"])
        self.keypoint_opacity_loss = KeypointOpacityLoss(config["training"]["loss_weights"]["keypoint_opacity_loss_threshold"])

        # Instantiates the perceptual loss if used
        if self.perceptual_loss_lambda > 0.0:
            self.perceptual_loss = ParallelPerceptualLoss(self.perceptual_features_count, use_radial_weights=self.use_radial_weights)

        self.dataloader = DataLoader(dataset, batch_size=self.config["training"]["batching"]["batch_size"], drop_last=True,
                                     shuffle=True, collate_fn=single_batch_elements_collate_fn,
                                     num_workers=self.config["training"]["batching"]["num_workers"], pin_memory=True)

        self.average_meter = AverageMeter()
        self.time_meter = TimeMeter()
        self.torch_time_meter = TorchTimeMeter()
        self.global_step = 0
        self.max_steps = self.config["training"]["max_steps"]

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        # If profiling is disabled use infinite profiler waiting so that it is never triggered
        self.profiler_wait = 2
        if not self.config["training"]["enable_profiling"]:
            self.profiler_wait = 1000000000

    def get_optimizer(self, model):

        if not "encoder_learning_rate" in self.config["training"]:
            # Gathers all the parameters apart the ones for the camera pose optimization
            adam_parameters = list(model.get_main_parameters())
            adam_parameters.extend(model.get_object_encoder_parameters())
            optimizer = torch.optim.Adam(adam_parameters, lr=self.config["training"]["learning_rate"], weight_decay=self.config["training"]["weight_decay"])
        else:
            raise NotImplementedError("Must also consider the other categories of parameters that are not included in the main parameters")

            first_param_group = {"params": model.get_main_parameters()}
            second_param_group = {
                "params": model.get_object_encoder_parameters(),
                "lr": self.config["training"]["encoder_learning_rate"],
                "weight_decay": self.config["training"]["encoder_weight_decay"]
            }
            optimizer = torch.optim.Adam([first_param_group, second_param_group], lr=self.config["training"]["learning_rate"], weight_decay=self.config["training"]["weight_decay"])

        return optimizer

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def add_noise(input: torch.Tensor, noise_range: List[float]) -> torch.Tensor:
        '''
        Adds noise to the input sampled from a uniform distribution in the specified range
        :param input: input tensor to which apply noise
        :param noise_range: range of the noise
        :return: input tensor with noise applied
        '''

        noise = torch.rand_like(input) * (noise_range[1] - noise_range[0]) + noise_range[0]
        return input + noise

    def save_checkpoint(self, model, name=None):
        '''
        Saves the current training state
        :param model: the model to save
        :param name: the name to give to the checkopoint. If None the default name is used
        :return:
        '''

        if name is None:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], f"{name}_.pth.tar")

        torch.save({"model": model.module.state_dict(), "optimizer": self.optimizer.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(), "camera_parameters_optimizer": self.camera_parameters_optimizer.state_dict(), "step": self.global_step}, filename)

    def load_checkpoint(self, model, name=None):
        """
        Loads the model from a saved state
        :param model: The model to load
        :param name: Name of the checkpoint to use. If None the default name is used
        :return:
        """

        if name is None:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], f"{name}.pth.tar")

        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        loaded_state = torch.load(filename)
        model.load_state_dict(loaded_state["model"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        self.camera_parameters_optimizer.load_state_dict(loaded_state["camera_parameters_optimizer"])
        self.global_step = loaded_state["step"]

    def tensor_to_patches(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Transforms an output tensor into images patches

        :param tensor: (..., patches_per_image * patch_size ^ 2, features_count)
        :return:
        '''

        initial_dimensions = list(tensor.size())[:-2]
        features_count = tensor.size(-1)
        final_dimensions = initial_dimensions + [self.patches_per_image, self.patch_size, self.patch_size, features_count]
        tensor = torch.reshape(tensor, final_dimensions)
        # Transforms the tensor to CHW format
        tensor = torch.moveaxis(tensor, -1, -3)
        return tensor

    def sum_loss_components(self, components: List[torch.Tensor], weights: Union[List[float], float]) -> List[torch.Tensor]:
        '''
        Produces the weighted sum of the loss components

        :param components: List of scalar tensors
        :param weights: List of weights of the same length of components, or single weight to apply to each component
        :return: Weighted sum of the components
        '''

        components_count = len(components)

        # If the weight is a scalar, broadcast it
        if not isinstance(weights, collections.Sequence):
            weights = [weights] * components_count

        total_sum = components[0] * 0.0
        for current_component, current_weight in zip(components, weights):
            total_sum += current_component * current_weight

        return total_sum

    def compute_pose_consistency_loss(self, model, batch: Batch, scene_encodings: Dict, detach_scene_encodings: bool=True) -> Tuple:
        '''
        Computes the pose consistency loss using the full model

        :param model: The network model
        :param batch: Batch of data
        :param detach_scene_encodings: If true detaches the scene encodings.
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

        if not batch.has_flow():
            raise Exception("Pose consistency loss computation was requested, but optical flow is not present in the dataset")

        optical_flow = batch.optical_flows

        # Gets parameters from the previous forward pass
        object_style = scene_encodings["object_style"]
        object_deformation = scene_encodings["object_deformation"]
        object_rotation_parameters = scene_encodings["object_rotation_parameters"]
        object_translation_parameters = scene_encodings["object_translation_parameters"]

        # Detaches the scene encodings if requested. Could be required if a preceding backward pass freed the computational graph
        if detach_scene_encodings:
            object_style = object_style.detach()
            object_deformation = object_deformation.detach()
            object_rotation_parameters = object_rotation_parameters.detach()
            object_translation_parameters = object_translation_parameters.detach()

        # Forwards the batch through the model
        results = model(optical_flow, camera_rotations, camera_translations, focals, bounding_boxes,
                        bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes,
                        object_style, object_deformation, object_rotation_parameters, object_translation_parameters,
                        samples_per_image=self.pose_consistency_samples_per_image, perturb=self.perturb, mode="pose_consistency")

        # Loss information
        loss_info = {}

        total_loss = torch.zeros((), dtype=torch.float32).cuda()
        for result_type in results.keys():
            # Filter away undesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            # Computes the loss for each object
            for current_object_key in results[result_type]:
                # Skip global results
                if current_object_key == "global":
                    continue
                dynamic_object_id = int(re.search(r'\d{0,3}$', current_object_key).group())

                current_results = results[result_type][current_object_key]

                current_bounding_boxes_validity = bounding_boxes_validity[..., dynamic_object_id]

                pose_consistency_loss = self.pose_consistency_loss(*current_results, current_bounding_boxes_validity)

                loss_info[f"{result_type}_{current_object_key}_pose_consistency_loss"] = pose_consistency_loss.item()
                total_loss += self.config["training"]["loss_weights"]["pose_consistency_loss_lambda"] * pose_consistency_loss

        additional_info = {}
        loss_info["pose_consistency_loss"] = total_loss.item()

        return total_loss, loss_info, additional_info

    def compute_keypoint_consistency_loss(self, model, batch: Batch, scene_encodings: Dict, detach_scene_encodings: bool=True) -> Tuple:
        '''
        Computes the pose consistency loss using the full model

        :param model: The network model
        :param batch: Batch of data
        :param detach_scene_encodings: If true detaches the scene encodings.
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

        if not batch.has_keypoints():
            raise Exception("Keypoint consistency loss computation was requested, but keypoints are not present in the dataset")

        # Gets keypoints and their validity
        keypoints, keypoints_validity = batch.to_keypoints_typle()

        # Gets parameters from the previous forward pass
        object_style = scene_encodings["object_style"]
        object_deformation = scene_encodings["object_deformation"]
        object_rotation_parameters = scene_encodings["object_rotation_parameters"]
        object_translation_parameters = scene_encodings["object_translation_parameters"]

        # Detaches the scene encodings if requested. Could be required if a preceding backward pass freed the computational graph
        if detach_scene_encodings:
            object_style = object_style.detach()
            object_deformation = object_deformation.detach()
            object_rotation_parameters = object_rotation_parameters.detach()
            object_translation_parameters = object_translation_parameters.detach()

        # Forwards the batch through the model
        results = model(observations, camera_rotations, camera_translations, focals, bounding_boxes,
                        bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes,
                        object_style, object_deformation, object_rotation_parameters, object_translation_parameters,
                        keypoints, keypoints_validity,
                        max_samples_per_image=self.keypoint_consistency_samples_per_image, perturb=self.perturb, mode="keypoint_consistency")

        # Loss information
        loss_info = {}

        total_loss = torch.zeros((), dtype=torch.float32).cuda()
        for result_type in results.keys():
            # Filter away undesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            # Computes the loss for each object
            for current_object_key in results[result_type]:
                # Skip global results
                if current_object_key == "global":
                    continue
                dynamic_object_id = int(re.search(r'\d{0,3}$', current_object_key).group())

                current_results = results[result_type][current_object_key]

                current_expected_positions, current_keypoints_scores, current_opacities, current_keypoints = current_results

                keypoint_consistency_loss = self.keypoint_consistency_loss(current_expected_positions, current_keypoints_scores)

                loss_info[f"{result_type}_{current_object_key}_keypoint_consistency_loss"] = keypoint_consistency_loss.item()
                total_loss += self.config["training"]["loss_weights"]["keypoint_consistency_loss_lambda"] * keypoint_consistency_loss

                # Computes opacity loss and annealing factor
                keypoint_opacity_loss = self.keypoint_opacity_loss(current_opacities, current_keypoints_scores)
                keypoint_opacity_loss_annealing = 1.0
                if self.config["training"]["loss_weights"]["keypoint_opacity_loss_max_steps"] > 0:
                    keypoint_opacity_loss_annealing = max(0.0, 1 - self.global_step / self.config["training"]["loss_weights"]["keypoint_opacity_loss_max_steps"])

                loss_info[f"{result_type}_{current_object_key}_keypoint_opacity_loss"] = keypoint_opacity_loss.item()
                loss_info[f"{result_type}_{current_object_key}_keypoint_opacity_loss_annealing"] = keypoint_opacity_loss_annealing
                total_loss += self.config["training"]["loss_weights"]["keypoint_opacity_loss_lambda"] * keypoint_opacity_loss_annealing * keypoint_opacity_loss

                #print("Saving keypoints images for debug")
                #KeypointsDrawer.draw_keypoints(observations, current_keypoints, current_keypoints_scores, current_expected_positions, "results/keypoints_106")

        additional_info = {}
        loss_info["keypoint_consistency_loss"] = total_loss.item()

        return total_loss, loss_info, additional_info

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

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

        # Forwards the batch through the model
        results = model(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, samples_per_image=self.samples_per_image,
                        perturb=self.perturb, shuffle_style=self.shuffle_style)

        scene_encodings = results["scene_encoding"]

        objects_count = self.object_id_helper.objects_count
        static_object_models_count = self.object_id_helper.static_object_models_count
        static_objects_count = self.object_id_helper.static_objects_count

        # The object is in the scene if it is visible by at least one camera
        object_in_scene, _ = bounding_boxes_validity.max(dim=-2)

        sampled_observations = results["observations"]
        ray_object_distances = results["ray_object_distances"]
        # The initial static objects are not considered in computation of positional losses
        ray_object_distances = ray_object_distances[..., static_objects_count:]

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

        # Loss information
        loss_info = {}

        total_loss = torch.zeros((), dtype=torch.float32).cuda()
        for result_type in results.keys():
            # Filter away indesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            reconstructed_observations = results[result_type]["global"]["integrated_features"]
            integrated_displacements_magnitude = results[result_type]["global"]["integrated_displacements_magnitude"]
            integrated_divergence = results[result_type]["global"]["integrated_divergence"]

            reconstruction_loss = self.reconstruction_loss(sampled_observations, reconstructed_observations)
            # This loss should automatically account for the visibility of the different objects, so validity is not considered
            ray_object_distance_loss = self.ray_object_distance_loss(sampled_observations, reconstructed_observations, ray_object_distances)
            displacements_magnitude_loss = integrated_displacements_magnitude.mean()
            divergence_loss_annealing = (1.0 / 100.0) ** (1 - (self.global_step / self.max_steps))
            divergence_loss = integrated_divergence.mean()

            loss_info[f"{result_type}_reconstruction_loss"] = reconstruction_loss.item()
            loss_info[f"{result_type}_ray_object_distance_loss"] = ray_object_distance_loss.item()
            loss_info[f"{result_type}_displacements_magnitude_loss"] = displacements_magnitude_loss.item()
            loss_info[f"{result_type}_divergence_loss"] = divergence_loss.item()
            loss_info[f"{result_type}_divergence_loss_annealing"] = divergence_loss_annealing

            total_loss += self.config["training"]["loss_weights"]["reconstruction_loss_lambda"] * reconstruction_loss
            total_loss += self.config["training"]["loss_weights"]["ray_object_distance_loss_lambda"] * ray_object_distance_loss
            total_loss += self.config["training"]["loss_weights"]["displacements_magnitude_loss_lambda"] * displacements_magnitude_loss
            total_loss += self.config["training"]["loss_weights"]["divergence_loss_lambda"] * divergence_loss_annealing * divergence_loss

            # Computes losses specific to the rendering of each object
            for current_object_key in results[result_type]:
                # Skip global results
                if current_object_key == "global":
                    continue
                object_id = int(re.search(r'\d{0,3}$', current_object_key).group())

                current_results = results[result_type][current_object_key]
                # Computes the head selection loss only if needed
                head_selection_cross_entropy_loss_lambda = self.config["training"]["loss_weights"][
                    "head_selection_cross_entropy_loss_lambda"]
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

    def add_object_pose_loss_information(self, loss_info: Dict, batch: Batch, inferred_object_rotation, inferred_object_translation, rotation_axis: int=1):
        '''
        If ground truth object pose information is present in the batch, computes information about rotation and translation reconstruction quality

        :param loss_info: Dictionary with information about the loss
        :param batch: batch from which to retrieve ground truth object pose information
        :param inferred_object_rotation: (..., 3, objects_count) tensor with object rotations
        :param inferred_object_translation: (..., 3, objects_count) tensor with object translations
        :param rotation_axis: Axis to use to compute rotation statistics
        :return:
        '''

        # If ground truth information is not present don't do anything
        if not batch.has_object_poses():
            return

        with torch.no_grad():
            ground_truth_rotations, ground_truth_translations = batch.to_object_poses_tuple()

            inferred_dynamic_objects_count = inferred_object_rotation.size(-1) - self.object_id_helper.static_objects_count
            gt_dynamic_objects_count = ground_truth_rotations.size(-1)

            if inferred_dynamic_objects_count != gt_dynamic_objects_count:
                print(f"Warning: the number of ground truth dynamic objects ({gt_dynamic_objects_count}) differs from the number of inferred dynamic objects ({inferred_dynamic_objects_count})\n"
                      f"If this is intentional (eg. running baseline models) ignore this warning.")
                return

            # Selects only dynamic objects and the rotation axis
            dynamic_objects_count = ground_truth_rotations.size(-1)
            ground_truth_rotations = ground_truth_rotations[..., rotation_axis, :]
            inferred_object_rotation = inferred_object_rotation[..., rotation_axis, -dynamic_objects_count:]
            inferred_object_translation = inferred_object_translation[..., -dynamic_objects_count:]

            # Computes statistics for each object
            for object_idx in range(dynamic_objects_count):
                current_ground_truth_rotations = ground_truth_rotations[..., object_idx]
                current_inferred_rotations = inferred_object_rotation[..., object_idx]

                # Computes the L1 error on rotations
                rotation_error = (current_ground_truth_rotations - current_inferred_rotations).abs().mean()

                current_ground_truth_translations = ground_truth_translations[..., object_idx]
                current_inferred_translations = inferred_object_translation[..., object_idx]

                # Computes the norm of the translation vector difference
                translation_error = (current_ground_truth_translations - current_inferred_translations).pow(2).sum(-1).sqrt().mean()

                loss_info[f"object_{object_idx}_rotation_reconstruction_error_L1"] = rotation_error.item()
                loss_info[f"object_{object_idx}_translation_reconstruction_error_L2_norm"] = translation_error.item()

    def zero_grad_with_none(self, optimizer: Optimizer):
        '''
        Puts the gradient of each optimized parameter to None
        :param optimizer: The optimizer whose parameters must be put to none
        :return:
        '''

        for current_param_group in optimizer.param_groups:
            for current_parameter in current_param_group["params"]:
                current_parameter.grad = None

    def train_epoch(self, model):

        self.logger.print(f'== Train [{self.global_step}] ==')

        # Number of training steps performed in this epoch
        performed_steps = 0

        # Prints tensorboard trace and chrome trace
        def handle_profiler_trace(p):
            print("- Saving tensorboard profiling outoupt")
            torch.profiler.tensorboard_trace_handler('./results/profiler_test_5')(p)

        self.time_meter.start()
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=self.profiler_wait, warmup=1, active=1, repeat=1),
            on_trace_ready=handle_profiler_trace,
            with_stack=True
            ) as prof:
            for step, batch in enumerate(self.dataloader):

                # If the maximum number of training steps per epoch is exceeded, we interrupt the epoch
                if performed_steps > self.config["training"]["max_steps_per_epoch"]:
                    break

                self.global_step += 1
                performed_steps += 1

                # Starts times
                self.torch_time_meter.end("data_loading")

                # Sets the current step of the model
                model.module.set_step(self.global_step)

                with record_function("compute_losses"):
                    # Computes losses
                    loss, loss_info, additional_info, scene_encodings = self.compute_losses(model, batch)

                with record_function("backwards"):
                    self.torch_time_meter.start("backwards")
                    # Sets the gradients to None instead of zeroing them so that Adam would not get incorrect
                    # estimations for the gradient second moments if a tensor is not used in the current forward pass
                    self.zero_grad_with_none(self.optimizer)
                    self.zero_grad_with_none(self.camera_parameters_optimizer)

                    loss.backward()
                    self.torch_time_meter.end("backwards")

                with record_function("consistency_compute_losses"):
                    # Computes pose consistency losses. Accumulates the gradient over the ones for the previous pass
                    if self.use_pose_consistency:
                        pose_consistency_loss, pose_consistency_loss_info, pose_consistency_additional_info = self.compute_pose_consistency_loss(model, batch, scene_encodings)
                        self.torch_time_meter.start("pose_consistency_backwards")
                        pose_consistency_loss.backward()
                        self.torch_time_meter.end("pose_consistency_backwards")

                    # Computes keypoint consistency losses. Accumulates the gradient over the ones for the previous pass
                    if self.use_keypoint_consistency:
                        keypoint_consistency_loss, keypoint_consistency_loss_info, keypoint_consistency_additional_info = self.compute_keypoint_consistency_loss(model, batch, scene_encodings)
                        self.torch_time_meter.start("keypoint_consistency_backwards")
                        keypoint_consistency_loss.backward()
                        self.torch_time_meter.end("keypoint_consistency_backwards")

                with record_function("optimization"):
                    self.torch_time_meter.start("optimizer")
                    self.optimizer.step()
                    self.camera_parameters_optimizer.step()
                    self.torch_time_meter.end("optimizer")

                    # Triggers update of the learning rate
                    if self.global_step % self.lr_decay_iterations == 0:
                        self.lr_scheduler.step()
                        self.logger.print(f"- Learning rate updated to {self.lr_scheduler.get_lr()}")

                # Stops times
                self.time_meter.end()

                # Accumulates loss information
                self.average_meter.add(loss_info)
                # Merges pose consistency and keypoint consistency results if available
                if self.use_pose_consistency:
                    self.average_meter.add(pose_consistency_loss_info)
                    additional_info = {**additional_info, **pose_consistency_additional_info}
                if self.use_keypoint_consistency:
                    self.average_meter.add(keypoint_consistency_loss_info)
                    additional_info = {**additional_info, **keypoint_consistency_additional_info}

                # Logs information at regular intervals
                if (self.global_step - 1) % self.log_interval_steps == 0:

                    self.logger.print(f'step: {self.global_step}/{self.max_steps}', end=" ")

                    average_time = self.time_meter.get_average_time()
                    iterations_per_second = 1 / average_time
                    self.logger.print(f'ips: {iterations_per_second:.3f}', end=" ")

                    for timer_key in sorted(self.torch_time_meter.keys()):
                        current_time = self.torch_time_meter.get_time(timer_key)
                        self.logger.print(f'perf/{timer_key}: {current_time:.4f}', end=" ")

                    average_values = {description: self.average_meter.pop(description) for description in self.average_meter.keys()}
                    for description, value in average_values.items():
                        self.logger.print("{}:{:.3f}".format(description, value), end=" ")

                    current_lr = self._get_current_lr()
                    self.logger.print('lr: %.4f' % (current_lr))

                    # Logs on wandb
                    wandb = self.logger.get_wandb()
                    logged_map = {"train/" + description: item for description, item in average_values.items()}
                    logged_map["ips"] = iterations_per_second
                    logged_map["step"] = self.global_step
                    logged_map["train/lr"] = current_lr
                    wandb.log(logged_map, step=self.global_step)
                    additional_info["step"] = self.global_step
                    wandb.log(additional_info, step=self.global_step)

                self.time_meter.start()
                self.torch_time_meter.start("data_loading")

                prof.step()


def trainer(config, model, dataset, logger):
    return Trainer(config, model, dataset, logger)

