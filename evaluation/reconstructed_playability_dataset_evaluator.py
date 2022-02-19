import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import MulticameraVideoDataset
from evaluation.metrics.action_linear_classification import ActionClassificationScore
from evaluation.metrics.action_variance import ActionVariance
from evaluation.metrics.fid import FID
from evaluation.metrics.fvd import IncrementalFVD
from evaluation.metrics.lpips import LPIPS
from evaluation.metrics.metrics_accumulator import MetricsAccumulator
from evaluation.metrics.motion_masked_mse import MotionMaskedMSE
from evaluation.metrics.mse import MSE
from evaluation.metrics.psnr import PSNR
from evaluation.metrics.ssim import SSIM
from evaluation.metrics.vgg_cosine_similarity import VGGCosineSimilarity
from evaluation.plotting.density_plot import DensityPlotter
from evaluation.plotting.density_plot_2d import DensityPlotter2D
from evaluation.plotting.density_plot_2d_merged import DensityPlotter2DMerged
from evaluation.plotting.mean_vector_plot_2d import MeanVectorPlotter2D
from model.classic_object_parameters_encoder import ClassicObjectParametersEncoder
from model.object_parameters_encoder_v4 import ObjectParametersEncoderV4
from model.utils.object_ids_helper import ObjectIDsHelper
from utils.lib_3d.pose_parameters import PoseParameters
from utils.lib_3d.transformations_3d import Transformations3D
from utils.logger import Logger
from utils.tensor_folder import TensorFolder


class ReconstructedPlayabilityDatasetEvaluator:
    '''
    Generation results evaluator class
    '''

    def __init__(self, config, logger: Logger, reference_dataset: MulticameraVideoDataset, generated_dataset: MulticameraVideoDataset):
        '''
        Creates an evaluator

        :param config: The configuration file
        :param reference_dataset: the dataset to use as ground truth
        :param generated_dataset: the generated dataset to compare to ground truth
        '''

        self.config = config
        self.logger = logger
        self.reference_dataset = reference_dataset
        self.generated_dataset = generated_dataset
        self.reference_dataloader = DataLoader(self.reference_dataset,
                                               batch_size=self.config["evaluation"]["reconstructed_dataset_evaluation_batching"]["batch_size"],
                                               shuffle=False, collate_fn=single_batch_elements_collate_fn,
                                               num_workers=self.config["evaluation"]["reconstructed_dataset_evaluation_batching"]["num_workers"],
                                               pin_memory=True)
        self.generated_dataloader = DataLoader(self.generated_dataset,
                                               batch_size=self.config["evaluation"]["reconstructed_dataset_evaluation_batching"]["batch_size"],
                                               shuffle=False, collate_fn=single_batch_elements_collate_fn,
                                               num_workers=self.config["evaluation"]["reconstructed_dataset_evaluation_batching"]["num_workers"],
                                               pin_memory=True)

        if len(self.reference_dataloader) != len(self.generated_dataloader):
            raise Exception(f"Reference and generated datasets should have the same sequences, but their length differs:"
                            f"Reference ({len(self.reference_dataloader)}), Generated({len(self.generated_dataloader)})")

        self.actions_count = self.config["data"]["actions_count"]
        self.plots_output_directory = self.config["logging"]["output_directory"]

        self.normalization_transform = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.player_detector = self.get_player_detector()

        # Correction factor for the focal lengths
        self.focal_length_multiplier = config["data"]["focal_length_multiplier"]

        self.action_variance = ActionVariance()
        self.action_accuracy = ActionClassificationScore()
        # This will be a classic object parameters encoder, so it is fine to use the same for all the objects. Take the one defined for the last object
        model_config = self.config["model"]["object_parameters_encoder"][-1]
        if model_config["architecture"] == 'model.classic_object_parameters_encoder':
            self.object_parameters_encoder = ClassicObjectParametersEncoder(self.config, model_config).cuda()  # Moves the model on GPU
            self.zero_axis = self.object_parameters_encoder.zero_axis
        elif model_config["architecture"] == 'model.object_parameters_encoder_v4':
            self.object_parameters_encoder = ObjectParametersEncoderV4(self.config, model_config).cuda()  # Moves the model on GPU
            self.zero_axis = 1  # ObjectParametersEncoderV4 assumes xz ground plane
        else:
            raise Exception(f"Evaluator expected a 'model.classic_object_parameters_encoder' or a 'model.object_parameters_encoder_v4', but a '{model_config['architecture']}' was detected")

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        self.mse = MSE()
        self.motion_masked_mse = MotionMaskedMSE()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.vgg_sim = VGGCosineSimilarity()
        self.fid = FID()
        #self.inception_score = InceptionScore()
        self.fvd = IncrementalFVD()

    def get_player_detector(self):
        '''
        Returns a detector for the players
        :return:
        '''

        raise NotImplementedError("Need to implement the player detector")

    def check_range(self, images):
        '''
        Checks that the images have the expected range
        :param images: (...) tensor with images
        :return:
        '''

        max = torch.max(images).item()
        min = torch.min(images).item()
        if max > 1.0 or min < 0.0:
            raise Exception(f"Input tensor outside allowed range [0.0, 1.0]: [{min}, {max}]")

    def compute_positional_statistics(self, values: np.ndarray, prefix:str) -> Dict:
        '''
        Computes statistics per each position in the sequence plus the statistic averaged over all positions

        :param values: (bs, sequence_length) tensor with input values
        :param prefix: String to use as prefix for each returned key
        :return: Dictionary with keys in the form (prefix/i: value) where i is the positional index or avg for the average
        '''

        results = {}

        # Computes the statistics per postion
        positional_values = values.mean(axis=0)
        positional_variances = values.var(axis=0).tolist()
        global_variance = float(positional_values.var())
        positional_values = positional_values.tolist()
        global_value = float(sum(positional_values) / len(positional_values))

        results[f"{prefix}/avg"] = global_value
        results[f"{prefix}/var"] = global_variance
        for idx, current_value in enumerate(positional_values):
            results[f"{prefix}/{idx}"] = current_value
        for idx, current_variance in enumerate(positional_variances):
            results[f"{prefix}/{idx}/var"] = current_variance

        return results

    def compute_movements_and_actions(self, bounding_boxes: torch.Tensor,  bounding_boxes_validity: torch.Tensor, image_size: Tuple[int, int], metadata: Dict):
        '''

        :param bounding_boxes: (bs, observations_count, cameras_count, 4, dynamic_objects_count) tensor with (left, top, right, bottom) bounding boxes normalized in [0, 1]
        :param image_size: (width, height) size of the image
        :param metadata: metadata (batch_size, observations_count, cameras_count) list with metadata
        :return: Lists indexed by dynamic object idx with respectively (detected_movements, 2), (detected_movements) arrays with detected movements,
                 actions inferred by the model corresponding to the detected movements for each object
        '''

        # Denormalizes the bounding boxes
        bounding_boxes = bounding_boxes.clone()
        image_size = torch.as_tensor([image_size[0], image_size[1], image_size[0], image_size[1]], device=bounding_boxes.device).unsqueeze(-1)
        bounding_boxes = bounding_boxes * image_size
        bounding_box_centers = (bounding_boxes[..., 2:, :] + bounding_boxes[..., :2, :]) / 2

        # Movements and inferred actions by dynamic object idx
        all_movements = []
        all_inferred_actions = []

        batch_size = bounding_boxes.shape[0]
        observations_count = bounding_boxes.shape[1]
        cameras_count = bounding_boxes.shape[2]
        dynamic_objects_count = bounding_boxes.shape[-1]

        for dynamic_object_idx in range(dynamic_objects_count):
            movements = []
            inferred_actions = []

            for sequence_idx in range(batch_size):
                for observation_idx in range(observations_count - 1):
                    for camera_idx in range(cameras_count):
                        # If there was a successful detection for both the current and the successive frame
                        if bounding_boxes_validity[sequence_idx, observation_idx, camera_idx, dynamic_object_idx] and \
                           bounding_boxes_validity[sequence_idx, observation_idx + 1, camera_idx, dynamic_object_idx]:

                            # Extract movement and action
                            current_movement = (bounding_box_centers[sequence_idx, observation_idx + 1, camera_idx, :, dynamic_object_idx] - bounding_box_centers[sequence_idx, observation_idx, camera_idx, :, dynamic_object_idx]).cpu().numpy()
                            current_action = metadata[sequence_idx][observation_idx][camera_idx]["inferred_action"][dynamic_object_idx]

                            movements.append(current_movement)
                            inferred_actions.append(current_action)

            if len(movements) > 0:
                all_movements.append(np.stack(movements, axis=0))
                all_inferred_actions.append(np.stack(inferred_actions, axis=0))
            else: # Some sequences do not have any valid bounding box, add empty tensors
                all_movements.append(np.zeros((0, 2), dtype=np.float32))
                all_inferred_actions.append(np.zeros((0,), dtype=np.int64))

        return all_movements, all_inferred_actions

    def get_rotation_matrix(self, rotations: torch.Tensor, rotation_axis: int) -> torch.Tensor:
        '''

        :param rotations: (..., 3) tensor with rotations
        :return: (..., 2, 2) tensor of rotation matrices around the rotation_axis
        '''

        if rotation_axis == 0:
            rotation_function = Transformations3D.rotation_matrix_x
        elif rotation_axis == 1:
            rotation_function = Transformations3D.rotation_matrix_y
        elif rotation_axis == 2:
            rotation_function = Transformations3D.rotation_matrix_z
        else:
            raise Exception(f"Invalid rotation axis {rotation_axis}")

        preserved_axes = list(sorted(set(range(3)) - set([rotation_axis])))

        # Transforms in a 2D matrix the rotation around the rotation axis. Needs two steps to perform indexing
        rotation_matrices = rotation_function(rotations[..., rotation_axis])
        rotation_matrices = rotation_matrices[..., preserved_axes, :]
        rotation_matrices = rotation_matrices[..., preserved_axes]
        return rotation_matrices

    def get_camera_relative_movements(self, movements_by_dynamic_object_idx: List[np.ndarray], camera_rotations_by_dynamic_object_idx: List[torch.Tensor], rotation_axis: int) -> List[np.ndarray]:
        '''

        :param movements_by_dynamic_object_idx: list indexed by dynamic object idx of (..., 3) tensor with object movements expressed in world coordinates
        :param camera_rotations_by_dynamic_object_idx: list indexed by dynamic object idx of (..., 3) tensor with camera rotations. Only a single camera must be present
        :param rotation_axis: Rotation axis to align to the camera. Should be the axis normal to the ground
        :return: list indexed by dynamic object idx of (..., 3) array with object movements expressed relative to the camera. Only a rotation along rotation_axis is applied
        '''

        camera_relative_movements_by_dynamic_object_idx = []

        dynamic_objects_count = len(movements_by_dynamic_object_idx)
        for dynamic_object_idx in range(dynamic_objects_count):
            camera_rotations = camera_rotations_by_dynamic_object_idx[dynamic_object_idx]
            # Gets movements and converts them to tensors
            movements = movements_by_dynamic_object_idx[dynamic_object_idx]
            movements = torch.from_numpy(movements).to(camera_rotations.device)

            movements, initial_dimensions = TensorFolder.flatten(movements, -1)
            camera_rotations, _ = TensorFolder.flatten(camera_rotations, -1)

            elements_count = movements.size(0)
            camera_elements_count = camera_rotations.size(0)

            if elements_count != camera_elements_count:
                raise Exception(f"The number of movements does not match the number of cameras, got respectively {elements_count} and {camera_elements_count}.")

            # (elements_count, 2, 2)
            rotation_matrices = self.get_rotation_matrix(-camera_rotations, rotation_axis)  # Rotations are negated to get teh transformations from world to camera

            # (elements_count, 2, 1) Translations expressed in the object coordinate system
            movements = movements.unsqueeze(-1)
            # (elements_count, 2). Translations expresesd in the world coordinate system
            camera_relative_movements = torch.matmul(rotation_matrices, movements).squeeze(-1)

            # Folds the results
            camera_relative_movements = TensorFolder.fold(camera_relative_movements, initial_dimensions)
            camera_relative_movements = camera_relative_movements.cpu().numpy()  # Results are returned in numpy format
            camera_relative_movements_by_dynamic_object_idx.append(camera_relative_movements)

        return camera_relative_movements_by_dynamic_object_idx

    def get_selected_cameras(self, tensor_to_index, indexes):
        '''

        :param tensor_to_index: (batch_size, observations_count, cameras_count, ...)
        :param indexes: list indexed by dynamic object idx of (batch_idx, observation_idx, camera_idx) indexes of values to select
        :return: list indexed by dynamic object idx of (elements_count, ...) tensor with the selected tensor values
        '''

        all_indexed_tensors = []

        dynamic_objects_count = len(indexes)
        # Indexes each object
        for dynamic_object_idx in range(dynamic_objects_count):
            current_indexes = indexes[dynamic_object_idx]

            indexed_tensors = []
            # Gets all values at the specified indexes for the current object
            for current_index_tuple in current_indexes:
                current_indexed_tensor = tensor_to_index[current_index_tuple]
                indexed_tensors.append(current_indexed_tensor)

            if len(indexed_tensors) > 0:
                indexed_tensors = torch.stack(indexed_tensors, dim=0)
            # Handles the case where there is nothing to select
            else:
                indexed_tensor_dimensions = [0] + list(tensor_to_index.size())[3:]
                indexed_tensors = torch.zeros(indexed_tensor_dimensions, dtype=tensor_to_index.dtype, device=tensor_to_index.device)
            all_indexed_tensors.append(indexed_tensors)

        return all_indexed_tensors

    def compute_world_movements(self, observations: torch.Tensor, transformation_matrix_w2c: torch.Tensor, camera_rotations: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor,  bounding_boxes_validity: torch.Tensor, metadata: Dict):
        '''

        :param observations: (bs, observations_count, cameras_count, 3, height, width) tensor with observations
        :param transformation_matrix_w2c: (bs, observations_count, cameras_count, 4, 4) tensor with transformation
                                          from world to camera coordinates
        :param camera_rotations: (bs, observations_count, cameras_count, 3) tensor with camera rotations
        :param focals: (bs, observations_count, cameras_count) tensor with camera focals
        :param bounding_boxes: (bs, observations_count, cameras_count, 4, dynamic_objects_count) tensor with (left, top, right, bottom) bounding boxes normalized in [0, 1]
        :param image_size: (width, height) size of the image
        :param metadata: metadata (batch_size, observations_count, cameras_count) list with metadata
        :return: Lists indexed by dynamic object idx with (detected_movements, 2) arrays with detected movements on the 2 axes of the ground plane
                 List indexed by dynamic object idx with (sequence_idx, observation_idx, camera_idx) indexes corresponding for the starting frame of each selected movement
                 List indexed by dynamic object idx with actions inferred by the model corresponding to the detected movements
        '''


        bounding_boxes = bounding_boxes.clone()

        # Movements and inferred actions by dynamic object idx
        all_movements = []
        all_inferred_actions = []
        # list of (sequence_idx, observation_idx, camera_idx) indexes corresponding for the starting frame of each selected movement
        all_selected_indexes = []

        batch_size = bounding_boxes.shape[0]
        observations_count = bounding_boxes.shape[1]
        cameras_count = bounding_boxes.shape[2]
        dynamic_objects_count = bounding_boxes.shape[-1]

        parameter_encoder_results = self.object_parameters_encoder(observations, transformation_matrix_w2c, camera_rotations, focals, bounding_boxes, bounding_boxes_validity, apply_ranges=False)

        object_translations = parameter_encoder_results[1]

        # Computes the axes that represent the ground plane
        zero_axis = self.zero_axis
        axes_to_select = list(sorted(set(range(3)) - set([zero_axis])))
        object_translations = object_translations[..., axes_to_select, :]

        for dynamic_object_idx in range(dynamic_objects_count):
            movements = []
            selected_indexes = []
            inferred_actions = []

            for sequence_idx in range(batch_size):
                for observation_idx in range(observations_count - 1):
                    for camera_idx in range(cameras_count):
                        # If there was a successful detection for both the current and the successive frame
                        if bounding_boxes_validity[sequence_idx, observation_idx, camera_idx, dynamic_object_idx] and \
                           bounding_boxes_validity[sequence_idx, observation_idx + 1, camera_idx, dynamic_object_idx]:

                            successor_translation = object_translations[sequence_idx, observation_idx + 1, :, dynamic_object_idx]
                            predecessor_translation = object_translations[sequence_idx, observation_idx, :, dynamic_object_idx]

                            current_movement = (successor_translation - predecessor_translation)
                            # Filters out translations known to be not valid to reduce noise (eg. bad detections)
                            if self.is_movement_valid(current_movement):
                                # Extract movement

                                current_movement = current_movement.cpu().numpy()
                                current_action = metadata[sequence_idx][observation_idx][camera_idx]["inferred_action"][dynamic_object_idx]

                                selected_indexes.append((sequence_idx, observation_idx, camera_idx))
                                movements.append(current_movement)  # Transforms movements to numpy
                                inferred_actions.append(current_action)

            if len(movements) > 0:
                all_movements.append(np.stack(movements, axis=0))
                all_inferred_actions.append(np.stack(inferred_actions, axis=0))
            else:  # Some sequences do not have any valid bounding box, add empty tensors
                all_movements.append(np.zeros((0, 2), dtype=np.float32))
                all_inferred_actions.append(np.zeros((0,), dtype=np.int64))

            all_selected_indexes.append(selected_indexes)

        return all_movements, all_selected_indexes, all_inferred_actions

    def is_movement_valid(self, movement: torch.Tensor) -> bool:
        '''

        :param movement: (3) tensor with movement
        :return: True if the current movement is valid for the current dataset
        '''

        return True

    def compensate_bounding_boxes(self, bounding_boxes: torch.Tensor):
        '''
        Applies compensation to ground truth bounding boxes if needed

        :param bounding_boxes: (..., 4, dynamic_objects_count) tensor with boudning boxes
        :return:
        '''

        # Apply no compensation
        return bounding_boxes

    def match_generated_detections_to_reference(self, reference_detections: List[List[torch.Tensor]], reference_detections_validities: List[List[bool]], generated_detections: List[List[torch.Tensor]], detected_bounding_boxes: List[List[torch.Tensor]], threshold=0.1):
        '''
        Matches generated detections to reference detections

        :param reference_detections: (observations_count, dynamic_objects_count) list of (rows, cols) tensors of detected centers in the reference images
        :param reference_detections_validities: (observations_count, dynamic_objects_count) list of booleans with True if the corresponding detection is valid
        :param generated_detections: (observations_count, detections_count) list of (rows, cols) tensors of detected centers in the generated images
        :param detected_bounding_boxes: (observations_count, detections_count) list of (left, top, right, bottom) tensors of detected bopunding boxes in the generated images normalized in [0, 1]
        :return: (observations_count, dynamic_objects_count) list of (rows, cols) tensors of detected centers in the generated detections that match the reference detections
                 (observations_count, dynamic_objects_count) list of (left, top, right, bottom) tensors of detected bounding boxes normalized in [0, 1] in the detected bounding boxes that match the reference detections
                 (observations_count, dynamic_objects_count) list of booleans with True if the corresponding detection is valid
        '''

        device = reference_detections[0][0].device

        all_matched_generated_detections = []
        all_matched_detected_boxes = []
        all_matched_generated_detections_validities = []

        # Iterates over each observation
        for observation_idx, (current_reference_detections, current_reference_detections_validities, current_generated_detections, current_detected_bounding_boxes) in enumerate(zip(reference_detections, reference_detections_validities, generated_detections, detected_bounding_boxes)):
            current_matched_generated_detections = []
            current_matched_detected_boxes = []
            current_matched_generated_detections_validities = []

            # Copies the list so that it can be modified subsequently
            current_generated_detections = list(current_generated_detections)
            current_detected_bounding_boxes = list(current_detected_bounding_boxes)

            # Iterates over all matches for the current observation (Should be 2 for tennis in the reference)
            for candidate_reference_detection, candidate_reference_detection_validity in zip(current_reference_detections, current_reference_detections_validities):
                # If the current reference detection is not valid, match nothing to it
                if not candidate_reference_detection_validity:
                    # Add placeholder
                    current_matched_generated_detections.append(torch.as_tensor((0.5, 0.5), device=device))
                    current_matched_detected_boxes.append(torch.as_tensor((0.25, 0.25, 0.5, 0.5), device=device))
                    current_matched_generated_detections_validities.append(False)
                else:
                    current_min_distance = 1e6
                    current_min_idx = -1
                    # Finds the id of the generated detection closest to the reference detection
                    for candidate_idx, candidate_generated_detection in enumerate(current_generated_detections):
                        distance = (candidate_generated_detection - candidate_reference_detection).pow(2).sum().item()
                        if distance < threshold and distance < current_min_distance:
                            current_min_distance = distance
                            current_min_idx = candidate_idx

                    # A match was found
                    if current_min_idx != -1:
                        best_match = current_generated_detections[current_min_idx]
                        best_box_match = current_detected_bounding_boxes[current_min_idx]
                        del current_generated_detections[current_min_idx]  # Removes the element so that it can no longer be matched
                        del current_detected_bounding_boxes[current_min_idx]  # Removes the element so that it can no longer be matched
                        current_matched_generated_detections.append(best_match)
                        current_matched_detected_boxes.append(best_box_match)
                        current_matched_generated_detections_validities.append(True)

                    # No match was found
                    else:
                        # Add placeholder
                        current_matched_generated_detections.append(torch.as_tensor((0.5, 0.5), device=device))
                        current_matched_detected_boxes.append(torch.as_tensor((0.25, 0.25, 0.5, 0.5), device=device))
                        current_matched_generated_detections_validities.append(False)

            all_matched_generated_detections.append(current_matched_generated_detections)
            all_matched_detected_boxes.append(current_matched_detected_boxes)
            all_matched_generated_detections_validities.append(current_matched_generated_detections_validities)

        return all_matched_generated_detections, all_matched_detected_boxes, all_matched_generated_detections_validities

    def make_movement_plots(self, actions: np.ndarray, movements: np.ndarray, prefix: str):
        '''
        Produces plots showing interactions between actions and movements
        :param actions: (...) array with actions
        :param movements: (..., 2) array with movements
        :param prefix: prefix to assign to the produced plots
        :return:
        '''
        DensityPlotter.plot_density(actions, movements, self.actions_count, self.plots_output_directory, prefix=prefix)
        try:
            DensityPlotter2D.plot_density(actions, movements, self.actions_count, self.plots_output_directory, prefix=prefix)
        # Sometimes risen by seaborn, continue with the evaluation anyway
        except ValueError:
            print(traceback.format_exc())

        DensityPlotter2DMerged.plot_density(actions, movements, self.actions_count, self.plots_output_directory, xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), prefix=prefix)
        MeanVectorPlotter2D.plot(actions, movements, self.actions_count, self.plots_output_directory, xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), prefix=prefix)

    def make_all_movement_plots(self, movement_values: List, world_movement_values: List, camera_relative_world_movement_values: List, inferred_action_values: List, inferred_world_action_values: List, prefix: str):
        '''
        Makes all movement plots

        :param movement_values: list with an entry for each dynamic object count. Each element must be in the format required by make_movement_plots
        :param world_movement_values: list with an entry for each dynamic object count. Each element must be in the format required by make_movement_plots
        :param camera_relative_world_movement_values: list with an entry for each dynamic object count. Each element must be in the format required by make_movement_plots
        :param inferred_action_values: list with an entry for each dynamic object count. Each element must be in the format required by make_movement_plots
        :param inferred_world_action_values: list with an entry for each dynamic object count. Each element must be in the format required by make_movement_plots
        :param prefix: Prefix to use for all saved plots
        :return:
        '''

        for dynamic_object_idx in range(len(movement_values)):
            plot_prefix = f"{prefix}object_{dynamic_object_idx}_"
            self.make_movement_plots(inferred_action_values[dynamic_object_idx], movement_values[dynamic_object_idx], plot_prefix)
            plot_prefix = f"{prefix}world_object_{dynamic_object_idx}_"
            self.make_movement_plots(inferred_world_action_values[dynamic_object_idx], world_movement_values[dynamic_object_idx], plot_prefix)
            plot_prefix = f"{prefix}camera_relative_world_object_{dynamic_object_idx}_"
            self.make_movement_plots(inferred_world_action_values[dynamic_object_idx], camera_relative_world_movement_values[dynamic_object_idx], plot_prefix)

    def pop_movements_and_actions(self, accumulator, prefix: str, dynamic_objects_count: int):
        '''
        Pops movement and action information from the accumulator

        :param accumulator: accumulator from which to pop the results
        :param prefix: prefix to use on the keys to retrieve
        :param dynamic_objects_count: Number of dynamic objects
        :return:
        '''

        movement_values = []
        world_movement_values = []
        camera_relative_world_movement_values = []
        inferred_action_values = []
        inferred_world_action_values = []

        for dynamic_object_idx in range(dynamic_objects_count):
            movement_values.append(accumulator.pop(f"{prefix}movements_object_{dynamic_object_idx}"))
            world_movement_values.append(accumulator.pop(f"{prefix}world_movements_object_{dynamic_object_idx}"))
            camera_relative_world_movement_values.append(accumulator.pop(f"{prefix}camera_relative_world_movements_object_{dynamic_object_idx}"))
            inferred_action_values.append(accumulator.pop(f"{prefix}inferred_actions_object_{dynamic_object_idx}"))
            inferred_world_action_values.append(accumulator.pop(f"{prefix}inferred_world_actions_object_{dynamic_object_idx}"))

        return movement_values, world_movement_values, camera_relative_world_movement_values, inferred_action_values, inferred_world_action_values

    def compute_action_accuracy_and_variation(self, movement_values: List, world_movement_values: List, camera_relative_world_movement_values: List, inferred_action_values: List, inferred_world_action_values: List, results: Dict, prefix: str):
        '''
        Computes action accuracy and variation results

        :param movement_values: list with an entry for each dynamic object count. Each element must be in the format required by ActionVariance
        :param world_movement_values: list with an entry for each dynamic object count. Each element must be in the format required by ActionVariance
        :param camera_relative_world_movement_values: list with an entry for each dynamic object count. Each element must be in the format required by ActionVariance
        :param inferred_action_values: list with an entry for each dynamic object count. Each element must be in the format required by ActionVariance
        :param inferred_world_action_values: list with an entry for each dynamic object count. Each element must be in the format required by ActionVariance
        :param results: dictionary where to append results
        :param prefix: prefix to use for results keys
        :return:
        '''

        for dynamic_object_idx in range(len(movement_values)):
            animation_model_idx = self.object_id_helper.animation_model_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_actions_count = self.config["playable_model"]["object_animation_models"][animation_model_idx]["actions_count"]

            # Action results for image plane movements
            action_variance_results = self.action_variance(inferred_action_values[dynamic_object_idx], movement_values[dynamic_object_idx], current_actions_count, dynamic_object_idx)
            action_accuracy_results = self.action_accuracy(inferred_action_values[dynamic_object_idx], movement_values[dynamic_object_idx], current_actions_count, dynamic_object_idx)
            results = dict(results, **{f"{prefix}" + k: v for k, v in action_variance_results.items()})
            results = dict(results, **{f"{prefix}" + k: v for k, v in action_accuracy_results.items()})

            # Action results for world movements
            world_action_variance_results = self.action_variance(inferred_world_action_values[dynamic_object_idx], world_movement_values[dynamic_object_idx], current_actions_count, dynamic_object_idx)
            world_action_accuracy_results = self.action_accuracy(inferred_world_action_values[dynamic_object_idx], world_movement_values[dynamic_object_idx], current_actions_count, dynamic_object_idx)
            results = dict(results, **{f"{prefix}world_" + k: v for k, v in world_action_variance_results.items()})
            results = dict(results, **{f"{prefix}world_" + k: v for k, v in world_action_accuracy_results.items()})

            # Action results for camera relative world movements
            camera_relative_world_action_variance_results = self.action_variance(inferred_world_action_values[dynamic_object_idx], camera_relative_world_movement_values[dynamic_object_idx], current_actions_count, dynamic_object_idx)
            camera_relative_world_action_accuracy_results = self.action_accuracy(inferred_world_action_values[dynamic_object_idx], camera_relative_world_movement_values[dynamic_object_idx], current_actions_count, dynamic_object_idx)
            results = dict(results, **{f"{prefix}camera_relative_world_" + k: v for k, v in camera_relative_world_action_variance_results.items()})
            results = dict(results, **{f"{prefix}camera_relative_world_" + k: v for k, v in camera_relative_world_action_accuracy_results.items()})

        return results

    def compute_metrics(self) -> Dict:
        '''
        Computes evaluation metrics on the given datasets

        :return: Dictionary with evaluation results
        '''

        accumulator = MetricsAccumulator()

        #self.logger.print("- Computing IS")
        #is_results = self.inception_score(self.generated_dataloader)

        total_valid_detections = {}  # Number of valid reference detections by object_idx
        total_matched_valid_detections = {}  # Number of valid reference detections to which a valid generated detection is matched by object_idx
        total_matched_distance = {}  # Distance between matched generated and reference detections when a valid match is present by object_idx

        batches = len(self.reference_dataloader)
        with torch.no_grad():
            for idx, (reference_batch, generated_batch) in enumerate(zip(self.reference_dataloader, self.generated_dataloader)):

                # Logs the current batch
                if idx % 1 == 0:
                    self.logger.print(f"- Computing metrics for batch [{idx}/{batches}]")

                # Extracts reference data
                reference_batch_tuple = reference_batch.to_tuple()
                reference_observations, reference_actions, reference_rewards, reference_dones, reference_camera_rotations,\
                reference_camera_translations, reference_focals, reference_bounding_boxes, reference_bounding_boxes_validity,\
                reference_global_frame_indexes, reference_video_frame_indexes, reference_video_indexes = reference_batch_tuple

                image_size_tensor = torch.tensor((reference_observations.size(-2), reference_observations.size(-1)), dtype=torch.float, device=reference_observations.device)
                # Computes the transformation matrices for each image
                reference_transformation_c2w = PoseParameters(reference_camera_rotations, reference_camera_translations)
                # Gets the transformation matrix
                reference_transformation_matrix_w2c = reference_transformation_c2w.get_inverse_homogeneous_matrix()

                # Extracts generated data
                generated_batch_tuple = generated_batch.to_tuple()
                generated_observations, generated_actions, generated_rewards, generated_dones, generated_camera_rotations,\
                generated_camera_translations, generated_focals, generated_bounding_boxes, generated_bounding_boxes_validity,\
                generated_global_frame_indexes, generated_video_frame_indexes, generated_video_indexes = generated_batch_tuple

                # Applies compensation to the bounding boxes if needed
                reference_bounding_boxes = self.compensate_bounding_boxes(reference_bounding_boxes)
                generated_bounding_boxes = self.compensate_bounding_boxes(generated_bounding_boxes)

                # Corrects focal lengths since they are read directly from the dataset and thus are still to be corrected
                reference_focals = reference_focals * self.focal_length_multiplier
                generated_focals = generated_focals * self.focal_length_multiplier

                observations_count = reference_observations.size(1)
                objects_count = reference_bounding_boxes.size(-1)
                if reference_observations.size(0) != 1:
                    raise Exception("Expected evaluation batches to have size 1")

                # Initializes counters if they are not yet initialized
                if len(total_valid_detections) == 0:
                    for object_idx in range(objects_count):
                        total_valid_detections[object_idx] = 0
                        total_matched_valid_detections[object_idx] = 0
                        total_matched_distance[object_idx] = 0.0

                # Gets a list of detected human centers from the dataset bounding boxes
                reference_detections = []
                for observation_idx, current_observation_boxes in enumerate(reference_bounding_boxes[0]):
                    current_observation_reference_detections = []
                    current_observation_boxes = current_observation_boxes[0]  # Gets results for the first camers
                    objects_count = current_observation_boxes.size(-1)
                    for object_idx in range(objects_count):
                        current_object_box = current_observation_boxes[:, object_idx]
                        current_center = torch.stack([(current_object_box[3] + current_object_box[1]) / 2, (current_object_box[2] + current_object_box[0]) / 2])
                        current_observation_reference_detections.append(current_center)
                    reference_detections.append(current_observation_reference_detections)

                reference_detections_validities = []
                for observation_idx, current_observation_boxes in enumerate(reference_bounding_boxes_validity[0]):
                    current_observation_reference_detections = []
                    current_observation_boxes = current_observation_boxes[0]  # Gets results for the first camers
                    objects_count = current_observation_boxes.size(-1)
                    for object_idx in range(objects_count):
                        current_validity = current_observation_boxes[object_idx].item()
                        current_observation_reference_detections.append(current_validity)
                    reference_detections_validities.append(current_observation_reference_detections)

                #reference_detections = [[torch.as_tensor(((current_box[3] - current_box[1]) / 2, (current_box[2] - current_box[0]) / 2)) for current_box in current_observation_boxes[0]] for current_observation_boxes in reference_bounding_boxes[0]]
                #reference_detections_validities = [[current_box_validity.item() for current_box_validity in current_observation_boxes[0]] for current_observation_boxes in reference_bounding_boxes_validity[0]]
                # Gets a list of detected human centers for each of the observations_count in the only batch element
                normalized_generated_observations = self.normalization_transform(generated_observations)
                generated_detections, detected_bounding_boxes = self.player_detector(normalized_generated_observations[0, :, 0])  # Removes batch size and camera size

                # Performs matching between generated and reference detections
                generated_detections, detected_bounding_boxes, generated_detections_validities = self.match_generated_detections_to_reference(reference_detections, reference_detections_validities, generated_detections, detected_bounding_boxes)

                # Converts the detected bounding boxes to a tensor
                detected_bounding_boxes = [torch.stack(current_element, dim=1) for current_element in detected_bounding_boxes]  # Creates the dynamic_objects_count dimension
                detected_bounding_boxes = torch.stack(detected_bounding_boxes, dim=0)  # Creates the observations_count dimension
                # (1, observations_count, 1, 4, dynamic_objects_count)
                detected_bounding_boxes = detected_bounding_boxes.unsqueeze(0).unsqueeze(2)  # Creates batch size and camera dimensions

                # converts the detected boxes validities to a tensor
                detected_bounding_boxes_validities = torch.as_tensor(generated_detections_validities, device=detected_bounding_boxes.device)
                detected_bounding_boxes_validities = detected_bounding_boxes_validities.unsqueeze(0).unsqueeze(2)

                # Computes statistics on the matches
                for observation_idx in range(observations_count):
                    for object_idx in range(objects_count):
                        current_reference_center = reference_detections[observation_idx][object_idx]
                        current_reference_validity = reference_detections_validities[observation_idx][object_idx]
                        current_generated_center = generated_detections[observation_idx][object_idx]
                        current_generated_validity = generated_detections_validities[observation_idx][object_idx]

                        if current_reference_validity:
                            total_valid_detections[object_idx] += 1
                            # Both have been detected correctly, so compute distance
                            if current_generated_validity:
                                total_matched_valid_detections[object_idx] += 1
                                # Transforms the distance in pixels
                                current_reference_center = current_reference_center * image_size_tensor
                                current_generated_center = current_generated_center * image_size_tensor
                                total_matched_distance[object_idx] += (current_reference_center - current_generated_center).pow(2).sum().sqrt().item()

                image_size = (reference_observations.size(-1), reference_observations.size(-2))  # (width, height)
                # Evaluates how the action is computed based on ground truth movement
                movements, inferred_actions = self.compute_movements_and_actions(reference_bounding_boxes, reference_bounding_boxes_validity, image_size, generated_batch.metadata)
                world_movements, selected_movement_indexes, inferred_world_actions = self.compute_world_movements(reference_observations, reference_transformation_matrix_w2c, reference_camera_rotations, reference_focals, reference_bounding_boxes, reference_bounding_boxes_validity, generated_batch.metadata)
                selected_reference_camera_rotations = self.get_selected_cameras(reference_camera_rotations, selected_movement_indexes)
                camera_relative_world_movements = self.get_camera_relative_movements(world_movements, selected_reference_camera_rotations, self.zero_axis)

                # Evaluates how the object was moved in response to an action
                generated_movements, generated_inferred_actions = self.compute_movements_and_actions(detected_bounding_boxes, detected_bounding_boxes_validities, image_size, generated_batch.metadata)
                generated_world_movements, generated_selected_movement_indexes, generated_inferred_world_actions = self.compute_world_movements(reference_observations, reference_transformation_matrix_w2c, reference_camera_rotations, reference_focals, detected_bounding_boxes, detected_bounding_boxes_validities, generated_batch.metadata)
                generated_selected_reference_camera_rotations = self.get_selected_cameras(reference_camera_rotations, generated_selected_movement_indexes)
                generated_camera_relative_world_movements = self.get_camera_relative_movements(generated_world_movements, generated_selected_reference_camera_rotations, self.zero_axis)

                # TODO ensure here are not normalized
                # Checks the range of the input tensors
                self.check_range(reference_observations)
                self.check_range(generated_observations)

                # Computes metrics
                mse = self.mse(reference_observations, generated_observations)
                motion_masked_mse = self.motion_masked_mse(reference_observations, generated_observations)
                psnr = self.psnr(reference_observations, generated_observations)
                ssim = self.ssim(reference_observations, generated_observations)
                lpips = self.lpips(reference_observations, generated_observations)
                vgg_sim = self.vgg_sim(reference_observations, generated_observations)

                accumulator.add("reference_detections", reference_detections)
                accumulator.add("generated_detections", generated_detections)
                accumulator.add("mse", mse.cpu().numpy())
                accumulator.add("motion_masked_mse", motion_masked_mse.cpu().numpy())
                accumulator.add("psnr", psnr.cpu().numpy())
                accumulator.add("ssim", ssim.cpu().numpy())
                accumulator.add("lpips", lpips.cpu().numpy())
                accumulator.add("vgg_sim", vgg_sim.cpu().numpy())

                dynamic_objects_count = reference_bounding_boxes.size(-1)
                for dynamic_object_idx in range(dynamic_objects_count):
                    accumulator.add(f"movements_object_{dynamic_object_idx}", movements[dynamic_object_idx])
                    accumulator.add(f"world_movements_object_{dynamic_object_idx}", world_movements[dynamic_object_idx])
                    accumulator.add(f"camera_relative_world_movements_object_{dynamic_object_idx}", camera_relative_world_movements[dynamic_object_idx])
                    accumulator.add(f"inferred_actions_object_{dynamic_object_idx}", inferred_actions[dynamic_object_idx])
                    accumulator.add(f"inferred_world_actions_object_{dynamic_object_idx}", inferred_world_actions[dynamic_object_idx])

                    accumulator.add(f"generated_movements_object_{dynamic_object_idx}", generated_movements[dynamic_object_idx])
                    accumulator.add(f"generated_world_movements_object_{dynamic_object_idx}", generated_world_movements[dynamic_object_idx])
                    accumulator.add(f"generated_camera_relative_world_movements_object_{dynamic_object_idx}", generated_camera_relative_world_movements[dynamic_object_idx])
                    accumulator.add(f"generated_inferred_actions_object_{dynamic_object_idx}", generated_inferred_actions[dynamic_object_idx])
                    accumulator.add(f"generated_inferred_world_actions_object_{dynamic_object_idx}", generated_inferred_world_actions[dynamic_object_idx])

        # Obtains the computed values for each observation in the dataset
        mse_values = accumulator.pop("mse")
        motion_masked_mse_values = accumulator.pop("motion_masked_mse")
        psnr_values = accumulator.pop("psnr")
        ssim_values = accumulator.pop("ssim")
        lpips_values = accumulator.pop("lpips")
        vgg_sim_values = accumulator.pop("vgg_sim")

        # Extracts accumulated values from the accumulator
        movement_values, world_movement_values, camera_relative_world_movement_values, inferred_action_values, inferred_world_action_values = self.pop_movements_and_actions(accumulator, "", dynamic_objects_count)
        generated_movement_values, generated_world_movement_values, generated_camera_relative_world_movement_values, generated_inferred_action_values, generated_inferred_world_action_values = self.pop_movements_and_actions(accumulator, "generated_", dynamic_objects_count)

        # Makes all plots for movements
        self.make_all_movement_plots(movement_values, world_movement_values, camera_relative_world_movement_values, inferred_action_values, inferred_world_action_values, "")
        self.make_all_movement_plots(generated_movement_values, generated_world_movement_values, generated_camera_relative_world_movement_values, generated_inferred_action_values, generated_inferred_world_action_values, "generated_")

        all_reference_detections = accumulator.pop("reference_detections")
        all_generated_detections = accumulator.pop("generated_detections")

        self.logger.print("- Computing detection score")
        #detection_results = self.detection_metric_2d(all_reference_detections, all_generated_detections, "detection")

        # TODO check normalization is correct here
        self.logger.print("- Computing FID score")
        fid_result = self.fid(self.reference_dataloader, self.generated_dataloader)

        self.logger.print("- Computing FVD score")
        fvd_result = self.fvd(self.reference_dataloader, self.generated_dataloader)

        results = {}

        # Computes action accuracy and variation results
        results = self.compute_action_accuracy_and_variation(movement_values, world_movement_values, camera_relative_world_movement_values, inferred_action_values, inferred_world_action_values, results, "")
        results = self.compute_action_accuracy_and_variation(generated_movement_values, generated_world_movement_values, generated_camera_relative_world_movement_values, generated_inferred_action_values, generated_inferred_world_action_values, results, "generated_")

        # Computes the results for each position in the sequence
        mse_results = self.compute_positional_statistics(mse_values, "mse")
        motion_masked_mse_results = self.compute_positional_statistics(motion_masked_mse_values, "motion_masked_mse")
        psnr_results = self.compute_positional_statistics(psnr_values, "psnr")
        ssim_results = self.compute_positional_statistics(ssim_values, "ssim")
        lpips_results = self.compute_positional_statistics(lpips_values, "lpips")
        vgg_sim_results = self.compute_positional_statistics(vgg_sim_values, "vgg_sim")

        # Merges all the results
        results = dict(results, **mse_results)
        results = dict(results, **motion_masked_mse_results)
        results = dict(results, **psnr_results)
        results = dict(results, **ssim_results)
        results = dict(results, **lpips_results)
        results = dict(results, **vgg_sim_results)
        #results = dict(results, **detection_results)
        for object_idx in range(objects_count):
            results[f"mdr_{object_idx}"] = 1 - (total_matched_valid_detections[object_idx] / (total_valid_detections[object_idx] + 1e-6))
            results[f"add_{object_idx}"] = total_matched_distance[object_idx] / (total_matched_valid_detections[object_idx] + 1e-6)
        #results = dict(results, **is_results)
        results["fid"] = fid_result
        results["fvd"] = fvd_result

        return results


def evaluator(config, logger: Logger, reference_dataset: MulticameraVideoDataset, generated_dataset: MulticameraVideoDataset):
    return ReconstructedPlayabilityDatasetEvaluator(config, logger, reference_dataset, generated_dataset)