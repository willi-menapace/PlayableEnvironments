from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import MulticameraVideoDataset
from evaluation.metrics.action_linear_classification import ActionClassificationScore
from evaluation.metrics.action_variance import ActionVariance
from evaluation.metrics.fid import FID
from evaluation.metrics.lpips import LPIPS
from evaluation.metrics.metrics_accumulator import MetricsAccumulator
from evaluation.metrics.motion_masked_mse import MotionMaskedMSE
from evaluation.metrics.mse import MSE
from evaluation.metrics.psnr import PSNR
from evaluation.metrics.ssim import SSIM
from evaluation.metrics.vgg_cosine_similarity import VGGCosineSimilarity
from utils.logger import Logger


class ReconstructedDatasetEvaluator:
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

        self.normalization_transform = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.player_detector = self.get_player_detector()

        self.action_variance = ActionVariance()
        self.action_accuracy = ActionClassificationScore()

        self.mse = MSE()
        self.motion_masked_mse = MotionMaskedMSE()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.vgg_sim = VGGCosineSimilarity()
        self.fid = FID()
        #self.inception_score = InceptionScore()
        #self.fvd = IncrementalFVD()

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

    def compute_movements_and_actions(self, reference_detections: np.ndarray, generated_batch):
        '''


        :param reference_detections: (bs, observations_count, 2) tensor with x and y coordinates of the detection, -1 if any
        :param generated_batch: batch of generated data
        :return: (detected_movements, 2), (detected_movements) arrays with detected movements,
                 actions inferred by the model corresponding to the detected movements
        '''
        movements = []
        inferred_actions = []

        batch_size = reference_detections.shape[0]
        observations_count = reference_detections.shape[1]

        for sequence_idx in range(batch_size):
            for observation_idx in range(observations_count - 1):
                # If there was a successful detection for both the current and the successive frame
                if reference_detections[sequence_idx, observation_idx, 0] != -1 and \
                   reference_detections[sequence_idx, observation_idx + 1, 0] != -1:
                    # Extract movement and action
                    current_movement = reference_detections[sequence_idx, observation_idx + 1] - reference_detections[sequence_idx, observation_idx]
                    current_action = generated_batch.video[sequence_idx].metadata[:-1][observation_idx]["inferred_action"]

                    movements.append(current_movement)
                    inferred_actions.append(current_action)

        return np.asarray(movements, dtype=np.float), np.asarray(inferred_actions, dtype=np.int)

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

                # Extracts generated data
                generated_batch_tuple = generated_batch.to_tuple()
                generated_observations, generated_actions, generated_rewards, generated_dones, generated_camera_rotations,\
                generated_camera_translations, generated_focals, generated_bounding_boxes, generated_bounding_boxes_validity,\
                generated_global_frame_indexes, generated_video_frame_indexes, generated_video_indexes = generated_batch_tuple

                # Applies compensation to the bounding boxes if needed
                reference_bounding_boxes = self.compensate_bounding_boxes(reference_bounding_boxes)
                generated_bounding_boxes = self.compensate_bounding_boxes(generated_bounding_boxes)

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

        # Obtains the computed values for each observation in the dataset
        mse_values = accumulator.pop("mse")
        motion_masked_mse_values = accumulator.pop("motion_masked_mse")
        psnr_values = accumulator.pop("psnr")
        ssim_values = accumulator.pop("ssim")
        lpips_values = accumulator.pop("lpips")
        vgg_sim_values = accumulator.pop("vgg_sim")

        all_reference_detections = accumulator.pop("reference_detections")
        all_generated_detections = accumulator.pop("generated_detections")

        self.logger.print("- Computing detection score")
        #detection_results = self.detection_metric_2d(all_reference_detections, all_generated_detections, "detection")

        # TODO check normalization is correct here
        self.logger.print("- Computing FID score")
        fid_result = self.fid(self.reference_dataloader, self.generated_dataloader)

        #self.logger.print("- Computing FVD score")
        #fvd_result = self.fvd(self.reference_dataloader, self.generated_dataloader)

        results = {}
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
        #results["fvd"] = fvd_result

        return results


def evaluator(config, logger: Logger, reference_dataset: MulticameraVideoDataset, generated_dataset: MulticameraVideoDataset):
    return ReconstructedDatasetEvaluator(config, logger, reference_dataset, generated_dataset)