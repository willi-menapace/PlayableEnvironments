import torch

from dataset.video_dataset import MulticameraVideoDataset
from evaluation.metrics.minecraft_player_detector_2 import MinecraftPlayerDetector
from evaluation.reconstructed_playability_dataset_evaluator import ReconstructedPlayabilityDatasetEvaluator
from utils.logger import Logger


class ReconstructedMinecraftPlayabilityDatasetEvaluator(ReconstructedPlayabilityDatasetEvaluator):
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

        super(ReconstructedMinecraftPlayabilityDatasetEvaluator, self).__init__(config, logger, reference_dataset, generated_dataset)

    def get_player_detector(self):
        '''
        Returns a detector for the players
        :return:
        '''

        player_detector = MinecraftPlayerDetector(self.config["evaluation"]["minecraft_detector_weights_filename"])

        return player_detector

    def compensate_bounding_boxes(self, bounding_boxes: torch.Tensor):
        '''
        Applies compensation to ground truth bounding boxes if needed

        :param bounding_boxes: (..., 4, dynamic_objects_count) tensor with boudning boxes
        :return:
        '''

        expansion_factor_cols = 1.0
        expansion_factor_rows = 2.6

        # Avoids modification to be reflected to the original tensor
        bounding_boxes = bounding_boxes.clone()

        bounding_boxes_dimensions = bounding_boxes[..., 2:, :] - bounding_boxes[..., :2, :]
        bounding_boxes[..., 0, :] -= bounding_boxes_dimensions[..., 0, :] * expansion_factor_cols
        bounding_boxes[..., 2, :] += bounding_boxes_dimensions[..., 0, :] * expansion_factor_cols
        bounding_boxes[..., 1, :] -= bounding_boxes_dimensions[..., 1, :] * expansion_factor_rows
        # Do not expand the bounding boxes to the bottom

        bounding_boxes = torch.clamp(bounding_boxes, min=0.0, max=1.0)
        return bounding_boxes

    def is_movement_valid(self, movement: torch.Tensor) -> bool:
        '''

        :param movement: (2) tensor with translation
        :return: True if the current translation is valid for the current dataset
        '''

        # Checks that the object is not too far from the center
        center_distance = torch.sqrt(movement.pow(2).sum()).item()
        if center_distance > 2.0:
            return False
        else:
            return True


def evaluator(config, logger: Logger, reference_dataset: MulticameraVideoDataset, generated_dataset: MulticameraVideoDataset):
    return ReconstructedMinecraftPlayabilityDatasetEvaluator(config, logger, reference_dataset, generated_dataset)