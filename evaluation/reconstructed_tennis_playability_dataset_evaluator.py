import torch

from dataset.video_dataset import MulticameraVideoDataset
from evaluation.metrics.tennis_player_detector import TennisPlayerDetector
from evaluation.reconstructed_playability_dataset_evaluator import ReconstructedPlayabilityDatasetEvaluator
from utils.logger import Logger


class ReconstructedTennisPlayabilityDatasetEvaluator(ReconstructedPlayabilityDatasetEvaluator):
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

        super(ReconstructedTennisPlayabilityDatasetEvaluator, self).__init__(config, logger, reference_dataset, generated_dataset)

    def get_player_detector(self):
        '''
        Returns a detector for the players
        :return:
        '''

        player_detector = TennisPlayerDetector()
        player_detector.threshold = 0.60

        return player_detector

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
    return ReconstructedTennisPlayabilityDatasetEvaluator(config, logger, reference_dataset, generated_dataset)