from dataset.video_dataset import MulticameraVideoDataset
from evaluation.metrics.tennis_player_detector import TennisPlayerDetector
from evaluation.reconstructed_dataset_evaluator import ReconstructedDatasetEvaluator
from utils.logger import Logger


class ReconstructedTennisDatasetEvaluator(ReconstructedDatasetEvaluator):
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

        super(ReconstructedTennisDatasetEvaluator, self).__init__(config, logger, reference_dataset, generated_dataset)

    def get_player_detector(self):
        '''
        Returns a detector for the players
        :return:
        '''

        player_detector = TennisPlayerDetector()
        player_detector.threshold = 0.60

        return player_detector


def evaluator(config, logger: Logger, reference_dataset: MulticameraVideoDataset, generated_dataset: MulticameraVideoDataset):
    return ReconstructedTennisDatasetEvaluator(config, logger, reference_dataset, generated_dataset)