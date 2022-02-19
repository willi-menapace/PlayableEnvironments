from typing import Dict

from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import MulticameraVideoDataset
from evaluation.metrics.fvd import IncrementalFVD
from utils.logger import Logger


class ReconstructedDatasetFVDEvaluator:
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

        self.fvd = IncrementalFVD()

    def compute_metrics(self) -> Dict:
        '''
        Computes evaluation metrics on the given datasets

        :return: Dictionary with evaluation results
        '''

        self.logger.print("- Computing FVD score")
        fvd_result = self.fvd(self.reference_dataloader, self.generated_dataloader)

        results = {}
        results["fvd"] = fvd_result

        return results


def evaluator(config, logger: Logger, reference_dataset: MulticameraVideoDataset, generated_dataset: MulticameraVideoDataset):
    return ReconstructedDatasetFVDEvaluator(config, logger, reference_dataset, generated_dataset)