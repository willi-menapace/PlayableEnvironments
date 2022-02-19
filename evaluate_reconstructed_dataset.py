import argparse
import importlib
import os

import torch.multiprocessing
import yaml

from dataset.dataset_splitter import DatasetSplitter
from dataset.transforms import TransformsGenerator, OpticalFlowTransformsGenerator
from dataset.video_dataset import MulticameraVideoDataset
from utils.configuration import Configuration
from utils.logger import Logger

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = parser.parse_args()

    config_path = arguments.config

    configuration = Configuration(config_path)
    configuration.check_config()
    configuration.create_directory_structure()

    config = configuration.get_config()

    logger = Logger(config)

    reference_datasets = {}
    generated_datasets = {}

    reference_dataset_splits = DatasetSplitter.generate_evaluate_reconstructed_dataset_splits(config, "reference")
    generated_dataset_splits = DatasetSplitter.generate_evaluate_reconstructed_dataset_splits(config, "generated")
    transformations = TransformsGenerator.get_reconstructed_dataset_evaluation_transforms(config)
    optical_flow_transformations = OpticalFlowTransformsGenerator.get_final_transforms(config)

    for key in reference_dataset_splits:
        path, batching_config = reference_dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        reference_datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    for key in generated_dataset_splits:
        path, batching_config = generated_dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        generated_datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    # Creates the evaluator
    evaluator_inferred_actions = getattr(importlib.import_module(config["evaluation"]["dataset_reconstruction_evaluator"]), 'evaluator')(config, logger, reference_datasets["test"], generated_datasets["test"])

    evaluation_results = evaluator_inferred_actions.compute_metrics()

    output_path = os.path.join(config["logging"]["output_directory"], "reconstructed_dataset_evaluation.yaml")
    with open(output_path, 'w') as outfile:
        yaml.dump(evaluation_results, outfile)

    print("- All done")



