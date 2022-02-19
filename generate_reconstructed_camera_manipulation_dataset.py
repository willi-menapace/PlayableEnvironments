import argparse
import importlib

import torch.multiprocessing
import torch.nn as nn

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
    search_name = config["model"]["architecture"]
    model = getattr(importlib.import_module(search_name), 'model')(config)
    model.cuda()

    datasets = {}

    dataset_splits = DatasetSplitter.generate_camera_manipulation_dataset_reconstruction_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)
    optical_flow_transformations = OpticalFlowTransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config = dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    # Creates trainer and evaluator
    trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, model, datasets["test"], logger)
    evaluator_inferred_actions = getattr(importlib.import_module(config["evaluation"]["evaluator"]), 'evaluator')(config, datasets["test"], logger, logger_prefix="test")

    # Creates the dataset creators for reconstruction
    test_dataset_creator = getattr(importlib.import_module(config["evaluation"]["camera_manipulation_dataset_creator"]), 'dataset_creator')(config, datasets["test"], logger, name="test")

    # Loads the model
    try:
        trainer.load_checkpoint(model)
    except Exception as e:
        logger.print(e)
        raise Exception("- Cannot find checkpoint to load")
        logger.print()

    model = nn.DataParallel(model)
    model.cuda()

    logger.get_wandb().watch(model, log='all')

    model.eval()

    # Reconstructs the test and validation datasets
    test_dataset_creator.reconstruct_dataset(model)


