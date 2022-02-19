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
    environment_model = getattr(importlib.import_module(search_name), 'model')(config)
    environment_model.cuda()

    datasets = {}

    dataset_splits = DatasetSplitter.generate_playability_dataset_reconstruction_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)
    optical_flow_transformations = OpticalFlowTransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config = dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    # Creates trainer and evaluator
    environment_model_trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, environment_model, datasets["validation"], logger)

    # Loads the environment model
    try:
        environment_model_trainer.load_checkpoint(environment_model)
    except Exception as e:
        logger.print(e)
        raise Exception("Could not load environment model. Playable model training aborted")

    # Creates the playable model
    environment_model.cuda()
    environment_model.eval()
    search_name = config["playable_model"]["architecture"]
    playable_environment_model = getattr(importlib.import_module(search_name), 'model')(config, environment_model)
    playable_environment_model.cuda()
    # Creates the trainer and evaluator for the playable model
    playable_environment_model_trainer = getattr(importlib.import_module(config["playable_model_training"]["trainer"]), 'trainer')(config, playable_environment_model, datasets["validation"], logger)

    # Loads the playable model
    try:
        playable_environment_model_trainer.load_checkpoint(playable_environment_model)
    except Exception as e:
        logger.print(e)
        logger.print("- Warning: training without loading saved checkpoint")

    playable_environment_model = nn.DataParallel(playable_environment_model)
    playable_environment_model.cuda()

    # Creates the dataset creators for reconstruction
    validation_dataset_creator = getattr(importlib.import_module(config["playable_model_evaluation"]["dataset_creator"]), 'dataset_creator')(config, datasets["validation"], logger, name="val")
    test_dataset_creator = getattr(importlib.import_module(config["playable_model_evaluation"]["dataset_creator"]), 'dataset_creator')(config, datasets["test"], logger, name="test")

    logger.get_wandb().watch(environment_model, log='all')

    playable_environment_model.eval()

    # Reconstructs the test and validation datasets
    test_dataset_creator.reconstruct_dataset(playable_environment_model)
    #validation_dataset_creator.reconstruct_dataset(playable_environment_model)


