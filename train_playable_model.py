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
    parser.add_argument('--force_eval', default=False, action='store_true')
    arguments = parser.parse_args()

    config_path = arguments.config
    force_eval = arguments.force_eval

    configuration = Configuration(config_path)
    configuration.check_config()
    configuration.create_directory_structure()

    config = configuration.get_config()

    logger = Logger(config, project="playable-environments-playability")
    search_name = config["model"]["architecture"]
    environment_model = getattr(importlib.import_module(search_name), 'model')(config)
    environment_model.cuda()

    datasets = {}

    dataset_splits = DatasetSplitter.generate_playable_model_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)
    optical_flow_transformations = OpticalFlowTransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config = dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    # Creates trainer and evaluator
    environment_model_trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, environment_model, datasets["train"], logger)

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
    playable_environment_model_trainer = getattr(importlib.import_module(config["playable_model_training"]["trainer"]), 'trainer')(config, playable_environment_model, datasets["train"], logger)
    evaluator_inferred_actions = getattr(importlib.import_module(config["playable_model_evaluation"]["evaluator"]), 'evaluator')(config, playable_environment_model_trainer, datasets["validation"], logger, logger_prefix="playable_model_validation")

    # Loads the playable model
    try:
        playable_environment_model_trainer.load_checkpoint(playable_environment_model)
    except Exception as e:
        logger.print(e)
        logger.print("- Warning: training without loading saved checkpoint")

    playable_environment_model = nn.DataParallel(playable_environment_model)
    playable_environment_model.cuda()

    logger.get_wandb().watch(playable_environment_model, log='all')

    last_save_step = 0
    last_quick_save_step = 0
    last_eval_step = 0

    # Makes the model parallel and train
    while playable_environment_model_trainer.global_step < config["playable_model_training"]["max_steps"] or force_eval:

        if not force_eval:
            playable_environment_model.train()

            playable_environment_model_trainer.train_epoch(playable_environment_model)

            # Saves the model
            if playable_environment_model_trainer.global_step > last_quick_save_step + 500:
                playable_environment_model_trainer.save_checkpoint(playable_environment_model)
                last_quick_save_step = playable_environment_model_trainer.global_step
            if playable_environment_model_trainer.global_step > last_save_step + config["playable_model_training"]["save_freq"]:
                playable_environment_model_trainer.save_checkpoint(playable_environment_model, f"checkpoint_{playable_environment_model_trainer.global_step}")
                last_save_step = playable_environment_model_trainer.global_step

        playable_environment_model.eval()

        # Evaluates the model
        if playable_environment_model_trainer.global_step > last_eval_step + config["playable_model_evaluation"]["eval_freq"] or force_eval:
            evaluator_inferred_actions.evaluate(playable_environment_model, environment_model, playable_environment_model_trainer.global_step)
            last_eval_step = playable_environment_model_trainer.global_step
            pass

