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

    logger = Logger(config)
    search_name = config["model"]["architecture"]
    model = getattr(importlib.import_module(search_name), 'model')(config)
    model.cuda()

    datasets = {}

    dataset_splits = DatasetSplitter.generate_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)
    optical_flow_transformations = OpticalFlowTransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config = dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    # Creates trainer and evaluator
    trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, model, datasets["train"], logger)
    evaluator_inferred_actions = getattr(importlib.import_module(config["evaluation"]["evaluator"]), 'evaluator')(config, datasets["validation"], logger, logger_prefix="validation")

    # Resume training
    try:
        trainer.load_checkpoint(model)
    except Exception as e:
        logger.print(e)
        logger.print("- Warning: training without loading saved checkpoint")

    model = nn.DataParallel(model)
    model.cuda()

    logger.get_wandb().watch(model, log='all')

    last_save_step = 0
    last_quick_save_step = 0
    last_eval_step = 0

    evaluations_counter = 1  # Number of performed evaluations

    # Makes the model parallel and train
    while trainer.global_step < config["training"]["max_steps"] or force_eval:

        if not force_eval:
            model.train()

            trainer.train_epoch(model)

            # Saves the model
            if trainer.global_step > last_quick_save_step + 500:
                trainer.save_checkpoint(model)
                last_quick_save_step = trainer.global_step
            if trainer.global_step > last_save_step + config["training"]["save_freq"]:
                trainer.save_checkpoint(model, f"checkpoint_{trainer.global_step}")
                last_save_step = trainer.global_step

        model.eval()

        # Evaluates the model
        if trainer.global_step > last_eval_step + config["evaluation"]["eval_freq"] or force_eval:

            # Each 10 evaluations, log complete images for each object in the scene
            log_only_global = True
            if evaluations_counter % 10 == 0:
                log_only_global = False

            evaluator_inferred_actions.evaluate(model, trainer.global_step, log_only_global)
            last_eval_step = trainer.global_step
            evaluations_counter += 1

