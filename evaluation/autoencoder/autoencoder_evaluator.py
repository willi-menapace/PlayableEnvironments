from typing import Callable

import torch
from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import MulticameraVideoDataset
from model.autoencoder_models.layers.latent_transformations_helper import LatentTransformationsHelper
from training.autoencoder.autoencoder_trainer import AutoencoderTrainer
from utils.average_meter import AverageMeter
from utils.drawing.image_helper import ImageHelper


class AutoencoderEvaluator:
    '''
    Helper class for model evaluation
    '''

    def __init__(self, config, trainer: AutoencoderTrainer, dataset: MulticameraVideoDataset, logger, logger_prefix="test"):

        self.config = config
        self.trainer = trainer
        self.logger = logger
        self.logger_prefix = logger_prefix
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"],
                                     shuffle=True, collate_fn=single_batch_elements_collate_fn,
                                     num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)
        self.imaging_dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"],
                                             shuffle=True, collate_fn=single_batch_elements_collate_fn,
                                             num_workers=self.config["evaluation"]["batching"]["num_workers"],
                                             pin_memory=True)

        # Gets the transformation to apply to the bottleneck
        self.bottleneck_transform = []
        for current_transform_configuration in self.config["evaluation"]["bottleneck_transforms"]:
            current_transform = LatentTransformationsHelper.transforms_from_config(current_transform_configuration)
            self.bottleneck_transform.append(current_transform)

        self.average_meter = AverageMeter()

        # Helper for logging the images
        self.image_helper = ImageHelper(self.config, logger, logger_prefix)

        # Maximum number of batches to use for evaluation
        self.max_batches = self.config["evaluation"]["max_evaluation_batches"]

    def evaluate(self, model, step: int):
        '''
        Evaluates the performances of the given model for all the bottleneck transformations

        :param model: The model to evaluate
        :param step: The current step
        :return:
        '''

        # Evaluates with no transformations
        self.evaluate_inner(model, step, bottleneck_transform=None)
        # Evaluates with every transformationn
        for current_transformation in self.bottleneck_transform:
            self.evaluate_inner(model, step, current_transformation)

    def evaluate_inner(self, model, step: int, bottleneck_transform: Callable=None):
        '''
        Evaluates the performances of the given model using the given bottleneck transformation

        :param model: The model to evaluate
        :param step: The current step
        :return:
        '''

        self.logger.print(f'== Evaluation [{self.trainer.global_step}] ==')

        # Computes the name of the current transformations
        transform_prefix = ""
        if bottleneck_transform is not None:
            transform_prefix = f"{bottleneck_transform.transformation_name}_"

        # Saves sample images
        with torch.no_grad():

            self.logger.print(f"- Saving sample images")
            for idx, batch in enumerate(self.imaging_dataloader):

                # Performs inference
                batch_tuple = batch.to_tuple()
                observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

                # Reconstructs the observations using the model
                results = model(observations, bottleneck_transform)

                reconstructed_observations = results["reconstructed_observations"]

                ground_truth_observations = observations
                ground_truth_observations = ImageHelper.normalize(ground_truth_observations, (-1, +1))

                # Saves images and reconsutrcted images. Puts the observations as the columns of the produces image instead of the cameras
                self.image_helper.save_image_pairs(ground_truth_observations, reconstructed_observations, step, f"{transform_prefix}reconstructed_observations", cameras_as_columns=False)

                break

            # Computes validaton losses and logs them
            for idx, batch in enumerate(self.dataloader):

                # Breaks if the maximum number of batches is reached
                if self.max_batches is not None and idx > self.max_batches:
                    break

                loss, loss_info, additional_info = self.trainer.compute_losses(model, batch, bottleneck_transform)

                # Accumulates loss information
                self.average_meter.add(loss_info)

            self.logger.print(f'step: {self.trainer.global_step}/{self.trainer.max_steps}', end=" ")

            average_values = {description: self.average_meter.pop(description) for description in self.average_meter.keys()}
            for description, value in average_values.items():
                self.logger.print("{}:{:.3f}".format(description, value), end=" ")
            self.logger.print("")

            # Logs on wandb
            wandb = self.logger.get_wandb()
            logged_map = {f"{self.logger_prefix}/" + f"{transform_prefix}{description}": item for description, item in average_values.items()}
            logged_map["step"] = self.trainer.global_step
            wandb.log(logged_map, step=self.trainer.global_step)
            additional_info["step"] = self.trainer.global_step
            wandb.log(additional_info, step=self.trainer.global_step)

        return


def evaluator(config, trainer: AutoencoderTrainer, dataset: MulticameraVideoDataset, logger, logger_prefix="test"):
    return AutoencoderEvaluator(config, trainer, dataset, logger, logger_prefix)
