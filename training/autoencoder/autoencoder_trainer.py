import collections
import os
import re
from typing import Tuple, List, Union, Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn, Batch
from dataset.utils import compute_ground_truth_object_translations
from dataset.video_dataset import MulticameraVideoDataset
from model.autoencoder_models.layers.latent_transformations_helper import LatentTransformationsHelper
from model.utils.object_ids_helper import ObjectIDsHelper
from training.losses import ParallelPerceptualLoss, ImageReconstructionLoss, SquaredL2NormLoss, \
    SpatialKLGaussianDivergenceLoss

from utils.average_meter import AverageMeter
from utils.logger import Logger
from utils.tensor_folder import TensorFolder
from utils.tensor_displayer import TensorDisplayer
from utils.tensor_splitter import TensorSplitter
from utils.time_meter import TimeMeter
from utils.torch_time_meter import TorchTimeMeter


class AutoencoderTrainer:
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: MulticameraVideoDataset, logger: Logger):

        self.config = config
        self.dataset = dataset
        self.logger = logger

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config["training"]["lr_gamma"])
        self.lr_decay_iterations = self.config["training"]["lr_decay_iterations"]

        # Number of steps between each logging
        self.log_interval_steps = config["training"]["log_interval_steps"]

        # Gets the transformation to apply to the bottleneck
        self.bottleneck_transform = LatentTransformationsHelper.transforms_from_config(self.config["training"]["bottleneck_transforms"])

        # Number of preceptual features to use in the computation of the loss
        self.perceptual_features_count = config["training"]["perceptual_features"]
        # Weights to use for the perceptual loss
        self.perceptual_loss_lambda = self.config["training"]["loss_weights"]["perceptual_loss_lambda"]

        # Initializes losses
        self.reconstruction_loss = ImageReconstructionLoss()
        self.squared_l2_norm_loss = SquaredL2NormLoss()
        self.kl_divergence_loss = SpatialKLGaussianDivergenceLoss()

        self.is_variational = False
        if "variational" in model.model_config and model.model_config["variational"]:
            self.is_variational = True

        # Instantiates the perceptual loss if used
        if self.perceptual_loss_lambda > 0.0:
            self.perceptual_loss = ParallelPerceptualLoss(self.perceptual_features_count)

        self.dataloader = DataLoader(dataset, batch_size=self.config["training"]["batching"]["batch_size"], drop_last=True,
                                     shuffle=True, collate_fn=single_batch_elements_collate_fn,
                                     num_workers=self.config["training"]["batching"]["num_workers"], pin_memory=True)

        self.average_meter = AverageMeter()
        self.time_meter = TimeMeter()
        self.torch_time_meter = TorchTimeMeter()
        self.global_step = 0
        self.max_steps = self.config["training"]["max_steps"]

    def _get_current_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def add_noise(input: torch.Tensor, noise_range: List[float]) -> torch.Tensor:
        '''
        Adds noise to the input sampled from a uniform distribution in the specified range
        :param input: input tensor to which apply noise
        :param noise_range: range of the noise
        :return: input tensor with noise applied
        '''

        noise = torch.rand_like(input) * (noise_range[1] - noise_range[0]) + noise_range[0]
        return input + noise

    def save_checkpoint(self, model, name=None):
        '''
        Saves the current training state
        :param model: the model to save
        :param name: the name to give to the checkopoint. If None the default name is used
        :return:
        '''

        if name is None:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], f"{name}_.pth.tar")

        torch.save({"model": model.module.state_dict(), "optimizer": self.optimizer.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(), "step": self.global_step}, filename)

    def load_checkpoint(self, model, name=None):
        """
        Loads the model from a saved state
        :param model: The model to load
        :param name: Name of the checkpoint to use. If None the default name is used
        :return:
        """

        if name is None:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["checkpoints_root_directory"], f"{name}.pth.tar")

        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        loaded_state = torch.load(filename)
        model.load_state_dict(loaded_state["model"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        self.global_step = loaded_state["step"]

    def sum_loss_components(self, components: List[torch.Tensor], weights: Union[List[float], float]) -> List[torch.Tensor]:
        '''
        Produces the weighted sum of the loss components

        :param components: List of scalar tensors
        :param weights: List of weights of the same length of components, or single weight to apply to each component
        :return: Weighted sum of the components
        '''

        components_count = len(components)

        # If the weight is a scalar, broadcast it
        if not isinstance(weights, collections.Sequence):
            weights = [weights] * components_count

        total_sum = components[0] * 0.0
        for current_component, current_weight in zip(components, weights):
            total_sum += current_component * current_weight

        return total_sum

    def compute_losses(self, model, batch: Batch, bottleneck_transform: Callable=None) -> Tuple:
        '''
        Computes losses using the full model

        :param model: The network model
        :param batch: Batch of data
        :param bottleneck_transform: transformation to apply to the bottleneck features
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        # Computes forward and losses for the plain batch
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

        # Forwards the batch through the model
        results = model(observations, bottleneck_transform)

        reconstructed_observations = results["reconstructed_observations"]
        encoded_observations = results["encoded_observations"]

        # To support multiple resolution autoencoders, always convert output to a list
        if torch.is_tensor(encoded_observations):
            encoded_observations = [encoded_observations]

        # Loss information
        loss_info = {}

        total_loss = torch.zeros((), dtype=torch.float32).cuda()

        # Computes the reconstruction loss
        reconstruction_loss = self.reconstruction_loss(observations, reconstructed_observations)
        loss_info[f"reconstruction_loss"] = reconstruction_loss.item()
        total_loss += self.config["training"]["loss_weights"]["reconstruction_loss_lambda"] * reconstruction_loss

        # Computes KL loss for encoded observations at each resolution if needed
        for encoded_observations_idx, current_encoded_observation_mean in enumerate(encoded_observations):
            if self.is_variational:
                current_encoded_observations = current_encoded_observation_mean
                current_encoded_observations_mean, current_encoded_observations_log_variance = TensorSplitter.split(current_encoded_observations, dim=-3, factor=2)
                kl_divergence_loss = self.kl_divergence_loss(current_encoded_observations)
                loss_info[f"KL_loss_{encoded_observations_idx}"] = kl_divergence_loss.item()
                total_loss += self.config["training"]["loss_weights"]["KL_loss_lambda"] * kl_divergence_loss

            current_encoded_observations_squared_l2_norm_loss = self.squared_l2_norm_loss(current_encoded_observations_mean)
            loss_info[f"encoded_observations_squared_l2_norm_loss_{encoded_observations_idx}"] = current_encoded_observations_squared_l2_norm_loss.item()
            total_loss += self.config["training"]["loss_weights"]["encoded_observations_squared_l2_norm_loss_lambda"] * current_encoded_observations_squared_l2_norm_loss

            # Computes additional information
            with torch.no_grad():
                encoded_observations_l2_norm = torch.sqrt(current_encoded_observations_mean.pow(2).sum(-3)).mean()
                loss_info[f"encoded_observations_l2_norm_{encoded_observations_idx}"] = encoded_observations_l2_norm.item()

                if self.is_variational:
                    encoded_observations_variance_magnitude = torch.abs(torch.exp(current_encoded_observations_log_variance)).mean()
                    loss_info[f"encoded_observations_variance_magnitude_{encoded_observations_idx}"] = encoded_observations_variance_magnitude.item()

        # Computes the perceptual loss if used
        if self.perceptual_loss_lambda > 0.0:

            # Brings reconstructed observations in the range [-1, +1]
            normalized_reconstructed_observations = (reconstructed_observations - 0.5) / 0.5

            perceptual_loss, perceptual_loss_components = self.perceptual_loss(observations, normalized_reconstructed_observations)
            perceptual_loss_term = self.sum_loss_components(perceptual_loss_components, self.perceptual_loss_lambda)

            # Adds the perceptual loss to the total loss
            # Weights are already accounted for in the perceptual loss term
            total_loss += perceptual_loss_term

            loss_info[f"perceptual_loss"] = perceptual_loss.item()
            for loss_component_idx, current_perceptual_loss_component in enumerate(perceptual_loss_components):
                loss_info[f"perceptual_loss_{loss_component_idx}"] = current_perceptual_loss_component.item()

        additional_info = {}
        loss_info["loss"] = total_loss.item()

        return total_loss, loss_info, additional_info

    def zero_grad_with_none(self, optimizer: Optimizer):
        '''
        Puts the gradient of each optimized parameter to None
        :param optimizer: The optimizer whose parameters must be put to none
        :return:
        '''

        for current_param_group in optimizer.param_groups:
            for current_parameter in current_param_group["params"]:
                current_parameter.grad = None

    def train_epoch(self, model):

        self.logger.print(f'== Train [{self.global_step}] ==')

        # Number of training steps performed in this epoch
        performed_steps = 0
        for step, batch in enumerate(self.dataloader):
            # If the maximum number of training steps per epoch is exceeded, we interrupt the epoch
            if performed_steps > self.config["training"]["max_steps_per_epoch"]:
                break

            self.global_step += 1
            performed_steps += 1

            # Starts times
            self.torch_time_meter.end("data_loading")
            self.time_meter.start()

            # Computes losses
            loss, loss_info, additional_info = self.compute_losses(model, batch, self.bottleneck_transform)

            self.torch_time_meter.start("backwards")

            # Sets the gradients to None instead of zeroing them so that Adam would not get incorrect
            # estimations for the gradient second moments if a tensor is not used in the current forward pass
            self.zero_grad_with_none(self.optimizer)

            loss.backward()
            self.torch_time_meter.end("backwards")

            self.torch_time_meter.start("optimizer")
            self.optimizer.step()
            self.torch_time_meter.end("optimizer")

            # Triggers update of the learning rate
            if self.global_step % self.lr_decay_iterations == 0:
                self.lr_scheduler.step()
                self.logger.print(f"- Learning rate updated to {self.lr_scheduler.get_lr()}")

            # Stops times
            self.time_meter.end()

            # Accumulates loss information
            self.average_meter.add(loss_info)

            # Logs information at regular intervals
            if (self.global_step - 1) % self.log_interval_steps == 0:

                self.logger.print(f'step: {self.global_step}/{self.max_steps}', end=" ")

                average_time = self.time_meter.get_average_time()
                iterations_per_second = 1 / average_time
                self.logger.print(f'ips: {iterations_per_second:.3f}', end=" ")

                for timer_key in sorted(self.torch_time_meter.keys()):
                    current_time = self.torch_time_meter.get_time(timer_key)
                    self.logger.print(f'perf/{timer_key}: {current_time:.4f}', end=" ")

                average_values = {description: self.average_meter.pop(description) for description in self.average_meter.keys()}
                for description, value in average_values.items():
                    self.logger.print("{}:{:.3f}".format(description, value), end=" ")

                current_lr = self._get_current_lr()
                self.logger.print('lr: %.4f' % (current_lr))

                # Logs on wandb
                wandb = self.logger.get_wandb()
                logged_map = {"train/" + description: item for description, item in average_values.items()}
                logged_map["ips"] = iterations_per_second
                logged_map["step"] = self.global_step
                logged_map["train/lr"] = current_lr
                wandb.log(logged_map, step=self.global_step)
                additional_info["step"] = self.global_step
                wandb.log(additional_info, step=self.global_step)

            self.torch_time_meter.start("data_loading")


def trainer(config, model, dataset, logger):
    return AutoencoderTrainer(config, model, dataset, logger)

