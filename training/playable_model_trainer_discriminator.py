import os
from typing import Tuple

import os
from typing import Tuple

import torch

from dataset.batching import Batch
from dataset.video_dataset import MulticameraVideoDataset
from training.losses import GANLoss
from training.playable_model_trainer import PlayableModelTrainer
from utils.logger import Logger


class PlayableModelTrainerDiscriminator(PlayableModelTrainer):
    '''
    Helper class for model training
    '''

    def __init__(self, config, model, dataset: MulticameraVideoDataset, logger: Logger):

        super(PlayableModelTrainerDiscriminator, self).__init__(config, model, dataset, logger)

        # Instantiates the discriminator optimizer and learning rate schedulers
        self.discriminator_optimizer = self.get_discriminator_optimizer(model)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, self.config["playable_model_training"]["lr_gamma"])
        self.discriminator_lr_decay_iterations = self.config["playable_model_training"]["lr_decay_iterations"]
        self.fix_discriminator_lr_update = self.config["playable_model_training"]["fix_discriminator_lr_update"]

        # Instantiates losses for the discriminator
        gan_mode = self.config["playable_model_training"]["gan_mode"]
        self.gan_loss = GANLoss(gan_mode).to("cuda:0")

    def get_discriminator_optimizer(self, model):
        '''
        Creates the optimizer for the discriminators

        :param model:
        :return:
        '''

        betas = self.config["playable_model_training"]["betas"]

        return torch.optim.Adam(model.discriminator_parameters(), lr=self.config["playable_model_training"]["discriminator_learning_rate"], weight_decay=self.config["playable_model_training"]["discriminator_weight_decay"], betas=betas)

    def save_checkpoint(self, model, name=None):
        '''
        Saves the current training state
        :param model: the model to save
        :param name: the name to give to the checkopoint. If None the default name is used
        :return:
        '''

        if name is None:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], f"{name}_.pth.tar")

        torch.save({"model": model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                    "discriminator_lr_scheduler": self.discriminator_lr_scheduler.state_dict(),
                    "mi_estimator": self.mutual_information_loss_modules.state_dict(),
                    "step": self.global_step},
                    filename)

    def load_checkpoint(self, model, name=None):
        """
        Loads the model from a saved state
        :param model: The model to load
        :param name: Name of the checkpoint to use. If None the default name is used
        :return:
        """

        if name is None:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], "latest.pth.tar")
        else:
            filename = os.path.join(self.config["logging"]["playable_model_checkpoints_directory"], f"{name}.pth.tar")

        if not os.path.isfile(filename):
            raise Exception(f"Cannot load model: no checkpoint found at '{filename}'")

        loaded_state = torch.load(filename)
        model.load_state_dict(loaded_state["model"])
        self.optimizer.load_state_dict(loaded_state["optimizer"])
        self.lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        self.discriminator_optimizer.load_state_dict(loaded_state["discriminator_optimizer"])
        self.discriminator_lr_scheduler.load_state_dict(loaded_state["discriminator_lr_scheduler"])
        self.mutual_information_loss_modules.load_state_dict(loaded_state["mi_estimator"])
        self.global_step = loaded_state["step"]

    def compute_losses_from_forward_results(self, *forward_results):
        '''
        Computes losses using the full model

        :param forward_results: results of calling forward on the model for which to compute the losses
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
                  nonlogged_additional_info: Dict with additional nonloggable information
        '''

        # Calls the superclass implementation
        total_loss, loss_info, additional_info, nonlogged_additional_info = super(PlayableModelTrainerDiscriminator, self).compute_losses_from_forward_results(*forward_results)

        scene_encoding, object_results = forward_results

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count
        # Computes losses for each dynamic object
        for dynamic_object_idx in range(dynamic_objects_count):
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_prefix = f"object_{object_idx}_"

            # Gets animation results for the current object
            current_object_results = object_results[dynamic_object_idx]

            # Asks the generated sequences to be predicted as real
            discriminator_output = current_object_results["discriminator_output"]
            current_gan_loss = self.gan_loss(discriminator_output, True)

            current_object_loss = self.config["playable_model_training"]["loss_weights"]["gan_loss_lambda"] * current_gan_loss

            total_loss += current_object_loss

            loss_info[current_prefix + "loss"] = loss_info[current_prefix + "loss"] + current_object_loss.item()  # Updates the total loss for the current object
            loss_info[current_prefix + "gan_generator_loss"] = current_gan_loss.item()
            loss_info[current_prefix + "gan_generator_D(G(z))"] = torch.sigmoid(discriminator_output).mean().item()

        return total_loss, loss_info, additional_info, nonlogged_additional_info

    def compute_discriminator_losses(self, model, scene_encoding, object_results) -> Tuple:
        '''
        Computes losses for the discriminator starting from results of the previous forward pass

        :param model: the model
        :param scene_encoding: scene encoding obtained by the model from the forward pass
        :param object_results: object results obtained by the model from the forward pass
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
                  nonlogged_additional_info: Dict with additional nonloggable information
        '''

        discriminator_results = model(scene_encoding, object_results, mode="only_discriminator")

        loss_info = {}
        total_loss = torch.zeros((), dtype=torch.float32).cuda()

        # Computes losses for each dynamic object
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count
        for dynamic_object_idx in range(dynamic_objects_count):
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)
            current_prefix = f"object_{object_idx}_"

            # Gets discriminator results for the current object
            current_object_discriminator_results = discriminator_results[dynamic_object_idx]

            discriminator_output_fake = current_object_discriminator_results["discriminator_output_fake"]
            discriminator_output_real = current_object_discriminator_results["discriminator_output_real"]

            gan_loss_fake = self.gan_loss(discriminator_output_fake, False)
            gan_loss_real = self.gan_loss(discriminator_output_real, True)
            current_gan_loss = gan_loss_fake + gan_loss_real

            current_object_loss = self.config["playable_model_training"]["loss_weights"]["discriminator_gan_loss_lambda"] * current_gan_loss

            total_loss += current_object_loss

            loss_info[current_prefix + "discriminator_loss"] = current_object_loss.item()
            loss_info[current_prefix + "gan_discriminator_loss"] = current_gan_loss.item()
            loss_info[current_prefix + "gan_discriminator_D(G(z))"] = torch.sigmoid(discriminator_output_fake).mean().item()
            loss_info[current_prefix + "gan_discriminator_D(x)"] = torch.sigmoid(discriminator_output_real).mean().item()

        additional_info = {}
        nonlogged_additional_info = {}

        return total_loss, loss_info, additional_info, nonlogged_additional_info

    def lr_scheduler_step(self):

        # Triggers update of the learning rate
        if self.global_step % self.lr_decay_iterations == 0:
            self.lr_scheduler.step()
            self.logger.print(f"- Learning rate updated to {self.lr_scheduler.get_lr()}")

        if self.fix_discriminator_lr_update:
            if self.global_step % self.discriminator_lr_decay_iterations == 0:
                self.discriminator_lr_scheduler.step()
                self.logger.print(f"- Discriminator learning rate updated to {self.discriminator_lr_scheduler.get_lr()}")

    def optimization_step(self, model, batch: Batch):
        '''
        Performs an optimization step
        :param model: the model
        :param batch: batch of data
        :return: (total_loss, loss_info)
                  total_loss: torch.Tensor with the total loss
                  loss_info: Dict with an entry for every additional information about the loss
                  additional_info: Dict with additional loggable information
        '''

        ###### Optimizes the generator

        model.module.set_discriminator_requires_grad(False)

        # Computes losses. Calls the forward pass method using also the discriminator
        generator_loss, generator_loss_info, generator_additional_info, nonlogged_additional_info = self.compute_losses(model, batch, forward_mode="vanilla_plus_discriminator")

        self.optimizer.zero_grad()
        generator_loss.backward()
        self.optimizer.step()

        ###### Optimizes the discriminator

        model.module.set_discriminator_requires_grad(True)

        scene_encoding = nonlogged_additional_info["scene_encoding"]
        object_results = nonlogged_additional_info["object_results"]
        discriminator_loss, discriminator_loss_info, discriminator_additional_info, discriminator_nonlogged_additional_info = self.compute_discriminator_losses(model, scene_encoding, object_results)

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Merges the results
        loss = generator_loss + discriminator_loss
        loss_info = dict(generator_loss_info, **discriminator_loss_info)
        additional_info = dict(generator_additional_info, **discriminator_additional_info)

        return loss, loss_info, additional_info


def trainer(config, model, dataset, logger):
    return PlayableModelTrainerDiscriminator(config, model, dataset, logger)

