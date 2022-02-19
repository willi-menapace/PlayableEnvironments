import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class TransformsGenerator:

    @staticmethod
    def check_and_resize(target_crop: List[int], target_size: Tuple[int]):
        '''
        Creates a function that transforms input PIL images to the target size
        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) touple representing the target height and width
        :return: function that transforms a PIL image to the target size
        '''

        # Creates the transformation function
        def transform(image: Image):
            if target_crop is not None:
                image = image.crop(target_crop)
            if image.size != tuple(target_size):
                image = image.resize(target_size, Image.BILINEAR)

            return image

        return transform

    @staticmethod
    def to_float_tensor(tensor):
        return tensor / 1.0

    @staticmethod
    def sample_augmentation_transform(batching_config: Dict):
        '''
        Samples an augmenting transformation from PIL.Image to PIL.Image that can be applied to multiple images
        with the same effect
        :param batching_config: Dict with the batching parameters to use for sampling. Must contain rotation_range,
                                scale_range and translation_range
        :return: function from PIL.Image to PIL.Image representing the samples augmentation transformation
        '''

        rotation_range = batching_config["rotation_range"]
        translation_range = batching_config["translation_range"]
        scale_range = batching_config["scale_range"]

        # Samples transformation parameters
        sampled_translation = [random.uniform(*translation_range),
                               random.uniform(*translation_range),
                               ]
        sampled_rotation = random.uniform(*rotation_range)
        sampled_scale = random.uniform(*scale_range)

        # Builds the transformation function
        def composed_transform(img):
            return transforms.functional.affine(img, sampled_rotation, sampled_translation, sampled_scale,
                                                shear=0, resample=Image.BILINEAR, fillcolor=None)

        return composed_transform

    @staticmethod
    def get_final_transforms(config):
        '''
        Obtains the transformations to use for training and evaluation
        :param config: The configuration file
        :return:
        '''

        resize_transform = TransformsGenerator.check_and_resize(config["data"]["crop"], config["data"]["target_input_size"])
        transform = transforms.Compose([resize_transform,
                                        transforms.ToTensor(),
                                        TransformsGenerator.to_float_tensor,
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }

    @staticmethod
    def get_reconstructed_dataset_evaluation_transforms(config):
        '''
        Obtains the transformations to use when evaluating the reconstructed dataset
        :param config: The configuration file
        :return:
        '''

        resize_transform = TransformsGenerator.check_and_resize(config["data"]["crop"], config["data"]["target_input_size"])
        transform = transforms.Compose([resize_transform,
                                        transforms.ToTensor(),
                                        TransformsGenerator.to_float_tensor,
                                        # Do not apply normalization
                                        ])

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }


class OpticalFlowTransformsGenerator:

    @staticmethod
    def numpy_to_torch(data: np.ndarray):
        torch_data = torch.from_numpy(data)

        return torch_data

    @staticmethod
    def check_and_resize(target_crop: List[int], target_size: Tuple[int]):
        '''
        Creates a function that transforms input tensor to the target size
        :param target_crop: [left_index, upper_index, right_index, lower_index] list representing the crop region
        :param target_size: (width, height) touple representing the target height and width
        :return: function that transforms a tensor to the target size
        '''

        # Creates the transformation function
        def transform(image: torch.Tensor):
            '''

            :param image: (channels, height, width) tensor
            :return:
            '''

            # Crops the image if required
            if target_crop is not None:
                left, top, right, bottom = target_crop
                image = image[:, top:bottom, left:right]

            # Scales the image if the required size is different from the original one
            image_height = image.size(-2)
            image_width = image.size(-1)
            target_width, target_height = target_size
            if (image_width, image_height) != (target_width, target_height):
                image = image.unsqueeze(0)
                image = F.interpolate(image, (target_height, target_width), mode="bilinear")
                image = image.squeeze(0)
            return image

        return transform

    @staticmethod
    def get_final_transforms(config):
        '''
        Obtains the transformations to use for training and evaluation
        :param config: The configuration file
        :return:
        '''

        resize_transform = OpticalFlowTransformsGenerator.check_and_resize(config["data"]["crop"], config["data"]["target_input_size"])
        transform = transforms.Compose([OpticalFlowTransformsGenerator.numpy_to_torch,
                                        resize_transform,
                                        ])

        return {
            "train": transform,
            "validation": transform,
            "test": transform,
        }


class AutoencoderTransformsGenerator:

    @staticmethod
    def get_transform_set_1():

        transformations = [
            transforms.RandomApply(transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.5), 0.5),
            transforms.RandomApply(transforms.RandomAffine(degrees=15, translate=(0.06, 0.06), scale=(0.9, 1.1)), 0.5),
        ]

        return transformations

    @staticmethod
    def get_transform_by_bottleneck_transform(config):
        '''
        Obtains the transformations to use for training and evaluation
        :param config: The configuration file
        :return:
        '''

        resize_transform = TransformsGenerator.check_and_resize(config["data"]["crop"],
                                                                config["data"]["target_input_size"])
        transforms_list = [resize_transform,
                           transforms.ToTensor(),
                           TransformsGenerator.to_float_tensor,
                           transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

        transform = transforms.Compose(transforms_list)

        # If we have to apply inputr transformations
        if hasattr(config["training"], "input_augmentation_trasformations_set"):
            augmentations_set_idx = config["training"]["input_augmentation_trasformations_set"]
            if augmentations_set_idx == 1:
                transforms_list[1:1] = AutoencoderTransformsGenerator.get_transform_set_1()
            else:
                raise NotImplementedError(f"Unknown transformations set {augmentations_set_idx}")

        training_transform = transforms.Compose(transforms_list)

        return {
            "train": training_transform,
            "validation": transform,
            "test": transform,
        }

    @staticmethod
    def get_final_transforms(config):
        '''
        Obtains the transformations to use for training and evaluation
        :param config: The configuration file
        :return:
        '''

        resize_transform = TransformsGenerator.check_and_resize(config["data"]["crop"], config["data"]["target_input_size"])
        transforms_list = [resize_transform,
                                transforms.ToTensor(),
                                TransformsGenerator.to_float_tensor,
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

        transform = transforms.Compose(transforms_list)

        # If we have to apply inputr transformations
        if hasattr(config["training"], "input_augmentation_trasformations_set"):
            augmentations_set_idx = config["training"]["input_augmentation_trasformations_set"]
            if augmentations_set_idx == 1:
                transforms_list[1:1] = AutoencoderTransformsGenerator.get_transform_set_1()
            else:
                raise NotImplementedError(f"Unknown transformations set {augmentations_set_idx}")

        training_transform = transforms.Compose(transforms_list)

        return {
            "train": training_transform,
            "validation": transform,
            "test": transform,
        }