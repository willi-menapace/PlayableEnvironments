import collections
import random
from typing import Dict
from functools import partial

import torch
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms.functional as F
from torchvision.transforms import RandomApply

from utils.tensor_folder import TensorFolder


class LatentTransformationsHelper:

    @staticmethod
    def apply_blur(features: torch.Tensor, kernel_size: int, sigma: float=1.5):
        '''
        Applies blur to the features

        :param features: (..., features, height, width) tensor with features
        :param kernel_size: size of the kernel to use
        :param sigma: the sigma for the gaussian filter. Can be a range (min, max)
        :return: (..., features, height, width) tensor with features blurred using the blur kernel of given dimensions
        '''

        features, initial_dimensions = TensorFolder.flatten(features, -3)

        # Applies a different sigma to each batch element if sigma is not fixed
        if isinstance(sigma, collections.Sequence):
            blurred_features = []
            for current_features in features:
                blurred_features.append(F.gaussian_blur(current_features, kernel_size, sigma))
            blurred_features = torch.stack(blurred_features, dim=0)
        else:
            blurred_features = F.gaussian_blur(features, kernel_size, sigma)

        blurred_features = TensorFolder.fold(blurred_features, initial_dimensions)

        return blurred_features

    @staticmethod
    def apply_gaussian_noise(features: torch.Tensor, intensity: float):
        '''
        Applies gaussian noise to the given features.
        Variance of the noise is expressed as var(features) * intensity

        :param features: (..., features, height, width) tensor with features
        :param intensity: fraction of the features variance to use as the variance for the noise to apply
        :return: (..., features, height, width) tensor with features with noise applied
        '''

        flat_features, initial_dimensions = TensorFolder.flatten(features, -3)

        noise_variance = torch.var(features, dim=[-1, -2]) * intensity
        noise_std = torch.sqrt(noise_variance)
        noise = torch.rand(features.size(), device=features.device, dtype=features.dtype) * noise_std.unsqueeze(-1).unsqueeze(-1)

        noisy_features = flat_features + noise

        folded_noisy_features = TensorFolder.fold(noisy_features, initial_dimensions)
        return folded_noisy_features

    @staticmethod
    def apply_cutout(features: torch.Tensor, size: int, min_count: int, max_count: int):
        '''
        Applies cutout to the given features. Cutout does not replace features with random colors, but with random
        features from different parts of the image.

        :param features: (..., features, height, width) tensor with features
        :param size: size of the holes to cut in the image
        :param min_count: minimum number of holes to cut
        :param max_count: maximum number of holes to cut
        :return: (..., features, height, width) tensor with features with holes applied
        '''

        flat_features, initial_dimensions = TensorFolder.flatten(features, -3)

        height = features.size(-2)
        width = features.size(-1)

        height_perm = torch.randperm(height)
        width_perm = torch.randperm(width)

        # Permutes rows and columns to scramble features in the space
        permuted_features = flat_features[:, :, height_perm].detach()
        permuted_features = permuted_features[:, :, :, width_perm].detach()
        # Mask of 1 where the original image should be kept, 0 where the permuted features should be kept
        features_mask = torch.ones_like(flat_features)

        # Computes the mask for each image
        all_cut_masks = []
        for current_feature_mask in features_mask:
            # Samples the number of holes and performs them
            holes_count = random.randrange(min_count, max_count)
            for hole_idx in range(holes_count):
                # Computes coodinates of the hole
                begin_row = random.randrange(0, height - size)
                begin_column = random.randrange(0, width - size)
                end_row = begin_row + size
                end_column = begin_column + size

                # Cuts a hole
                current_feature_mask[:, begin_row:end_row, begin_column:end_column] *= 0
            all_cut_masks.append(current_feature_mask)
        # Recreates a single tensor for the mask
        all_cut_masks = torch.stack(all_cut_masks, dim=0)

        # Makes the holes in the original image and fills them with content from other pixels
        final_features = flat_features * all_cut_masks + (1 - all_cut_masks) * permuted_features

        folded_final_features = TensorFolder.fold(final_features, initial_dimensions)
        return folded_final_features

    @staticmethod
    def transforms_from_config(transform_config: Dict):
        '''
        Computes transformations from a configuration parameter

        :param transform_config: Dictionary with parameters describing the transformations
        :return: Callable object representing the transformation
                 the object has attribute "transformation_name" with the name of the transformation
        '''

        transforms = []
        blur_probability = transform_config["gaussian_blur_probability"]
        kernel_size = transform_config["gaussian_blur_kernel"]
        sigma = transform_config["gaussian_blur_sigma"]
        noise_probability = transform_config["noise_probability"]
        intensity = transform_config["noise_intensity"]

        name = f"blur_kernel_size_{kernel_size}_blur_sigma_{sigma}_noise_intensity_{intensity}"

        # Checks whether cutout is to be used
        cutout_probability = 0.0
        cutout_size = 0
        cutout_min_count = 0
        cutout_max_count = 0
        if "cutout_probability" in transform_config:
            cutout_probability = transform_config["cutout_probability"]
            cutout_size = transform_config["cutout_size"]
            cutout_min_count = transform_config["cutout_min_count"]
            cutout_max_count = transform_config["cutout_max_count"]

            name = name + f"_cutout_size_{cutout_size}_cutout_count_[{cutout_min_count},{cutout_max_count}]"

        if kernel_size > 0:
            blur_transformation = partial(LatentTransformationsHelper.apply_blur, kernel_size=kernel_size, sigma=sigma)
            blur_transformation = RandomApply([blur_transformation], blur_probability)
            transforms.append(blur_transformation)
        if cutout_size > 0 and cutout_max_count > 0:
            cutout_transformation = partial(LatentTransformationsHelper.apply_cutout, size=cutout_size, min_count=cutout_min_count, max_count=cutout_max_count)
            cutout_transformation = RandomApply([cutout_transformation], cutout_probability)
            transforms.append(cutout_transformation)
        if intensity > 0.0:
            noise_transformation = partial(LatentTransformationsHelper.apply_gaussian_noise, intensity=intensity)
            noise_transformation = RandomApply([noise_transformation], noise_probability)
            transforms.append(noise_transformation)

        transformation = torchvision.transforms.Compose(transforms)
        # Creates an attribute with the name of the transformation
        transformation.transformation_name = name
        return transformation
