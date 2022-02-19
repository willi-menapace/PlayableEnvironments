import collections
import importlib
import os
from pathlib import Path
from typing import Tuple, Dict, List, Union

import torch
import torch.nn as nn

from model.autoencoder_models.layers.latent_transformations_helper import LatentTransformationsHelper
from model.environment_model import EnvironmentModel
from utils.drawing.autoencoder_features_drawer import AutoencoderFeaturesDrawer
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_folder import TensorFolder


class EnvironmentModelBackpropagatedAutoencoder(EnvironmentModel):

    def __init__(self, config):
        '''
        Initializes the environment model

        :param config: the configuration file
        '''
        super(EnvironmentModelBackpropagatedAutoencoder, self).__init__(config)

        autoencoder_config = self.config["model"]["autoencoder"]
        autoencoder_architecture = autoencoder_config["architecture"]

        # Istantiates the model
        self.autoencoder_model = getattr(importlib.import_module(autoencoder_architecture), 'model')(autoencoder_config)
        self.autoencoder_model.eval()

        # Loads weights for the autoencoder
        model_weights_filename = autoencoder_config["weights_filename"]
        if model_weights_filename != "untrained_model":
            if not os.path.isfile(model_weights_filename):
                raise Exception(f"Cannot load autoencoder model: no checkpoint found at '{model_weights_filename}'")
            loaded_state = torch.load(model_weights_filename)
            self.autoencoder_model.load_state_dict(loaded_state["model"])

        # Defines the function to apply to autoencoder features. Identity function by default
        self.autoencoder_bottleneck_transform = lambda x: x
        if "bottleneck_transforms" in self.config["model"]:
            self.autoencoder_bottleneck_transform = LatentTransformationsHelper.transforms_from_config(self.config["model"]["bottleneck_transforms"])

        # Whether the autoencoder is frozen or not.
        # At the beginning the autoencoder is not frozen
        self.is_autoencoder_frozen = False

        # Computes how much the resolution of the autoencoder is reduced at each of the downsampling layers
        self.strides = self.config["model"]["autoencoder"]["downsample_factor"]

    def set_autoencoder_frozen(self, frozen: bool):
        '''
        Freezes and unfreezes the autoencoder
        :param frozen: True to freeze the autoencoder, False to unfreeze it
        :return:
        '''

        # Operate only if the state needs to be changed
        if frozen != self.is_autoencoder_frozen:

            # If batch norm freeze is required, put batch norm to eval
            if self.config["model"]["autoencoder"]["also_freeze_bn"]:
                for m in self.autoencoder_model.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                        if frozen:
                            m.eval()
                        else:
                            m.train()

            requires_gradient = not frozen
            self.autoencoder_model.requires_grad_(requires_grad=requires_gradient)

            # Records the new state of the autoencoder
            self.is_autoencoder_frozen = frozen

    def get_autoencoder_parameters(self):
        return self.autoencoder_model.parameters()

    def get_main_parameters(self, additional_excluded_parameters=None):
        '''
        Gets all parameters that are not part of the base model main parameters and that are not part of the autoencoder
        :param additional_excluded_parameters: set of additional parameter names to exclude. For override purposes
        :return:
        '''

        additional_excluded_parameters = set(["autoencoder_model." + name for name, param in self.autoencoder_model.named_parameters()])

        return super(EnvironmentModelBackpropagatedAutoencoder, self).get_main_parameters(additional_excluded_parameters)

    def run_decoder_on_results(self, results: Dict, draw_features=False):
        '''
        Dictionary with results from a forward pass where all the spatial locations are predicted
        "fine -> global" and "coarse -> global" after the method call will contain
                 "reconstructed_observations" (..., observations_count, cameras_count, 3, height, width) tensor with reconstructed observations

        '''

        # Iterates over fine and coarse results
        for current_key in ["fine", "coarse"]:
            if current_key not in results:
                continue

            current_results = results[current_key]["global"]
            # (..., observations_count, cameras_count, height, width, features_count)
            # Only one resolution is present, so take the only element from the list
            reconstructed_encoded_observations = current_results["integrated_features"][0]

            flat_reconstructed_encoded_observations, initial_dimensions = TensorFolder.flatten(reconstructed_encoded_observations, -3)

            # Puts in CHW order
            # (..., features_count, height, width)
            reconstructed_encoded_observations = flat_reconstructed_encoded_observations.permute([0, 3, 1, 2])

            # Saves reconstructed autoencoder features if required
            if draw_features:
                autoencoder_reconstructed_features_output_path = os.path.join(self.config["logging"]["output_images_directory"], f"autoencoder_reconstructed_features_layer_{self.current_step:05d}")
                Path(autoencoder_reconstructed_features_output_path).mkdir(parents=True, exist_ok=True)
                AutoencoderFeaturesDrawer.draw_features(reconstructed_encoded_observations[0], autoencoder_reconstructed_features_output_path)

            # Forwards through the decoder
            flat_reconstructed_observations = self.autoencoder_model.forward_decoder(reconstructed_encoded_observations)

            reconstructed_observations = TensorFolder.fold(flat_reconstructed_observations, initial_dimensions)
            current_results["reconstructed_observations"] = reconstructed_observations

    def fold_strided_tensors(self, dictionary: Dict, height: int, width: int, strides: Union[List[int], int]):
        '''
        Folds to a list of rectangular tensors all tensors in the dictionary with a dimension that corresponds to flattened samples taken with the given strides from a height by width image
        :param dictionary: The dictionary to fold
        :param height: target height
        :param width: target width
        :param strides: target strides
        :return: the original dictionary with its matching entries folded. Each tensor is folded to a list of tensors corresponding to the folded tensor at each stride
        '''

        # Makes stride a list
        if not isinstance(strides, collections.Sequence):
            strides = [strides]

        target_dimension_size = 0
        for current_stride in strides:
            target_dimension_size += height // current_stride * width // current_stride

        for key in dictionary:
            current_element = dictionary[key]
            # If element is dict perform recursive call
            if type(current_element) is dict:
                dictionary[key] = self.fold_strided_tensors(current_element, height, width, strides)
            # If the element is a tensor check if it needs flattening
            elif torch.is_tensor(current_element):

                current_tensor_sizes = list(current_element.size())
                match_found = False
                for dimension_idx, dimension_size in enumerate(current_tensor_sizes):
                    if dimension_size == target_dimension_size:
                        match_found = True
                        break
                # If an element has matching dimensions, fold the tensor
                if match_found:
                    reshaped_element = RayHelper.fold_strided_grid_samples(current_element, strides, (height, width), dim=dimension_idx)
                    dictionary[key] = reshaped_element
            else:
                pass  # Nothing to modify

        return dictionary

    def render_full_frame_from_observations(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                                            focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor,
                                            global_frame_indexes: torch.Tensor, video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor,
                                            perturb: bool, samples_per_image_batching: int = 1000, upsample_factor: float = 1.0, canonical_pose: bool=False) -> Dict[str, torch.Tensor]:
        '''
        Renders a full frames starting from the given observations and camera parameters

        :param observations: see original method
        :param camera_rotations: see original method
        :param camera_translations: see original method
        :param focals: see original method
        :param bounding_boxes: see original method
        :param bounding_boxes_validity: see original method
        :param global_frame_indexes: see original method
        :param video_frame_indexes: see original method
        :param video_indexes: see original method
        :param perturb: see original method
        :param samples_per_image_batching: see original method
        :param upsample_factor: see original method
        :param canonical_pose: see original method

        :return: see original method. Additionally, "fine -> global" and "coarse -> global" contain
                 "reconstructed_observations" (..., observations_count, cameras_count, 3, height, width) tensor with reconstructed observations
        '''

        results = EnvironmentModel.forward_from_observations(self, observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity,
                            global_frame_indexes, video_frame_indexes, video_indexes, 0, perturb, samples_per_image_batching, patch_size=0, patch_stride=self.strides,
                            upsample_factor=upsample_factor, canonical_pose=canonical_pose)

        # Takes flattened results and folds them to rectangular tensors
        height = observations.size(-2)
        width = observations.size(-1)
        results = self.fold_strided_tensors(results, height, width, self.strides)

        # Runs the decoder on the output to produce the decoded images from the rendered features
        self.run_decoder_on_results(results, draw_features=True)

        return results

    def render_full_frame_from_scene_encoding(self, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                                            focals: torch.Tensor, image_size: Tuple[int, int], object_rotation_parameters_o2w: torch.Tensor, object_translation_parameters_o2w: torch.Tensor,
                                            object_style: torch.Tensor, object_deformation: torch.Tensor, object_in_scene: torch.Tensor,
                                            perturb: bool, samples_per_image_batching: int = 1000, upsample_factor: float = 1.0, canonical_pose: bool=False) -> Dict[str, torch.Tensor]:
        '''
        Renders a full frames starting from the given observations and camera parameters

        :param camera_rotations: see original method
        :param camera_translations: see original method
        :param focals: see original method
        :param image_size: see original method
        :param object_rotation_parameters_o2w: see original method
        :param object_translation_parameters_o2w: see original method
        :param object_style: see original method
        :param object_deformation: see original method
        :param object_in_scene: see original method
        :param perturb: see original method
        :param samples_per_image_batching: msee original method
        :param upsample_factor: see original method
        :param canonical_pose: see original method

        :return: see original method. Additionally, "fine -> global" and "coarse -> global" contain
                 "reconstructed_observations" (..., observations_count, cameras_count, 3, height, width) tensor with reconstructed observations
        '''

        results = EnvironmentModel.forward_from_scene_encoding(self, camera_rotations, camera_translations, focals, image_size,
                                                              object_rotation_parameters_o2w, object_translation_parameters_o2w, object_style, object_deformation, object_in_scene, 0,
                                                              perturb, samples_per_image_batching, upsample_factor, patch_size=0, patch_stride=self.strides, canonical_pose=canonical_pose)

        # Takes flattened results and folds them to rectangular tensors
        height, width = image_size
        results = self.fold_strided_tensors(results, height, width, self.strides)

        # Runs the decoder on the output to produce the decoded images from the rendered features
        self.run_decoder_on_results(results)

        return results

    @staticmethod
    def insert_samples_into_features(features: torch.Tensor, samples: torch.Tensor, samples_positions: torch.Tensor, original_image_size: Tuple[int, int]):
        '''
        Inserts the samples into the given feature matrix
        Size of the original image size must be a multiple of the features

        :param features: (..., features_count, height, width) feature matrix into which to insert the samples
        :param samples: (..., samples_per_image, features_count) samples to insert into the feature matrix
        :param samples_positions: (..., samples_per_image, 2) sample positions where to insert the sampled features.
                                                              Normalized in [0, 1], last dimension represents (height, width)
        :param original_image_size: (height, width) tuple with the size of the original image
        :return: modified_features: (..., features_count, height, width) feature matrix into which the samples have been inserted at their specified position
        '''

        features_height = features.size(-2)
        features_width = features.size(-1)
        features_count = features.size(-3)

        original_image_height, original_image_width = original_image_size

        downsample_factor = original_image_height // features_height
        if original_image_height % features_height != 0 or features_width * downsample_factor != original_image_width:
            raise Exception("Inconsistent size of features and original images. Size of the original image size must be a multiple of the features.")

        flat_features, initial_dimensions = TensorFolder.flatten(features, -3)
        flat_samples, _ = TensorFolder.flatten(samples, -2)
        flat_samples_positions, _ = TensorFolder.flatten(samples_positions, -2)

        elements_count = flat_features.size(0)

        # Gets the samples positions in integer coordinates in the original image
        original_image_size = torch.tensor(original_image_size, dtype=flat_features.dtype, device=flat_features.device)
        flat_samples_positions = (flat_samples_positions * original_image_size).round().long()
        # Obtains the samples positions in integer coordinates in the feature space
        flat_samples_positions = torch.floor(flat_samples_positions / downsample_factor).long()

        # Transforms the coordinates from (height, width) to coordinates in the height * width dimension
        linearized_flat_samples_positions = flat_samples_positions[..., 0] * features_width + flat_samples_positions[..., 1]
        # (elements_count, samples_per_image, features_count)
        linearized_flat_samples_positions = linearized_flat_samples_positions.unsqueeze(-1).repeat(1, 1, features_count)

        # Exchanges the samples_per_image with the features_count dimension
        # (elements_count, features_count, samples_per_image)
        flat_samples = torch.transpose(flat_samples, -2, -1)
        linearized_flat_samples_positions = torch.transpose(linearized_flat_samples_positions, -2, -1)

        # Flattens the height and width dimension
        flat_features = flat_features.reshape((elements_count, features_count, features_height * features_width))
        # WARNING: Gradient generated by this operation is wrong when multiple indexes map different features to the
        # same positions. We assume however collisions to be rare
        modified_features = flat_features.scatter(dim=-1, index=linearized_flat_samples_positions, src=flat_samples)

        modified_features = modified_features.reshape((elements_count, features_count, features_height, features_width))
        modified_features = TensorFolder.fold(modified_features, initial_dimensions)

        return modified_features

    def forward_from_observations(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, samples_per_image: int, perturb: bool,
                samples_per_image_batching: int = 0, shuffle_style: bool = False, upsample_factor: float = 1.0,
                patch_size: int = 0, patch_stride: int = 0, align_grid: bool = False, canonical_pose: bool=False) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param observations: see original method
        :param camera_rotations: see original method
        :param camera_translations: see original method
        :param focals: see original method
        :param bounding_boxes: see original method
        :param bounding_boxes_validity: see original method
        :param global_frame_indexes: see original method
        :param video_frame_indexes: see original method
        :param video_indexes: see original method
        :param samples_per_image: see original method
        :param perturb: see original method
        :param samples_per_image_batching: see original method
        :param shuffle_style: see original method
        :param patch_size: see original method
        :param patch_stride: see original method
        :param align_grid: see original method
        :param canonical_pose: see original method

        :return: Dictionary with the fields returned by the original method

                 "fine -> global" and "coarse -> global" contain
                    "reconstructed_observations" (..., observations_count, cameras_count, 3, height, width) tensor with reconstructed observations

                 Additional fields are present:
                     "encoded_observations": (..., observations_count, cameras_count, features_count, bottleneck_height, bottleneck_width) tensor with encoded observations. Not passed through autoencoder bottleneck
                     "encoded_observations_bottleneck": (..., observations_count, cameras_count, features_count, bottleneck_height, bottleneck_width) tensor with encoded observations. Passed through autoencoder bottleneck
                     "sampled_encoded_observations": (..., observations_count, cameras_count, samples_per_image, features_count) tensor with the encoded observations that correspond to the sampled positions. Passed through autoencoder bottleneck
                 Additional fields are present if the autoencoder is variational:
                     "encoded_observations_log_var": (..., observations_count, cameras_count, features_count, bottleneck_height, bottleneck_width) tensor with encoded observations log variance
        '''

        original_image_height = observations.size(-2)
        original_image_width = observations.size(-1)

        # Calls the original method
        results = super(EnvironmentModelBackpropagatedAutoencoder, self).forward_from_observations(observations, camera_rotations, camera_translations,
                                                                                     focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes,
                                                                                     video_frame_indexes, video_indexes, samples_per_image,
                                                                                     perturb, samples_per_image_batching, shuffle_style, upsample_factor,
                                                                                     patch_size, patch_stride, align_grid, canonical_pose)

        encoded_observations = self.autoencoder_model.forward_encoder(observations)
        encoded_observations_to_sample = encoded_observations  # The encoded observations that are to be sampled. Used eg. for feature matching loss

        # if the model is variational perform sampling
        if "variational" in self.autoencoder_model.model_config and self.autoencoder_model.model_config["variational"]:
            encoded_observations_mean, encoded_observations_log_var = torch.split(encoded_observations, encoded_observations.size(-3) // 2, dim=-3)
            encoded_observations_to_sample = encoded_observations_mean  # Do not use sampled features, but rather the mean to have a more stable optimization objective

            encoded_observations = self.autoencoder_model.sample(encoded_observations_mean, encoded_observations_log_var)

            # Inserts in the results
            results["encoded_observations"] = encoded_observations_mean
            results["encoded_observations_log_var"] = encoded_observations_log_var
        else:
            # If the autoencoder is not variational just insert the output of the encoder
            results["encoded_observations"] = encoded_observations

        encoded_observations_bottleneck = self.autoencoder_bottleneck_transform(encoded_observations)
        results["encoded_observations_bottleneck"] = encoded_observations_bottleneck

        # Positions that were sampled to produce the features that are present in the results
        sampled_positions = results["positions"]
        # Features output by the autoencoder must be passed throught the bottleneck transforms before being used
        encoded_observations_to_sample_bottleneck = self.autoencoder_bottleneck_transform(encoded_observations_to_sample)
        # Extracts the encoded observations that correspond to the given positions
        # TODO here mode="nearest" should be used when patches of features aligned with the centers of pixel blocks are used
        sampled_encoded_observations = RayHelper.sample_features_at(encoded_observations_to_sample_bottleneck, sampled_positions, mode="bilinear", original_image_size=(original_image_height, original_image_width))
        results["sampled_encoded_observations"] = sampled_encoded_observations

        # Iterates over fine and coarse results
        for current_key in ["fine", "coarse"]:
            if current_key not in results:
                continue

            current_results = results[current_key]["global"]
            # (..., observations_count, cameras_count, height, width, features_count)
            nerf_reconstructed_encoded_observations = current_results["integrated_features"]

            # If requested, erase the encode features to exclude encoder contribution on the generated patch
            base_encoded_observations = encoded_observations_bottleneck
            if self.config["model"]["autoencoder"]["exclude_encoder"]:
                base_encoded_observations = base_encoded_observations * 0.0

            merged_features = EnvironmentModelBackpropagatedAutoencoder.insert_samples_into_features(base_encoded_observations, nerf_reconstructed_encoded_observations, sampled_positions, (original_image_height, original_image_width))

            reconstructed_observations = self.autoencoder_model.forward_decoder(merged_features)

            current_results["reconstructed_observations"] = reconstructed_observations

        return results


def model(config):
    return EnvironmentModelBackpropagatedAutoencoder(config)


if __name__ == "__main__":

    features_count = 2
    features_height = 3
    features_width = 4

    downsample_factor = 2
    original_image_height = features_height * downsample_factor
    original_image_width = features_width * downsample_factor
    original_image_size = (original_image_height, original_image_width)

    original_image_size_tensor = torch.tensor(original_image_size, device="cuda:0")

    features = torch.zeros((features_count, features_height, features_width), requires_grad=True, device="cuda:0")
    sample_positions = torch.tensor([
        [0, 0],
        [2, 5],
        [5, 7],
    ], device="cuda:0")
    sample_positions = sample_positions / original_image_size_tensor

    samples = torch.tensor([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ], requires_grad=True, device="cuda:0")

    merged_features = EnvironmentModelBackpropagatedAutoencoder.insert_samples_into_features(features, samples, sample_positions, original_image_size)

    merged_features_numpy = merged_features.detach().cpu().numpy()

    loss = merged_features.sum()
    loss.backward()

    print(merged_features)

