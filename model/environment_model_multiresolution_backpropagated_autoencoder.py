import os
from pathlib import Path
from typing import Dict

import torch

from model.environment_model import EnvironmentModel
from model.environment_model_backpropagated_autoencoder import EnvironmentModelBackpropagatedAutoencoder
from utils.drawing.autoencoder_features_drawer import AutoencoderFeaturesDrawer
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_folder import TensorFolder
from utils.tensor_splitter import TensorSplitter


class EnvironmentModelMultiresolutionBackpropagatedAutoencoder(EnvironmentModelBackpropagatedAutoencoder):

    def __init__(self, config):
        '''
        Initializes the environment model

        :param config: the configuration file
        '''
        super(EnvironmentModelMultiresolutionBackpropagatedAutoencoder, self).__init__(config)

        # Checks that not bottleneck transforms are specified
        if "bottleneck_transforms" in self.config["model"]:
            raise Exception("Bottleneck transforms are not supported by the MultiresolutionBackpropagatedAutoencdoer")

    def split_features_by_layer(self, features: torch.Tensor, channel_order="chw"):
        '''
        Splits the features into multiple tensors, each one corresponding to a specific downsampling layer
        :param features: (..., features_count, height, width) tensor to split for "chw". (..., features_count) for "hwc".
        :param channel_order: "chw" or "hwc" channel ordering. Note that despite the naming, "hwc" does not assume h and w are present.
        :return: [(..., features_count_i, height, width)] list of tensors with features corresponding to each downsampling layer for "chw"
        :return: [(..., features_count_i)] list of tensors with features corresponding to each downsampling layer for "hwc"
        '''

        # Gets the number of features that correspond to each output resolutions
        features_count_by_layer = self.autoencoder_model.get_features_count_by_layer()

        split_features = []
        current_features_begin_index = 0
        for current_features_count in features_count_by_layer:

            # Extracts only the features that correspond to the current downsample factor
            current_features_end_index = current_features_begin_index + current_features_count
            if channel_order == "chw":
                current_flat_reconstructed_encoded_observations = features[..., current_features_begin_index:current_features_end_index, :, :]
            elif channel_order == "hwc":
                current_flat_reconstructed_encoded_observations = features[..., current_features_begin_index:current_features_end_index]
            else:
                raise Exception(f"Invalid channel order '{channel_order}'")
            current_features_begin_index = current_features_end_index  # Updates the begin index of the features for the next downsample factor

            split_features.append(current_flat_reconstructed_encoded_observations)

        return split_features

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
            reconstructed_encoded_observations = current_results["integrated_features"]

            flat_reconstructed_encoded_observations, initial_dimensions = TensorFolder.flatten_list(reconstructed_encoded_observations, -3)

            # Splits the features according to the downsampling layer they correspond to.
            # Converts them to CHW format
            flat_splitted_reconstructed_encoded_observations = []
            for idx, current_reconstructed_encoded_observations in enumerate(flat_reconstructed_encoded_observations):
                # Puts in CHW order
                # (..., features_count, height, width)
                current_reconstructed_encoded_observations = current_reconstructed_encoded_observations.permute([0, 3, 1, 2])
                # Splits the features keeping only the ones corresponding to the current layer
                current_reconstructed_encoded_observations = self.split_features_by_layer(current_reconstructed_encoded_observations)[idx]
                flat_splitted_reconstructed_encoded_observations.append(current_reconstructed_encoded_observations)

                # Saves reconstructed autoencoder features if required
                if draw_features:
                    autoencoder_reconstructed_features_output_path = os.path.join(self.config["logging"]["output_images_directory"], f"autoencoder_reconstructed_features_layer_{idx}_{self.current_step:05d}")
                    Path(autoencoder_reconstructed_features_output_path).mkdir(parents=True, exist_ok=True)
                    AutoencoderFeaturesDrawer.draw_features(current_reconstructed_encoded_observations[0], autoencoder_reconstructed_features_output_path)

            # Forwards through the decoder
            flat_reconstructed_observations = self.autoencoder_model.forward_decoder(flat_splitted_reconstructed_encoded_observations)

            reconstructed_observations = TensorFolder.fold(flat_reconstructed_observations, initial_dimensions)
            current_results["reconstructed_observations"] = reconstructed_observations

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

        :return: Dictionary with the fields returned by the original method. If a batch of rays instead of the full image is being renderes, the following fields are also present:

                "splitted_positions" [(..., observations_count, cameras_count, samples_per_image_i, 2)] list of tensors with positions of the samples corresponding to each downsampling factor.
                                                                                                        Positions are normalized in [0, 1] and are expressed in (height, width) order.

                 "fine -> global" and "coarse -> global" contain
                    "reconstructed_observations" (..., observations_count, cameras_count, 3, height, width) tensor with reconstructed observations
                    "splitted_integrated_features" [(..., observations_count, cameras_count, samples_per_image_i, features_count_i)] list of tensors with NeRF features that correspond to the sampled positions at a given downsampling level
                                                                                                                                     One tensor is present for each corresponding resolution
                 Additional fields are present:
                     "encoded_observations": [(..., observations_count, cameras_count, features_count_i, bottleneck_height_i, bottleneck_width_i)] list of tensors with encoded observations. One tensor is present for each corresponding resolution
                     "sampled_encoded_observations": [(..., observations_count, cameras_count, samples_per_image_i, features_count_i)] list of tensors with the encoded observations that correspond to the sampled positions.
                                                                                                                                       One tensor is present for each corresponding resolution
                 Additional fields are present if the autoencoder is variational:
                     "encoded_observations_log_var": [(..., observations_count, cameras_count, features_count_i, bottleneck_height_i, bottleneck_width_i)] list of tensors with encoded observations log variance. One tensor is present for each corresponding resolution
        '''

        original_image_height = observations.size(-2)
        original_image_width = observations.size(-1)

        # Calls the original method
        results = EnvironmentModel.forward_from_observations(self, observations, camera_rotations, camera_translations,
                                                                                     focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes,
                                                                                     video_frame_indexes, video_indexes, samples_per_image,
                                                                                     perturb, samples_per_image_batching, shuffle_style, upsample_factor,
                                                                                     patch_size, patch_stride, align_grid, canonical_pose)

        # If the whole image is being rendered, then raw results are returned
        if samples_per_image == 0:
            return results

        encoded_observations = self.autoencoder_model.forward_encoder(observations)
        encoded_observations_to_sample = encoded_observations  # The encoded observations that are to be sampled. Used eg. for feature matching loss

        # if the model is variational perform sampling
        if "variational" in self.autoencoder_model.model_config and self.autoencoder_model.model_config["variational"]:

            splitted_encoded_observations = TensorSplitter.split(encoded_observations, dim=-3, factor=2)
            encoded_observations = self.autoencoder_model.sample_list(splitted_encoded_observations)

            encoded_observations_mean = [current_encoded_observation_mean for current_encoded_observation_mean, _ in splitted_encoded_observations]
            encoded_observations_log_var = [current_encoded_observation_log_var for _, current_encoded_observation_log_var in splitted_encoded_observations]
            encoded_observations_to_sample = encoded_observations_mean

            # Inserts in the results
            results["encoded_observations"] = encoded_observations_mean
            results["encoded_observations_log_var"] = encoded_observations_log_var
        else:
            # If the autoencoder is not variational just insert the output of the encoder
            results["encoded_observations"] = encoded_observations

        # Positions that were sampled to produce the features that are present in the results
        sampled_positions = results["positions"]
        # Splits the sampled positions into the samples that correspond to the patch at each resolution
        splitted_sampled_positions = RayHelper.split_strided_patch_ray_samples(sampled_positions, patch_size, patch_stride)
        results["splitted_positions"] = splitted_sampled_positions

        all_samples_encoded_observations = []
        # Foreach downsampling level, extracts the encoded observations that correspond to the given positions
        for current_encoded_observations_to_sample, current_sampled_positions in zip(encoded_observations_to_sample, splitted_sampled_positions):
            # TODO here mode="nearest" should be used when patches of features aligned with the centers of pixel blocks are used
            current_sampled_encoded_observations = RayHelper.sample_features_at(current_encoded_observations_to_sample, current_sampled_positions, mode="bilinear", original_image_size=(original_image_height, original_image_width))
            all_samples_encoded_observations.append(current_sampled_encoded_observations)
        results["sampled_encoded_observations"] = all_samples_encoded_observations

        # Iterates over fine and coarse results
        for current_key in ["fine", "coarse"]:
            if current_key not in results:
                continue

            current_results = results[current_key]["global"]
            # (..., observations_count, cameras_count, height, width, features_count)
            nerf_reconstructed_encoded_observations = current_results["integrated_features"]
            # Splits the features. NeRF produced features have channels in the last position, so use "hwc" ordering
            # NOTE: The features however need further splitting since they refer to sample positions at multiple resolutions
            nerf_reconstructed_encoded_observations = self.split_features_by_layer(nerf_reconstructed_encoded_observations, channel_order="hwc")

            # Inserts NeRF features into the encoder features for each downsampling level
            all_merged_features = []
            current_results["splitted_integrated_features"] = []
            for downsample_factor_idx, (current_encoded_observations, current_nerf_observations, current_sampled_positions) in enumerate(zip(encoded_observations, nerf_reconstructed_encoded_observations, splitted_sampled_positions)):
                # NeRF features refer to the correct downsampling level, but comprise samples from different resolutions
                # Gets the portion of samples that correspond to the current downsample factor, discarding the others
                current_nerf_observations = RayHelper.split_strided_patch_ray_samples(current_nerf_observations, patch_size, patch_stride)[downsample_factor_idx]
                current_results["splitted_integrated_features"].append(current_nerf_observations)  # Saves splitted NeRF features with direct correspondence to "sampled_encoded_observations"

                merged_features = EnvironmentModelMultiresolutionBackpropagatedAutoencoder.insert_samples_into_features(current_encoded_observations, current_nerf_observations, current_sampled_positions, (original_image_height, original_image_width))
                all_merged_features.append(merged_features)

            reconstructed_observations = self.autoencoder_model.forward_decoder(all_merged_features)

            current_results["reconstructed_observations"] = reconstructed_observations

        return results


def model(config):
    return EnvironmentModelMultiresolutionBackpropagatedAutoencoder(config)

