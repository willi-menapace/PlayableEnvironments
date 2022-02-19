from typing import Dict

import torch

from model.environment_model import EnvironmentModel
from model.environment_model_multiresolution_backpropagated_autoencoder import \
    EnvironmentModelMultiresolutionBackpropagatedAutoencoder
from utils.lib_3d.ray_helper import RayHelper


class EnvironmentModelMultiresolutionBackpropagatedDecoder(EnvironmentModelMultiresolutionBackpropagatedAutoencoder):

    def __init__(self, config):
        '''
        Initializes the environment model

        :param config: the configuration file
        '''
        super(EnvironmentModelMultiresolutionBackpropagatedDecoder, self).__init__(config)

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

        :return: Dictionary with the fields returned by the original method. If a batch of rays instead of the full image is being rendered, the following fields are also present:

                "splitted_positions" [(..., observations_count, cameras_count, samples_per_image_i, 2)] list of tensors with positions of the samples corresponding to each downsampling factor.
                                                                                                        Positions are normalized in [0, 1] and are expressed in (height, width) order.

                 "fine -> global" and "coarse -> global" contain
                    "reconstructed_observations" (..., observations_count, cameras_count, 3, patch_height, patch_width) tensor with reconstructed observations patch. The size of the patch is given by the size of the sampled patch of features
                                                                                                                        multiplied by the upscaling factor of the decoder.
                    "splitted_integrated_features" [(..., observations_count, cameras_count, samples_per_image_i, features_count_i)] list of tensors with NeRF features that correspond to the sampled positions at a given downsampling level
                                                                                                                                     One tensor is present for each corresponding resolution
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

        # Positions that were sampled to produce the features that are present in the results
        sampled_positions = results["positions"]
        # Splits the sampled positions into the samples that correspond to the patch at each resolution
        splitted_sampled_positions = RayHelper.split_strided_patch_ray_samples(sampled_positions, patch_size, patch_stride)
        results["splitted_positions"] = splitted_sampled_positions

        # Iterates over fine and coarse results
        for current_key in ["fine", "coarse"]:
            if current_key not in results:
                continue

            current_results = results[current_key]["global"]
            # (..., observations_count, cameras_count, samples_per_image, features_count)
            nerf_reconstructed_encoded_observations = current_results["integrated_features"]
            # Splits the features. NeRF produced features have channels in the last position, so use "hwc" ordering
            # NOTE: The features however need further splitting since they refer to sample positions at multiple resolutions
            nerf_reconstructed_encoded_observations = self.split_features_by_layer(nerf_reconstructed_encoded_observations, channel_order="hwc")

            # Inserts NeRF features into the encoder features for each downsampling level
            all_nerf_patches = []
            current_results["splitted_integrated_features"] = []
            for downsample_factor_idx, (current_nerf_encoded_observations, current_sampled_positions) in enumerate(zip(nerf_reconstructed_encoded_observations, splitted_sampled_positions)):
                # NeRF features refer to the correct downsampling level, but comprise samples from different resolutions
                # Gets the portion of samples that correspond to the current downsample factor, discarding the others
                current_nerf_encoded_observations = RayHelper.split_strided_patch_ray_samples(current_nerf_encoded_observations, patch_size, patch_stride)[downsample_factor_idx]
                current_results["splitted_integrated_features"].append(current_nerf_encoded_observations)  # Saves splitted NeRF features with direct correspondence to "sampled_encoded_observations"
                # Transforms the current nerf observations samples_per_image dimension into a (patch_size, patch_size) dimension
                current_nerf_encoded_observations_patch = RayHelper.strided_patch_ray_samples_to_patch(current_nerf_encoded_observations)

                all_nerf_patches.append(current_nerf_encoded_observations_patch)

            reconstructed_observations = self.autoencoder_model.forward_decoder(all_nerf_patches)

            current_results["reconstructed_observations"] = reconstructed_observations

        return results


def model(config):
    return EnvironmentModelMultiresolutionBackpropagatedDecoder(config)

