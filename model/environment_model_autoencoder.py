import importlib
import os
from typing import Tuple, Dict

import torch

from model.autoencoder_models.layers.latent_transformations_helper import LatentTransformationsHelper
from model.environment_model import EnvironmentModel
from utils.drawing.autoencoder_features_drawer import AutoencoderFeaturesDrawer
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_folder import TensorFolder


class EnvironmentModelAutoencoder(EnvironmentModel):

    def __init__(self, config):
        '''
        Initializes the environment model

        :param config: the configuration file
        '''
        super(EnvironmentModelAutoencoder, self).__init__(config)

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

        # Disables gradients for autoencoder parameters
        self.autoencoder_model.requires_grad_(requires_grad=False)

    def train(self, mode: bool = True):
        '''
        Overrides train not to affect the environment model which always remains in evaluation mode
        :param mode:
        :return:
        '''
        super(EnvironmentModelAutoencoder, self).train(mode)

        # The autoencoder model is always in evaluation mode
        self.autoencoder_model.eval()
        return self

    def get_main_parameters(self, additional_excluded_parameters=None):
        '''
        Gets all parameters that are not part of the encoders or of the camera offsets
        :param additional_excluded_parameters: set of additional parameter names to exclude. For override purposes
        :return:
        '''

        excluded_parameters_names = set(["autoencoder_model."+name for name, param in self.object_encoders.named_parameters()])

        if additional_excluded_parameters is not None:
            excluded_parameters_names.union(additional_excluded_parameters)

        # Computes the parameters excluding also the ones coming from the autoencoder
        selected_parameters = super(EnvironmentModelAutoencoder, self).get_main_parameters(additional_excluded_parameters=excluded_parameters_names)

        return selected_parameters

    def run_decoder_on_results(self, results: Dict):
        '''
        Dictionary with results from a forward pass where all the spatial locations are predicted
       "fine -> global" and "coarse -> global" after the method call will contain
                 "reconstructed_observations" (..., observations_count, cameras_count, 3, height, width) tensor with reconstructed observations

        '''

        # Computes how much the resolution of the autoencoder is reduced at the bottleneck
        downsample_factor = self.config["model"]["autoencoder"]["downsampling_layers_count"] ** 2

        # Iterates over fine and coarse results
        for current_key in ["fine", "coarse"]:
            if current_key not in results:
                continue

            current_results = results[current_key]["global"]
            # (..., observations_count, cameras_count, height, width, features_count)
            reconstructed_encoded_observations = current_results["integrated_features"]
            flat_reconstructed_encoded_observations, initial_dimensions = TensorFolder.flatten(reconstructed_encoded_observations, -3)
            # Puts in CHW order
            # (..., features_count, height, width)
            flat_reconstructed_encoded_observations = flat_reconstructed_encoded_observations.permute([0, 3, 1, 2])

            AutoencoderFeaturesDrawer.draw_features(flat_reconstructed_encoded_observations[0], "results/autoencoder_reconstructed_features")

            # Downsamples the features to make them match the bottleneck size
            # (..., features_count, bottleneck_height, bottleneck_width)
            image_height = flat_reconstructed_encoded_observations.size(-2)
            image_width = flat_reconstructed_encoded_observations.size(-1)
            bottleneck_height = image_height // downsample_factor
            bottleneck_width = image_width // downsample_factor
            center_pixel_offset = downsample_factor // 2  # The position in the (downsample_factor, downsample_factor) pixel grid
                                                          # of the pixel corresponding to the center
            # Computes the indices of rows and columns that must be sampled to get center pixel
            row_indices = [idx * downsample_factor + center_pixel_offset for idx in range(bottleneck_height)]
            column_indices = [idx * downsample_factor + center_pixel_offset for idx in range(bottleneck_width)]
            # Samples rows and columns
            flat_reconstructed_encoded_observations = flat_reconstructed_encoded_observations[..., row_indices, :]
            flat_reconstructed_encoded_observations = flat_reconstructed_encoded_observations[..., column_indices]

            flat_reconstructed_observations = self.autoencoder_model.forward_decoder(flat_reconstructed_encoded_observations)

            reconstructed_observations = TensorFolder.fold(flat_reconstructed_observations, initial_dimensions)
            current_results["reconstructed_observations"] = reconstructed_observations

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

        # Gradients are not needed when computing encoded observations
        with torch.no_grad():
            encoded_observations = self.autoencoder_model.forward_encoder(observations)

        # if the model is variational just consider the mean and discard log variance
        if "variational" in self.autoencoder_model.model_config and self.autoencoder_model.model_config["variational"]:
            encoded_observations, _ = torch.split(encoded_observations, encoded_observations.size(-3) // 2, dim=-3)

        # Features output by the autoencoder must be passed throught the bottleneck transforms before being used
        encoded_observations = self.autoencoder_bottleneck_transform(encoded_observations)
        AutoencoderFeaturesDrawer.draw_features(encoded_observations[0, 0, 0], "results/autoencoder_features")

        results = super(EnvironmentModelAutoencoder, self).render_full_frame_from_observations(observations, camera_rotations, camera_translations,
                                                                                               focals, bounding_boxes, bounding_boxes_validity,
                                                                                               global_frame_indexes, video_frame_indexes, video_indexes,
                                                                                               perturb, samples_per_image_batching, upsample_factor, canonical_pose)

        # Runs the decoder on the output to produce the decoded images from the rendered features
        self.run_decoder_on_results(results)

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

        results = super(EnvironmentModelAutoencoder, self).render_full_frame_from_scene_encoding(camera_rotations, camera_translations, focals, image_size,
                                                                                                 object_rotation_parameters_o2w, object_translation_parameters_o2w,
                                                                                                 object_style, object_deformation, object_in_scene,
                                                                                                 perturb, samples_per_image_batching, upsample_factor, canonical_pose)

        # Runs the decoder on the output to produce the decoded images from the rendered features
        self.run_decoder_on_results(results)

        return results

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
                 Additional fields are present:
                 "sampled_encoded_observations": (..., observations_count, cameras_count, samples_per_image, features_count) tensor with encoded observation features corresponding to the sampled rays. Passed through the bottleneck

        '''

        # Gradients are not needed when computing encoded observations
        with torch.no_grad():
            encoded_observations = self.autoencoder_model.forward_encoder(observations)

        # if the model is variational just consider the mean and discard log variance
        if "variational" in self.autoencoder_model.model_config and self.autoencoder_model.model_config["variational"]:
            encoded_observations, _ = torch.split(encoded_observations, encoded_observations.size(-3) // 2, dim=-3)

        # Features output by the autoencoder must be passed throught the bottleneck transforms before being used
        encoded_observations = self.autoencoder_bottleneck_transform(encoded_observations)

        # Calls the original method
        results = super(EnvironmentModelAutoencoder, self).forward_from_observations(observations, camera_rotations, camera_translations,
                                                                                     focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes,
                                                                                     video_frame_indexes, video_indexes, samples_per_image,
                                                                                     perturb, samples_per_image_batching, shuffle_style, upsample_factor,
                                                                                     patch_size, patch_stride, align_grid, canonical_pose)

        # Positions that were sampled to produce the features that are present in the results
        sampled_positions = results["positions"]

        original_image_height = observations.size(-2)
        original_image_width = observations.size(-1)

        # TODO if samples_per_image = 0 then encoded observations can be flattened along height and width instead of sampled
        # Extracts the encoded observations that correspond to the given positions
        sampled_encoded_observations = RayHelper.sample_features_at(encoded_observations, sampled_positions, mode="bilinear", original_image_size=(original_image_height, original_image_width))
        results["sampled_encoded_observations"] = sampled_encoded_observations

        return results


def model(config):
    return EnvironmentModelAutoencoder(config)
