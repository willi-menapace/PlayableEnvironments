import os
from typing import Tuple, Dict, List

import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.utils import make_grid, save_image

from model.utils.object_ids_helper import ObjectIDsHelper
from utils.drawing.bounding_box_drawer import BoundingBoxDrawer
from utils.drawing.text_drawer import TextDrawer
from utils.logger import Logger
from utils.tensor_folder import TensorFolder


class ImageHelper:

    def __init__(self, config, logger: Logger, logger_prefix: str):
        '''
        Helper class for logging images

        :param config:
        :param logger:
        :param logger_prefix:
        '''

        self.config = config
        self.logger = logger
        self.logger_prefix = logger_prefix

        self.bounding_box_color = (255, 0, 0)
        self.ground_truth_bounding_box_color = (0, 0, 255)

        try:
            # Helper for handling the relationships between object ids and their models
            self.object_id_helper = ObjectIDsHelper(self.config)
        except Exception as exception:
            print("Warning: object_id_helper could not be instanced in ImageHelper. Some functionality will be compromised.")
            print(f"Cause: {exception}")

    @staticmethod
    def normalize(values: torch.Tensor, range: Tuple[float, float]) -> torch.Tensor:
        '''
        Normalizes values from the specified range to [0, 1]
        :param values: arbitrarily shaped tensor to normalize
        :param range: the values that will correspond to 0 and 1 respectively
        :return: the input tensor normalized in [0, 1]
        '''

        min = range[0]
        max = range[1]

        values = (values - min) / (max - min)

        return values

    @staticmethod
    def check_and_normalize_range(observations: torch.Tensor) -> torch.Tensor:
        '''
        If the range of the observations is in [-1, 1] instead of [0, 1] it normalizes it
        :param observations: arbitrarily shaped tensor to normalize
        :return: the input tensor normalized in [0, 1]
        '''

        minimum_value = torch.min(observations).item()

        # Check range and normalize
        if minimum_value < 0:
            observations = (observations + 1) / 2
        return observations

    def save_image_pairs(self, first_images: torch.Tensor, second_images: torch.Tensor, step, log_key="observations", max_batches=100, cameras_as_columns=True):
        '''
        Saves images showing organized in corresponding pairs. If necessary upsamples the second image tensor to the
        size of the first

        :param first_images: (bs, observations_count, cameras_count, 3, h, w) tensor with observed images
        :param second_images: (bs, observations_count, cameras_count, 3, h, w) tensor with reconstructed images
        :param log_key: key to use for logging
        :params cameras_as_columns: if true puts cameras as columns of the image matrix, otherwise puts frames of the same video
        :return:
        '''

        # Swaps observations_count with cameras_count dimensions so that observations_count will end up as columns
        # in the image matrix
        if not cameras_as_columns:
            first_images = first_images.transpose(1, 2)
            second_images = second_images.transpose(1, 2)

        # Flattens so that each observation is treated as a batch and cameras will be displayed in the columns
        first_images, _ = TensorFolder.flatten(first_images)
        second_images, _ = TensorFolder.flatten(second_images)

        # Resizes the second tensor to the size of the first if necessary
        if first_images.size(-2) != second_images.size(-2) or first_images.size(-1) != second_images.size(-1):
            target_height = first_images.size(-2)
            target_width = first_images.size(-1)

            # Performs the upsampling
            flat_second_images, second_images_dimensions = TensorFolder.flatten(second_images, -3)
            flat_second_images = F.upsample(flat_second_images, (target_height, target_width), mode="bilinear")
            second_images = TensorFolder.fold(flat_second_images, second_images_dimensions)

        if first_images.size(0) > max_batches:
            first_images = first_images[:max_batches]
            second_images = second_images[:max_batches]

        batch_size = first_images.size(0)
        cameras_count = first_images.size(1)
        reconstructed_observations_count = second_images.size(1)

        observations_list = []
        # Disposes the images in alternating rows with originals on the top and reconstructed at the bottom
        for batch_element_idx in range(batch_size):
            for observation_element_idx in range(cameras_count):
                observations_list.append(first_images[batch_element_idx, observation_element_idx])
            for observation_element_idx in range(cameras_count):
                observations_list.append(second_images[batch_element_idx, observation_element_idx])

        observations_grid = np.transpose(make_grid(observations_list, padding=2, pad_value=1, nrow=cameras_count).cpu().numpy(), (1, 2, 0))

        # Increases jpeg quality to the maximum quality still using compression (95). Disables chroma subsampling for sharper image.
        Image.fromarray((observations_grid * 255).astype(np.uint8)).save(os.path.join(self.config["logging"]["output_images_directory"], f"{step:09}_{log_key}.jpg"), quality=95, subsampling=0)

        wandb = self.logger.get_wandb()
        wandb.log({f"{self.logger_prefix}/{log_key}": wandb.Image(observations_grid), "step": step}, step=step)

    def fix_multiresolution(self, input) -> torch.Tensor:
        '''
        If the input represents values at multiple resolutions get the values corresponding to the highest available resolution
        :param input: input value, possibly representing results at multiple resolutions
        :return: tensor with the highest resolution values
        '''

        # If the input is not a tensor it must be the multiresolution values
        if not torch.is_tensor(input):
            return input[0]

        return input

    def save_images_from_results(self, render_results: Dict, step: int, prefix: str = "", ground_truth_observations=None, object_attention_maps=None, object_crops=None, bounding_boxes=None, reconstructed_bounding_boxes=None, reconstructed_3d_bounding_boxes=None, projected_axes=None, log_only_global: bool=True, text: List[List[str]]=None):
        '''
        Generates and saves images from the given results

        :param render_results: model results
        :param step: current step
        :param prefix: prefix to assign to each saved value
        :param text: list of dimension (batch_size, observations_count) of text to apply to each image
        :return:
        '''

        if prefix != "":
            prefix = prefix + "_"

        # Checks whether we have the data to save certain results
        save_attention = object_attention_maps is not None and object_crops is not None
        save_bounding_boxes = bounding_boxes is not None and reconstructed_bounding_boxes is not None
        save_axes = projected_axes is not None

        static_objects_count = self.object_id_helper.static_objects_count
        objects_count = self.object_id_helper.objects_count

        if save_attention:
            for object_idx in range(objects_count):
                current_attention_map = object_attention_maps[object_idx]
                current_crops = object_crops[object_idx]

                current_crops = ImageHelper.normalize(current_crops, [-1.0, 1.0])
                self.save_image_pairs(current_crops, current_attention_map, step, log_key=f"{prefix}attention_object_{object_idx}")

        for result_type in render_results.keys():
            # Filters away undesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            current_type_render_results = render_results[result_type]
            # Saves the rendering for each object
            for current_object_key in current_type_render_results.keys():
                # If requested, log only global objects
                if log_only_global and current_object_key != "global":
                    continue

                current_render_results = current_type_render_results[current_object_key]

                # Autoencoder rendering
                if "reconstructed_observations" in current_render_results:
                    reconstructed_observations = current_render_results["reconstructed_observations"]
                # Standard NeRF rendering
                else:
                    reconstructed_observations = current_render_results["integrated_features"]
                    reconstructed_observations = self.fix_multiresolution(reconstructed_observations)
                    reconstructed_observations = reconstructed_observations.permute([0, 1, 2, 5, 3, 4])

                # Checks whether the reconstructed observations are images or features
                reconstructed_observation_features_count = reconstructed_observations.size(-3)
                reconstructed_observations_are_images = (reconstructed_observation_features_count == 3)

                # Creates fake ground truth observations if not available
                if ground_truth_observations is None:
                    ground_truth_observations = torch.zeros_like(reconstructed_observations[..., :3, :, :])

                depth_maps = current_render_results["depth"]
                depth_maps = self.fix_multiresolution(depth_maps)
                depth_maps = torch.stack([depth_maps] * 3, dim=-1)  # Transforms it in a 3 channel tensor
                depth_maps = depth_maps.permute([0, 1, 2, 5, 3, 4])
                max_depth = torch.max(depth_maps).item()
                depth_maps = ImageHelper.normalize(depth_maps, (0, max_depth))

                opacity_maps = current_render_results["opacity"]
                opacity_maps = self.fix_multiresolution(opacity_maps)
                opacity_maps = torch.stack([opacity_maps] * 3, dim=-1)  # Transforms it in a 3 channel tensor
                opacity_maps = opacity_maps.permute([0, 1, 2, 5, 3, 4])

                displacements_maps = current_render_results["integrated_displacements_magnitude"]
                displacements_maps = self.fix_multiresolution(displacements_maps)
                displacements_maps = torch.stack([displacements_maps] * 3, dim=-1)  # Transforms it in a 3 channel tensor
                displacements_maps = displacements_maps.permute([0, 1, 2, 5, 3, 4])
                max_displacement = torch.max(displacements_maps).item()
                displacements_maps = ImageHelper.normalize(displacements_maps, (0, max_displacement))

                divergence_maps = current_render_results["integrated_divergence"]
                divergence_maps = self.fix_multiresolution(divergence_maps)
                divergence_maps = torch.stack([divergence_maps] * 3, dim=-1)  # Transforms it in a 3 channel tensor
                divergence_maps = divergence_maps.permute([0, 1, 2, 5, 3, 4])
                max_divergence = torch.max(divergence_maps).item()
                divergence_maps = ImageHelper.normalize(divergence_maps, (0, max_divergence))

                # Renders the reconstructed observations only if they are images
                if reconstructed_observations_are_images:
                    reference_observations = reconstructed_observations
                    # If text drawing is requested draw it on the images
                    if text is not None:
                        reference_observations = TextDrawer.draw_text_on_bidimensional_batch(reference_observations, text)
                    self.save_image_pairs(ground_truth_observations, reference_observations, step, log_key=f"{prefix}observations_{result_type}_{current_object_key}")
                # If reconstructed observations are features, use ground truth observations for image saving
                if not reconstructed_observations_are_images:
                    reference_observations = ground_truth_observations

                self.save_image_pairs(reference_observations, depth_maps, step, log_key=f"{prefix}depth_{result_type}_{current_object_key}")
                self.save_image_pairs(reference_observations, opacity_maps, step, log_key=f"{prefix}opacity_{result_type}_{current_object_key}")
                self.save_image_pairs(reference_observations, displacements_maps, step, log_key=f"{prefix}displacements_{result_type}_{current_object_key}")
                self.save_image_pairs(reference_observations, divergence_maps, step, log_key=f"{prefix}divergence_{result_type}_{current_object_key}")

                if save_bounding_boxes and current_object_key == "global":
                    # Computation for each object
                    for object_idx in range(objects_count):
                        current_bounding_boxes = reconstructed_bounding_boxes[..., object_idx]

                        # Draws bounding boxes on ground truth observations
                        ground_truth_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(ground_truth_observations, current_bounding_boxes, self.bounding_box_color)

                        # Draws bounding boxes on reconstructed observations
                        reference_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(reference_observations, current_bounding_boxes, self.bounding_box_color)

                        # If the object is dynamic, also draw ground truth
                        if object_idx >= static_objects_count:
                            dynamic_object_idx = self.object_id_helper.dynamic_object_idx_by_object_idx(object_idx)
                            current_ground_truth_bounding_boxes = bounding_boxes[..., dynamic_object_idx]
                            ground_truth_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(ground_truth_observations_with_bbox, current_ground_truth_bounding_boxes, self.ground_truth_bounding_box_color)
                            reference_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(reference_observations_with_bbox, current_ground_truth_bounding_boxes, self.ground_truth_bounding_box_color)

                        # Draws the 3d bounding boxes if available
                        if reconstructed_3d_bounding_boxes is not None:
                            current_3d_bounding_bbox = reconstructed_3d_bounding_boxes[..., object_idx]
                            ground_truth_observations_with_bbox = BoundingBoxDrawer.draw_3d_bounding_boxes(ground_truth_observations_with_bbox, current_3d_bounding_bbox, self.bounding_box_color)
                            reference_observations_with_bbox = BoundingBoxDrawer.draw_3d_bounding_boxes(reference_observations_with_bbox, current_3d_bounding_bbox, self.bounding_box_color)

                        self.save_image_pairs(ground_truth_observations_with_bbox, reference_observations_with_bbox, step, log_key=f"{prefix}observations_{result_type}_{current_object_key}_with_bbox_{object_idx}")

                if save_axes and current_object_key == "global":
                    # Computation for each object
                    for object_idx in range(objects_count):

                        current_projected_axes = projected_axes[..., object_idx]
                        ground_truth_observations_with_axes = BoundingBoxDrawer.draw_axes(ground_truth_observations, current_projected_axes)
                        reference_observations_with_axes = BoundingBoxDrawer.draw_axes(reference_observations, current_projected_axes)

                        self.save_image_pairs(ground_truth_observations_with_axes, reference_observations_with_axes, step, log_key=f"{prefix}observations_{result_type}_{current_object_key}_with_axes_{object_idx}")


    def save_decoded_images_from_results(self, render_results: Dict, step: int, prefix: str = "", ground_truth_observations=None, bounding_boxes=None, reconstructed_bounding_boxes=None):
        '''
        Generates and saves images from the given results

        :param render_results: model results
        :param step: current step
        :param prefix: prefix to assign to each saved value
        :return:
        '''

        if prefix != "":
            prefix = prefix + "_"

        # Checks whether we have the data to save certain results
        save_bounding_boxes = bounding_boxes is not None and reconstructed_bounding_boxes is not None

        static_objects_count = self.object_id_helper.static_objects_count
        objects_count = self.object_id_helper.objects_count

        for result_type in render_results.keys():

            # Filters away undesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            current_type_render_results = render_results[result_type]
            # Saves the rendering for each object
            for current_object_key in current_type_render_results.keys():
                # Log only global objects
                if current_object_key != "global":
                    continue

                current_render_results = current_type_render_results[current_object_key]

                reconstructed_observations = current_render_results["decoded_images"]

                # Creates fake ground truth observations if not available
                if ground_truth_observations is None:
                    ground_truth_observations = torch.zeros_like(reconstructed_observations[..., :3, :, :])

                # Renders the reconstructed observations only if they are images
                self.save_image_pairs(ground_truth_observations, reconstructed_observations, step, log_key=f"{prefix}observations_{result_type}_{current_object_key}")

                if save_bounding_boxes and current_object_key == "global":
                    # Computation for each object
                    for object_idx in range(static_objects_count, objects_count):
                        dynamic_object_idx = self.object_id_helper.dynamic_object_idx_by_object_idx(object_idx)
                        current_bounding_boxes = reconstructed_bounding_boxes[..., object_idx]
                        current_ground_truth_bounding_boxes = bounding_boxes[..., dynamic_object_idx]

                        # Draws bounding boxes on ground truth observations
                        ground_truth_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(ground_truth_observations, current_bounding_boxes, self.bounding_box_color)
                        ground_truth_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(ground_truth_observations_with_bbox, current_ground_truth_bounding_boxes, self.ground_truth_bounding_box_color)

                        # Draws bounding boxes on reconstructed observations
                        reconstructed_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(reconstructed_observations, current_bounding_boxes, self.bounding_box_color)
                        reconstructed_observations_with_bbox = BoundingBoxDrawer.draw_bounding_boxes(reconstructed_observations_with_bbox, current_ground_truth_bounding_boxes, self.ground_truth_bounding_box_color)

                        self.save_image_pairs(ground_truth_observations_with_bbox, reconstructed_observations_with_bbox, step, log_key=f"{prefix}decoded_observations_{result_type}_{current_object_key}_with_bbox_{object_idx}")

    def save_image_pairs(self, first_images: torch.Tensor, second_images: torch.Tensor, step, log_key="observations", max_batches=100, cameras_as_columns=True):
        '''
        Saves images showing organized in corresponding pairs. If necessary upsamples the second image tensor to the
        size of the first

        :param first_images: (bs, observations_count, cameras_count, 3, h, w) tensor with observed images
        :param second_images: (bs, observations_count, cameras_count, 3, h, w) tensor with reconstructed images
        :param log_key: key to use for logging
        :params cameras_as_columns: if true puts cameras as columns of the image matrix, otherwise puts frames of the same video
        :return:
        '''

        # Swaps observations_count with cameras_count dimensions so that observations_count will end up as columns
        # in the image matrix
        if not cameras_as_columns:
            first_images = first_images.transpose(1, 2)
            second_images = second_images.transpose(1, 2)

        # Flattens so that each observation is treated as a batch and cameras will be displayed in the columns
        first_images, _ = TensorFolder.flatten(first_images)
        second_images, _ = TensorFolder.flatten(second_images)

        # Resizes the second tensor to the size of the first if necessary
        if first_images.size(-2) != second_images.size(-2) or first_images.size(-1) != second_images.size(-1):
            target_height = first_images.size(-2)
            target_width = first_images.size(-1)

            # Performs the upsampling
            flat_second_images, second_images_dimensions = TensorFolder.flatten(second_images, -3)
            flat_second_images = F.upsample(flat_second_images, (target_height, target_width), mode="bilinear")
            second_images = TensorFolder.fold(flat_second_images, second_images_dimensions)

        if first_images.size(0) > max_batches:
            first_images = first_images[:max_batches]
            second_images = second_images[:max_batches]

        batch_size = first_images.size(0)
        cameras_count = first_images.size(1)
        reconstructed_observations_count = second_images.size(1)

        observations_list = []
        # Disposes the images in alternating rows with originals on the top and reconstructed at the bottom
        for batch_element_idx in range(batch_size):
            for observation_element_idx in range(cameras_count):
                observations_list.append(first_images[batch_element_idx, observation_element_idx])
            for observation_element_idx in range(cameras_count):
                observations_list.append(second_images[batch_element_idx, observation_element_idx])

        observations_grid = np.transpose(make_grid(observations_list, padding=2, pad_value=1, nrow=cameras_count).cpu().numpy(), (1, 2, 0))

        # Increases jpeg quality to the maximum quality still using compression (95). Disables chroma subsampling for sharper image.
        Image.fromarray((observations_grid * 255).astype(np.uint8)).save(os.path.join(self.config["logging"]["output_images_directory"], f"{step:09}_{log_key}.jpg"), quality=95, subsampling=0)

        wandb = self.logger.get_wandb()
        wandb.log({f"{self.logger_prefix}/{log_key}": wandb.Image(observations_grid), "step": step}, step=step)

    def fix_multiresolution(self, input) -> torch.Tensor:
        '''
        If the input represents values at multiple resolutions get the values corresponding to the highest available resolution
        :param input: input value, possibly representing results at multiple resolutions
        :return: tensor with the highest resolution values
        '''

        # If the input is not a tensor it must be the multiresolution values
        if not torch.is_tensor(input):
            return input[0]

        return input

    def save_reconstructed_images_to_paths(self, render_results: Dict, output_paths: np.ndarray, crop_regions: np.ndarray=None, target_output_size: Tuple[int]=None):
        '''
        Saves the reconstructed images from the given results

        :param render_results: model results
        :param output_paths: (batch_size, observations_count, cameras_count) string array with output path where to save each image
        :param crop_regions: (batch_size, observations_count, cameras_count, 4) array of (left, top, right, bottom) coordinates in [0, 1] where to crop each image
                            If None performs no cropping
        :param target_output_size (width, height) target output size. Used only if crop is specified
        :return:
        '''

        for result_type in render_results.keys():
            # Filters away undesired keys
            if result_type not in ["coarse", "fine"]:
                continue

            current_type_render_results = render_results[result_type]
            # Saves the rendering for each object
            for current_object_key in current_type_render_results.keys():
                # Render only the global results
                if current_object_key != "global":
                    continue

                current_render_results = current_type_render_results[current_object_key]

                # Autoencoder rendering
                if "reconstructed_observations" in current_render_results:
                    reconstructed_observations = current_render_results["reconstructed_observations"]
                # Standard NeRF rendering
                else:
                    reconstructed_observations = current_render_results["integrated_features"]
                    reconstructed_observations = self.fix_multiresolution(reconstructed_observations)
                    reconstructed_observations = reconstructed_observations.permute([0, 1, 2, 5, 3, 4])

                height = reconstructed_observations.size(-2)
                width = reconstructed_observations.size(-1)

                # Checks whether the reconstructed observations are images or features
                reconstructed_observation_features_count = reconstructed_observations.size(-3)
                if reconstructed_observation_features_count != 3:
                    raise Exception(f"The reconstructed observations are expected to be RGB images, but {reconstructed_observation_features_count} channels were found")

                flattened_reconstructed_observations, _ = TensorFolder.flatten(reconstructed_observations, dimensions=-3)
                flattened_output_paths = output_paths.reshape((-1,))
                # Prepares the crop regions converting them to pixel image coordinates
                if crop_regions is not None:
                    crop_regions = crop_regions.copy()
                    flattened_crop_regions = crop_regions.reshape((-1, 4))
                    flattened_crop_regions[:, 0] *= width
                    flattened_crop_regions[:, 1] *= height
                    flattened_crop_regions[:, 2] *= width
                    flattened_crop_regions[:, 3] *= height
                    flattened_crop_regions = np.rint(flattened_crop_regions).astype(int)

                elements_count = flattened_reconstructed_observations.size(0)
                paths_count = flattened_output_paths.shape[0]
                if elements_count != paths_count:
                    raise Exception(f"The number of images ({elements_count}) is different from the number of paths ({paths_count})")

                # Saves each image
                for element_idx in range(elements_count):

                    current_reconstructed_observation = flattened_reconstructed_observations[element_idx]
                    current_output_path = flattened_output_paths[element_idx]

                    # Crop the image if crops are requested
                    if crop_regions is not None:
                        if target_output_size is None:
                            target_output_size = (width, height)

                        current_crop_region = flattened_crop_regions[element_idx].tolist()
                        pil_image = torchvision.transforms.ToPILImage()(current_reconstructed_observation)
                        pil_image = pil_image.crop(current_crop_region)
                        pil_image = pil_image.resize(target_output_size, PIL.Image.LANCZOS)
                        pil_image.save(current_output_path)
                    # No crop, save images normally
                    else:
                        if target_output_size is not None:
                            raise Exception("Target output size can be used only if crop is specified")
                        save_image(current_reconstructed_observation, current_output_path)



