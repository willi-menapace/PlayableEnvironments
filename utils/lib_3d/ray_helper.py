import collections
import math
from typing import Tuple, List, Union

import torch
import torch.nn.functional as F
import numpy as np

from utils.lib_3d.pose_parameters import PoseParameters
from utils.tensor_folder import TensorFolder


class RayHelper:

    @staticmethod
    def create_camera_rays(initial_dimensions: List[int], height: int, width: int, focal: Union[float, torch.Tensor]) -> torch.Tensor:
        '''
        Creates the camera rays corresponding to each pixel

        :param initial_dimensions: list of initial dimensions to insert in the created tensors
        :param height: height of the ray grid to generate
        :param width: width of the ray grid to generate
        :param focal: single scalar of (initial_dimensions) tensor focal length
        :return: ray_directions (*initial_dimensions, height, width, 3) tensor with ray directions
                 ray_origins (*initial_dimensions, 3) tensor with ray origins
                 focal_normals (*initial_dimensions, 3) tensor with focal normals
        '''

        if not torch.is_tensor(focal):
            focal = torch.as_tensor([focal]).repeat(initial_dimensions).cuda()

        # Adds dimensions for height and width
        focal = focal.unsqueeze(-1).unsqueeze(-1)

        mesh_rows, mesh_columns = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        mesh_rows = mesh_rows.cuda()
        mesh_columns = mesh_columns.cuda()

        ray_directions_x = (mesh_columns - width / 2) / focal
        ray_directions_y = -(mesh_rows - height / 2) / focal  # y inverted because matrix rows grow going down but the
                                                              # y axis grows going up
        ray_directions_z = -torch.ones_like(ray_directions_x) # z inverted because cameras look in the -z direction
        ray_directions = torch.stack([ray_directions_x, ray_directions_y, ray_directions_z], -1)

        #ray_directions = ray_directions.repeat(initial_dimensions + [1, 1, 1]) # Using focals as tensors already gives the correct size

        # Creates focal normals and camera origins
        focal_normals = torch.zeros(initial_dimensions + [3], device=ray_directions.device)
        focal_normals[..., 2] = -1
        ray_origins = torch.zeros_like(focal_normals)

        return ray_directions, ray_origins, focal_normals

    @staticmethod
    def sample_rays_patched(ray_directions: torch.Tensor, observations: torch.Tensor,
                             patch_size: int, patch_count: int, bounding_boxes: torch.Tensor, weights: List[float]) -> Tuple[torch.Tensor]:
        '''
        Samples rays from the batch of rays and couples them with the corresponding observation
        The sampled rays form patches in the original image

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param observations: (..., 3, height, width) tensor with observations
        :param patch_size: size of each side of the patch to sample.
                           Must be a multiple of 2
        :param patch_count: number of patches to sample
        :param bounding_boxes: (..., 4, objects_count) tensor with (left, top, right, bottom) bounding boxes for each object
                                                       bounding box values are normalized in [0, 1]
        :param weights: [objects_count] list of weights to assign to each object. Rays will be sampled for each object
                                        in the proportion given by the weights, eliminating the effect of some objects
                                        being bigger than others
        :return: (..., parch_count * patch_size ^ 2, 3) tensor with sampled ray directions
                 (..., parch_count * patch_size ^ 2, 3) tensor with corresponding observations
        '''

        if patch_size % 2 != 0:
            raise Exception("Patch size must be a multiple of 2")

        # Computes observations in H,W,C order instead of C,H,W
        observation_axes_order = list(range(len(observations.size())))
        observation_axes_order = observation_axes_order[:-3] + [observation_axes_order[-2], observation_axes_order[-1], observation_axes_order[-3]]
        permuted_observations = observations.permute(observation_axes_order)

        # Computes the dimensions of the tensor
        initial_dimensions = ray_directions.size()[:-3]
        height = ray_directions.size()[-3]
        width = ray_directions.size()[-2]

        # Creates the tensor holding the spatial weights
        weight_masks = torch.zeros_like(permuted_observations[..., 0])
        flat_weight_masks, _ = TensorFolder.flatten(weight_masks, -2)  # (..., height, width)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -2)  # (..., 4, objects_count)

        # Denormalizes the boudning boxes and aligns them to the largest pixel
        flat_normalized_bounding_boxes = flat_bounding_boxes.clone()  # Avoids overwriting the original tensor
        flat_normalized_bounding_boxes[:, 0, :] = torch.floor(flat_normalized_bounding_boxes[:, 0, :] * width)
        flat_normalized_bounding_boxes[:, 2, :] = torch.ceil(flat_normalized_bounding_boxes[:, 2, :] * width)
        flat_normalized_bounding_boxes[:, 1, :] = torch.floor(flat_normalized_bounding_boxes[:, 1, :] * height)
        flat_normalized_bounding_boxes[:, 3, :] = torch.ceil(flat_normalized_bounding_boxes[:, 3, :] * height)

        objects_count = bounding_boxes.size(-1)
        scenes_count = flat_weight_masks.size(0)
        # Computes the weight mask for each image
        for scene_idx in range(scenes_count):
            current_bounding_box = flat_normalized_bounding_boxes[scene_idx]

            # Increases the weights in the bounding box of each object
            for object_idx in range(objects_count):
                # Gets the margins of the bounding box
                current_object_bounding_box = current_bounding_box[:, object_idx]
                left = int(current_object_bounding_box[0].item())
                top = int(current_object_bounding_box[1].item())
                right = int(current_object_bounding_box[2].item())
                bottom = int(current_object_bounding_box[3].item())

                current_bounding_box_area = (right - left) * (bottom - top)
                # Increases the weight in that region
                flat_weight_masks[scene_idx, top:bottom, left:right] += weights[object_idx] / current_bounding_box_area

        # Merges height and width in the  weight mask (..., height * width)
        flat_linearized_weight_mask = flat_weight_masks.reshape([-1, height * width])

        # Merges width and height and flattens the initial dimensions
        flat_ray_directions = ray_directions.reshape([-1, height * width, 3])
        flat_permuted_observations = permuted_observations.reshape([-1, height * width, 3])

        # Performs sampling
        all_ray_direction_samples = []
        all_observation_samples = []
        scenes_count = flat_ray_directions.size(0)
        # Takes samples from each input image
        for scene_idx in range(scenes_count):

            # Gets the current weights and normalizes them
            current_weights = flat_linearized_weight_mask[scene_idx]
            current_weights = current_weights / current_weights.sum()

            cdf = torch.cumsum(current_weights, dim=0)
            # Uniformly samples values in the cdf
            cdf_samples = torch.rand((patch_count,), device=current_weights.device)
            # Finds the indices corresponding to the sampled cdf position
            sample_indexes = torch.searchsorted(cdf, cdf_samples)
            sample_indexes = torch.clamp(sample_indexes, max=cdf.size(0) - 1)

            # Computes the indexes to sample for the current patches
            flat_indexes_to_sample = []
            for current_patch_idx in range(patch_count):
                # Index to sample expressed in flat coordinates
                flat_index_to_sample = sample_indexes[current_patch_idx].item()
                coordinate_center_to_sample = RayHelper.coordinate_from_flat_index(flat_index_to_sample, width)
                center_row, center_column = coordinate_center_to_sample

                # Shifts the patch away from the borders such that no pixel must be sampled out of the image
                half_patch_size = patch_size // 2
                center_row = max(half_patch_size, center_row)
                center_row = min(height - half_patch_size, center_row)
                center_column = max(half_patch_size, center_column)
                center_column = min(width - half_patch_size, center_column)

                # Computes the bounds for the patch
                start_row = center_row - half_patch_size
                end_row = center_row + half_patch_size
                start_column = center_column - half_patch_size
                end_column = center_column + half_patch_size

                # Computes the indexes for the complete patch
                for current_row in range(start_row, end_row):
                    for current_column in range(start_column, end_column):
                        flat_indexes_to_sample.append(RayHelper.flat_index_from_coordinate((current_row, current_column), width))

            # Samples with the indices
            current_ray_directions_samples = flat_ray_directions[scene_idx, flat_indexes_to_sample]
            current_observations_samples = flat_permuted_observations[scene_idx, flat_indexes_to_sample]

            all_ray_direction_samples.append(current_ray_directions_samples)
            all_observation_samples.append(current_observations_samples)

        flat_sampled_ray_directions = torch.stack(all_ray_direction_samples, dim=0)
        flat_sampled_observations = torch.stack(all_observation_samples, dim=0)

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        sampled_observations = TensorFolder.fold(flat_sampled_observations, initial_dimensions)

        return sampled_ray_directions, sampled_observations

    @ staticmethod
    def strided_patch_ray_samples_to_patch(samples: torch.Tensor):
        '''
        Transforms samples corresponding to a patch into the corresponding patch in CHW format

        :param samples: (..., patch_size ^ 2, features_count) tensor with feature samples forming a patch
        :return: (..., features_count, patch_size, patch_size) tensor with features reshaped to form a patch in CHW format
        '''

        # (..., features_count, patch_size ^ 2)
        samples = torch.transpose(samples, -1, -2)

        # Computes the new shape
        patch_size = int(math.sqrt(samples.size(-1)))
        original_size = list(samples.size())
        reshaped_size = original_size[:-1] + [patch_size, patch_size]

        # Reshapes the samples to their corresponding patch
        patches = samples.reshape(reshaped_size)
        return patches

    @staticmethod
    def split_strided_patch_ray_samples(samples: torch.Tensor, patch_size: int, strides: Union[List[int], int]) -> List[torch.Tensor]:
        '''
        Splits ray samples obtained with the 'sample_rays_strided_patch' function into the rays corresponding to each stride
        :param samples: (..., sum(patch_size_i ^ 2), features_count) tensor with samples to split
        :param patch_size: see 'sample_rays_strided_patch'
        :param strides: see 'sample_rays_strided_patch'
        :return: list of (..., patch_size_i ^ 2, features_count) tensor with splitted samples
        '''

        # Makes stride a list
        if not isinstance(strides, collections.Sequence):
            strides = [strides]
        smallest_stride = strides[0]

        patch_sizes = []
        for current_stride in strides:
            patch_sizes.append((patch_size * smallest_stride) // current_stride)

        # Performs the splitting
        splitted_samples = []
        current_begin_index = 0
        for current_patch_size in patch_sizes:
            # Each patch contains patch_size ^ 2 samples
            current_end_index = current_begin_index + current_patch_size ** 2
            splitted_samples.append(samples[..., current_begin_index:current_end_index, :])
            current_begin_index = current_end_index

        return splitted_samples

    @staticmethod
    def sample_rays_strided_patch(ray_directions: torch.Tensor, observations: torch.Tensor,
                                  patch_size: int, strides: Union[List[int], int], bounding_boxes: torch.Tensor,
                                  weights: List[float], align_grid=False) -> Tuple[torch.Tensor]:
        '''
        Samples rays from the batch of rays and couples them with the corresponding observation
        The sampled rays form patches in the original image. Each ray forming the patch is in a grid in the original image
        where points are stride pixels apart from each other. If stride = 1 samples are adjacent

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param observations: (..., 3, height, width) tensor with observations
        :param patch_size: size of each side of the patch to sample.
                           Must be a multiple of 2
        :param strides: distance in pixels between each sample in the grid of sampled locations.
                       if stride is a list, sampling is performed with each of the specified strides.
                       Strides must be sorted from smallest to biggest
                       patch size refers to the first stride and the other patch sizes will be inversely proportional to their stride.
                       The output will contain the samples at each stride concatenated together, starting with the samples
                       corresponding to the smallest stride
        :param bounding_boxes: (..., 4, objects_count) tensor with (left, top, right, bottom) bounding boxes for each object
                                                       bounding box values are normalized in [0, 1]
        :param weights: [objects_count] list of weights to assign to each object. Rays will be sampled for each object
                                        in the proportion given by the weights, eliminating the effect of some objects
                                        being bigger than others
        :param align_grid: if False, the sampled feature grid has no location constraints. Deprecated
                           if True, the sampled feature grid is sampled such that each sampled ray lies in the center
                           of (stride, stride) cell of pixels defined by dividing height and width in height/stride and width/stride
                           blocks of pixels
        :return: (..., patch_size ^ 2, 3) tensor with sampled ray directions
                 (..., patch_size ^ 2, 3) tensor with corresponding observations
                 (..., patch_size ^ 2, 2) tensor with sampled positions (height, width) from the top left corner normalized in [0, 1]
        '''

        if not align_grid:
            raise Exception("Align grid is required for patched ray sampling.")

        if patch_size % 2 != 0:
            raise Exception("Patch size must be a multiple of 2")

        # Makes stride a list
        if not isinstance(strides, collections.Sequence):
            strides = [strides]
        smallest_stride = strides[0]
        biggest_stride = strides[-1]

        # Checks the smallest patch is divisible by 2
        if (patch_size * smallest_stride) % (2 * biggest_stride) != 0:
            raise Exception("Patch size is not compatible with the chosen strides. Make patch size divisible by a higher power of 2")
        patch_sizes = []
        for current_stride in strides:
            patch_sizes.append((patch_size * smallest_stride) // current_stride)

        biggest_stride_patch_size = patch_sizes[-1]

        # Computes observations in H,W,C order instead of C,H,W
        observation_axes_order = list(range(len(observations.size())))
        observation_axes_order = observation_axes_order[:-3] + [observation_axes_order[-2], observation_axes_order[-1],
                                                                observation_axes_order[-3]]
        permuted_observations = observations.permute(observation_axes_order)

        # Computes the dimensions of the tensor
        initial_dimensions = ray_directions.size()[:-3]
        image_height = ray_directions.size()[-3]
        image_width = ray_directions.size()[-2]

        # Creates the tensor holding the spatial weights
        weight_masks = torch.zeros_like(permuted_observations[..., 0])
        flat_weight_masks, _ = TensorFolder.flatten(weight_masks, -2)  # (..., height, width)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -2)  # (..., 4, objects_count)

        # Denormalizes the boudning boxes and aligns them to the largest pixel
        flat_normalized_bounding_boxes = flat_bounding_boxes.clone()  # Avoids overwriting the original tensor
        flat_normalized_bounding_boxes[:, 0, :] = torch.floor(flat_normalized_bounding_boxes[:, 0, :] * image_width)
        flat_normalized_bounding_boxes[:, 2, :] = torch.ceil(flat_normalized_bounding_boxes[:, 2, :] * image_width)
        flat_normalized_bounding_boxes[:, 1, :] = torch.floor(flat_normalized_bounding_boxes[:, 1, :] * image_height)
        flat_normalized_bounding_boxes[:, 3, :] = torch.ceil(flat_normalized_bounding_boxes[:, 3, :] * image_height)

        objects_count = bounding_boxes.size(-1)
        scenes_count = flat_weight_masks.size(0)
        # Computes the weight mask for each image
        for scene_idx in range(scenes_count):
            current_bounding_box = flat_normalized_bounding_boxes[scene_idx]

            # Increases the weights in the bounding box of each object
            for object_idx in range(objects_count):
                # Gets the margins of the bounding box
                current_object_bounding_box = current_bounding_box[:, object_idx]
                left = int(current_object_bounding_box[0].item())
                top = int(current_object_bounding_box[1].item())
                right = int(current_object_bounding_box[2].item())
                bottom = int(current_object_bounding_box[3].item())

                current_bounding_box_area = (right - left) * (bottom - top)
                # Increases the weight in that region
                flat_weight_masks[scene_idx, top:bottom, left:right] += weights[object_idx] / current_bounding_box_area

        # Merges height and width in the  weight mask (..., height * width)
        flat_linearized_weight_mask = flat_weight_masks.reshape([-1, image_height * image_width])

        # Merges width and height and flattens the initial dimensions
        flat_ray_directions = ray_directions.reshape([-1, image_height * image_width, 3])
        flat_permuted_observations = permuted_observations.reshape([-1, image_height * image_width, 3])

        # Computes how much each pixel should be moded forward or backward based on the modulo with stride to be aligned
        # Used only if grid has to be aligned
        backward_map = list(range(biggest_stride // 2, biggest_stride)) + list(range(0, biggest_stride // 2))
        forward_map = list(range(biggest_stride // 2 + biggest_stride, biggest_stride, -1)) + [0] + list(range(biggest_stride - 1, biggest_stride // 2, -1))

        # Performs sampling
        all_ray_direction_samples = []
        all_observation_samples = []
        all_selected_linearized_indexes = []
        scenes_count = flat_ray_directions.size(0)
        # Takes samples from each input image
        for scene_idx in range(scenes_count):

            # Gets the current weights and normalizes them
            current_weights = flat_linearized_weight_mask[scene_idx]
            current_weights = current_weights / current_weights.sum()

            cdf = torch.cumsum(current_weights, dim=0)
            # Uniformly samples values in the cdf
            patch_count = 1  # sample a single patch
            cdf_samples = torch.rand((patch_count,), device=current_weights.device)
            # Finds the indices corresponding to the sampled cdf position
            sample_indexes = torch.searchsorted(cdf, cdf_samples)
            sample_indexes = torch.clamp(sample_indexes, max=cdf.size(0) - 1)

            # Computes the indexes to sample for the current patches
            flat_indexes_to_sample = []
            for current_patch_idx in range(patch_count):  # Not necessary cycle since patch_count is 1, but we keep it
                # Index to sample expressed in flat coordinates
                flat_index_to_sample = sample_indexes[current_patch_idx].item()
                coordinate_center_to_sample = RayHelper.coordinate_from_flat_index(flat_index_to_sample, image_width)
                center_row, center_column = coordinate_center_to_sample

                # Shifts the patch away from the borders such that no pixel must be sampled out of the image
                biggest_stride_half_patch_size = biggest_stride_patch_size // 2
                center_row = max(biggest_stride_half_patch_size * biggest_stride, center_row)
                center_row = min(image_height - biggest_stride * (biggest_stride_half_patch_size - 1) - 1, center_row)
                center_column = max(biggest_stride_half_patch_size * biggest_stride, center_column)
                center_column = min(image_width - biggest_stride * (biggest_stride_half_patch_size - 1) - 1, center_column)

                # Computes the bounds for the patch accounting for the stride
                biggest_stride_start_row = center_row - biggest_stride_half_patch_size * biggest_stride
                biggest_stride_start_column = center_column - biggest_stride_half_patch_size * biggest_stride

                if align_grid:
                    # Row is not aligned with the grid
                    row_stride_difference = biggest_stride_start_row % biggest_stride
                    if row_stride_difference != biggest_stride // 2:
                        if biggest_stride_start_row >= biggest_stride // 2:
                            biggest_stride_start_row -= backward_map[row_stride_difference]
                        else:
                            biggest_stride_start_row += forward_map[row_stride_difference]
                    # Column is not aligned with the grid
                    column_stride_difference = biggest_stride_start_column % biggest_stride
                    if column_stride_difference != biggest_stride // 2:
                        if biggest_stride_start_column >= biggest_stride // 2:
                            biggest_stride_start_column -= backward_map[column_stride_difference]
                        else:
                            biggest_stride_start_column += forward_map[column_stride_difference]

                # Samples the grid for each stride.
                for current_stride, current_patch_size in zip(strides, patch_sizes):

                    # Grids to sample do not start in the same pixel since grids with smaller stride are denser and thus
                    # start and end further. Compute the exact starting row and column
                    stride_start_offset = biggest_stride // 2 - current_stride // 2
                    current_start_row = biggest_stride_start_row - stride_start_offset
                    current_start_column = biggest_stride_start_column - stride_start_offset

                    # Computes the indexes for the complete patch
                    for current_row in range(current_start_row, current_start_row + current_stride * current_patch_size, current_stride):
                        for current_column in range(current_start_column, current_start_column + current_stride * current_patch_size, current_stride):
                            flat_indexes_to_sample.append(RayHelper.flat_index_from_coordinate((current_row, current_column), image_width))

            # Samples with the indices
            current_ray_directions_samples = flat_ray_directions[scene_idx, flat_indexes_to_sample]
            current_observations_samples = flat_permuted_observations[scene_idx, flat_indexes_to_sample]

            all_ray_direction_samples.append(current_ray_directions_samples)
            all_observation_samples.append(current_observations_samples)
            all_selected_linearized_indexes.append(torch.as_tensor(flat_indexes_to_sample, dtype=torch.int, device=observations.device))

        flat_selected_linearized_indexes = torch.stack(all_selected_linearized_indexes, dim=0)

        flat_sampled_ray_directions = torch.stack(all_ray_direction_samples, dim=0)
        flat_sampled_observations = torch.stack(all_observation_samples, dim=0)
        flat_sampled_positions = RayHelper.permutation_indices_to_positions(flat_selected_linearized_indexes, image_height, image_width)

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        sampled_observations = TensorFolder.fold(flat_sampled_observations, initial_dimensions)
        sampled_positions = TensorFolder.fold(flat_sampled_positions, initial_dimensions)

        return sampled_ray_directions, sampled_observations, sampled_positions

    @staticmethod
    def sample_all_rays_strided_grid(ray_directions: torch.Tensor, observations: torch.Tensor, strides: Union[List[int], int]) -> Tuple[torch.Tensor]:
        '''
        Samples rays from the batch of rays and couples them with the corresponding observation
        The rays are sampled with the same pattern as sample_rays_strided_patch, but the patch is considered to be the whole image

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param observations: (..., 3, height, width) tensor with observations
        :param strides: distance in pixels between each sample in the grid of sampled locations.
                       if stride is a list, sampling is performed with each of the specified strides.
                       Strides must be sorted from smallest to biggest
                       The output will contain the samples at each stride concatenated together, starting with the samples
                       corresponding to the smallest stride
        :return: (..., height/stride * width/stride, 3) tensor with sampled ray directions
                 (..., height/stride * width/stride, 3) tensor with corresponding observations
                 (..., height/stride * width/stride, 2) tensor with sampled positions (height, width) from the top left corner normalized in [0, 1]
        '''

        # Makes stride a list
        if not isinstance(strides, collections.Sequence):
            strides = [strides]

        # Puts the observations in HWC order
        observation_permutation_indexes = list(range(len(observations.size())))
        observation_permutation_indexes = observation_permutation_indexes[:-3] + observation_permutation_indexes[-2:] + observation_permutation_indexes[-3:-2]
        observations = observations.permute(observation_permutation_indexes)

        all_sampled_directions = []
        all_sampled_indices = []
        all_sampled_observations = []
        # Samples for each stride
        for current_stride in strides:
            current_sampled_directions, current_sampled_indices = RayHelper.sample_strided_grid(ray_directions, current_stride)
            current_sampled_observations, _ = RayHelper.sample_strided_grid(observations, current_stride)

            # Merges the height and width dimensions together
            current_sampled_directions = current_sampled_directions.reshape(list(current_sampled_directions.size())[:-3] + [-1] + [current_sampled_directions.size(-1)])
            current_sampled_indices = current_sampled_indices.reshape(list(current_sampled_indices.size())[:-3] + [-1] + [current_sampled_indices.size(-1)])
            current_sampled_observations = current_sampled_observations.reshape(list(current_sampled_observations.size())[:-3] + [-1] + [current_sampled_observations.size(-1)])

            all_sampled_directions.append(current_sampled_directions)
            all_sampled_indices.append(current_sampled_indices)
            all_sampled_observations.append(current_sampled_observations)

        # Concatenates along the number of samples dimension
        all_sampled_directions = torch.cat(all_sampled_directions, dim=-2)
        all_sampled_indices = torch.cat(all_sampled_indices, dim=-2)
        all_sampled_observations = torch.cat(all_sampled_observations, dim=-2)

        return all_sampled_directions, all_sampled_observations, all_sampled_indices

    @staticmethod
    def fold_strided_grid_samples(samples: torch.Tensor, strides: Union[List[int], int], original_size: Tuple[int], dim: int) -> List[torch.Tensor]:
        '''
        Folds the linearized output of 'sample_all_rays_strided_grid' into rectangular tensors

        :param samples: (..., sum(height/stride_i * width/stride_i), ...)
        :param strides: strides as defined in 'sample_all_rays_strided_grid' used to produce the samples
        :param image_size: (height, width) of the original image
        :param dim: dimension on which the samples are present in the samples tensor
        :return: list with a tensor for each stride of size (..., height/stride_i, width/stride_i, ...)
        '''

        # Makes stride a list
        if not isinstance(strides, collections.Sequence):
            strides = [strides]

        image_height, image_width = original_size

        rectangular_samples = []
        current_start_index = 0
        for current_stride in strides:
            # Checks the dimensions are divisible by the strides
            if image_height % current_stride != 0:
                raise Exception("The image height is not divisible by the stride")
            if image_width % current_stride != 0:
                raise Exception("The image width is not divisible by the stride")

            # Computes the size
            current_grid_height = image_height // current_stride
            current_grid_width = image_width // current_stride

            current_end_index = current_start_index + (current_grid_height * current_grid_width)

            # Slices the tensor along the specified dimension
            current_slice = slice(current_start_index, current_end_index)
            all_slices = [slice(0, None)] * len(samples.size())
            all_slices[dim] = current_slice
            current_samples = samples[all_slices]

            # Makes the flat samples rectangular
            current_samples_size = list(current_samples.size())
            current_samples_size[dim:dim+1] = [current_grid_height, current_grid_width]
            current_rectangular_samples = current_samples.reshape(current_samples_size)
            rectangular_samples.append(current_rectangular_samples)

            current_start_index = current_end_index

        return rectangular_samples

    @staticmethod
    def sample_strided_grid(tensor: torch.Tensor, stride: int):
        '''
        Samples from the features according to the given stride

        :param tensor: (..., height, width, features_count) tensor to sample
        :param stride: distance in pixels between each sample in the grid
        :return: (..., height/stride, width/stride, features_count) tensor with sampled_grid
                 (..., height/stride, width/stride, 2) tensor with the sampled locations normalized in [0, 1]
        '''

        # Computes the original size of the image
        image_height = tensor.size(-3)
        image_width = tensor.size(-2)

        # Checks the dimensions are divisible by the strides
        if image_height % stride != 0:
            raise Exception("The image height is not divisible by the stride")
        if image_width % stride != 0:
            raise Exception("The image width is not divisible by the stride")

        # Downsamples the features to make them match the bottleneck size
        # (..., features_count, grid_height, grid_height)
        grid_height = image_height // stride
        grid_width = image_width // stride
        center_pixel_offset = stride // 2  # The position in the (stride, stride) pixel grid
                                           # of the pixel corresponding to the center
        # Computes the indices of rows and columns that must be sampled to get center pixel
        row_indices = [idx * stride + center_pixel_offset for idx in range(grid_height)]
        column_indices = [idx * stride + center_pixel_offset for idx in range(grid_width)]
        # Samples rows and columns
        tensor = tensor[..., row_indices, :, :]
        tensor = tensor[..., column_indices, :]

        # Computes the indices that make the grid
        indices = []
        for current_row in row_indices:
            current_indices = []
            for current_column in column_indices:
                current_indices.append([current_row / image_height, current_column / image_width])
            indices.append(current_indices)

        # Converts the indices to tensors and makes it the same size as the tensor
        indices = torch.as_tensor(indices, dtype=torch.float, device=tensor.device)
        initial_dimensions = list(tensor.size())[:-3]
        for _ in initial_dimensions:
            indices = indices.unsqueeze(0)
        indices = indices.repeat(initial_dimensions + [1, 1, 1])

        return tensor, indices

    @staticmethod
    def coordinate_from_flat_index(index: int, image_width: int) -> Tuple[int, int]:
        '''
        Computes the coordinate that correspond to a given index expressed in flattened height * width coordinates
        :param index:
        :param image_width:
        :return:
        '''

        row = index // image_width
        column = index % image_width

        return row, column

    @staticmethod
    def flat_index_from_coordinate(coordinate: Tuple[int, int], image_width: int):
        '''
        Computes the index expressed in flattened height * width coordinates from the coordinates
        :param coordinate:
        :param image_width:
        :return:
        '''

        row, column = coordinate

        return (row * image_width) + column

    @staticmethod
    def sample_rays_weighted(ray_directions: torch.Tensor, observations: torch.Tensor,
                             samples_per_image: int, bounding_boxes: torch.Tensor, weights: List[float]) -> Tuple[torch.Tensor]:
        '''
        Samples rays from the batch of rays and couples them with the corresponding observation

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param observations: (..., 3, height, width) tensor with observations
        :param samples_per_image: number of samples to draw for each image.
                                  If 0, uses all available height * width samples
        :param bounding_boxes: (..., 4, objects_count) tensor with (left, top, right, bottom) bounding boxes for each object
                                                       bounding box values are normalized in [0, 1]
        :param weights: [objects_count] list of weights to assign to each object. Rays will be sampled for each object
                                        in the proportion given by the weights, eliminating the effect of some objects
                                        being bigger than others
        :return: (..., samples_per_image, 3) tensor with sampled ray directions
                 (..., samples_per_image, 3) tensor with corresponding observations
                 (..., samples_per_image, 2) tensor with sampled positions (height, width) from the top left corner normalized in [0, 1]
        '''

        # Computes observations in H,W,C order instead of C,H,W
        observation_axes_order = list(range(len(observations.size())))
        observation_axes_order = observation_axes_order[:-3] + [observation_axes_order[-2], observation_axes_order[-1], observation_axes_order[-3]]
        permuted_observations = observations.permute(observation_axes_order)

        # Computes the dimensions of the tensor
        initial_dimensions = ray_directions.size()[:-3]
        height = ray_directions.size()[-3]
        width = ray_directions.size()[-2]

        if samples_per_image > 0:
            # Creates the tensor holding the spatial weights
            weight_masks = torch.zeros_like(permuted_observations[..., 0])
            flat_weight_masks, _ = TensorFolder.flatten(weight_masks, -2)  # (..., height, width)
            flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -2)  # (..., 4, objects_count)

            # Denormalizes the boudning boxes and aligns them to the largest pixel
            flat_normalized_bounding_boxes = flat_bounding_boxes.clone()  # Avoids overwriting the original tensor
            flat_normalized_bounding_boxes[:, 0, :] = torch.floor(flat_normalized_bounding_boxes[:, 0, :] * width)
            flat_normalized_bounding_boxes[:, 2, :] = torch.ceil(flat_normalized_bounding_boxes[:, 2, :] * width)
            flat_normalized_bounding_boxes[:, 1, :] = torch.floor(flat_normalized_bounding_boxes[:, 1, :] * height)
            flat_normalized_bounding_boxes[:, 3, :] = torch.ceil(flat_normalized_bounding_boxes[:, 3, :] * height)

            objects_count = bounding_boxes.size(-1)
            scenes_count = flat_weight_masks.size(0)
            # Computes the weight mask for each image
            for scene_idx in range(scenes_count):
                current_bounding_box = flat_normalized_bounding_boxes[scene_idx]

                # Increases the weights in the bounding box of each object
                for object_idx in range(objects_count):
                    # Gets the margins of the bounding box
                    current_object_bounding_box = current_bounding_box[:, object_idx]
                    left = int(current_object_bounding_box[0].item())
                    top = int(current_object_bounding_box[1].item())
                    right = int(current_object_bounding_box[2].item())
                    bottom = int(current_object_bounding_box[3].item())

                    current_bounding_box_area = (right - left) * (bottom - top)
                    # Some objects may be very thin or out of frame, eg. backplate in tennis, so area may be 0
                    if current_bounding_box_area != 0:
                        # Increases the weight in that region
                        flat_weight_masks[scene_idx, top:bottom, left:right] += weights[object_idx] / current_bounding_box_area

            # Merges height and width in the  weight mask (..., height * width)
            flat_linearized_weight_mask = flat_weight_masks.reshape([-1, height * width])

        # Merges width and height and flattens the initial dimensions
        flat_ray_directions = ray_directions.reshape([-1, height * width, 3])
        flat_permuted_observations = permuted_observations.reshape([-1, height * width, 3])

        # If sampling is required we perform it
        if samples_per_image > 0:
            all_selected_permutation = []
            all_ray_direction_samples = []
            all_observation_samples = []
            scenes_count = flat_ray_directions.size(0)
            # Takes samples from each input image
            for scene_idx in range(scenes_count):

                # Gets the current weights and normalizes them
                current_weights = flat_linearized_weight_mask[scene_idx]
                current_weights = current_weights / current_weights.sum()

                cdf = torch.cumsum(current_weights, dim=0)
                # Uniformly samples values in the cdf
                cdf_samples = torch.rand((samples_per_image,), device=current_weights.device)
                # Finds the indices corresponding to the sampled cdf position
                sample_indexes = torch.searchsorted(cdf, cdf_samples)
                sample_indexes = torch.clamp(sample_indexes, max=cdf.size(0) - 1)

                # Samples with the indices
                current_ray_directions_samples = flat_ray_directions[scene_idx, sample_indexes]
                current_observations_samples = flat_permuted_observations[scene_idx, sample_indexes]

                all_ray_direction_samples.append(current_ray_directions_samples)
                all_observation_samples.append(current_observations_samples)
                all_selected_permutation.append(sample_indexes)

            flat_sampled_ray_directions = torch.stack(all_ray_direction_samples, dim=0)
            flat_sampled_observations = torch.stack(all_observation_samples, dim=0)
            flat_selected_permutation = torch.stack(all_selected_permutation, dim=0)
        # If all rays should be sampled just the flattening of height and width was needed
        else:
            flat_sampled_ray_directions = flat_ray_directions
            flat_sampled_observations = flat_permuted_observations
            # If no sampling is used, then the selected indices are in natural order from 0 to height * width - 1
            flat_selected_permutation = torch.arange(flat_ray_directions.size(-2), dtype=ray_directions.dtype, device=ray_directions.device)
            flat_selected_permutation = flat_selected_permutation.repeat([flat_ray_directions.size(0), 1])

        # Transforms indices from 0 to height * width - 1 to normalized height and width positions
        flat_sampled_positions = RayHelper.permutation_indices_to_positions(flat_selected_permutation, height, width)

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        sampled_observations = TensorFolder.fold(flat_sampled_observations, initial_dimensions)
        sampled_positions = TensorFolder.fold(flat_sampled_positions, initial_dimensions)

        return sampled_ray_directions, sampled_observations, sampled_positions

    @staticmethod
    def sample_rays(ray_directions: torch.Tensor, observations: torch.Tensor,
                    samples_per_image: int) -> Tuple[torch.Tensor]:
        '''
        Samples rays from the batch of rays and couples them with the corresponding observation

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param observations: (..., 3, height, width) tensor with observations
        :param samples_per_image: number of samples to draw for each image.
                                  If 0, uses all available height * width samples
        :return: (..., samples_per_image, 3) tensor with sampled ray directions
                 (..., samples_per_image, 3) tensor with corresponding observations
                 (..., samples_per_image, 2) tensor with sampled positions (height, width) from the top left corner normalized in [0, 1]
        '''

        # Computes observations in H,W,C order instead of C,H,W
        observation_axes_order = list(range(len(observations.size())))
        observation_axes_order = observation_axes_order[:-3] + [observation_axes_order[-2], observation_axes_order[-1], observation_axes_order[-3]]
        permuted_observations = observations.permute(observation_axes_order)

        # Computes the dimensions of the tensor
        initial_dimensions = ray_directions.size()[:-3]
        height = ray_directions.size()[-3]
        width = ray_directions.size()[-2]

        # Merges width and height and flattens the initial dimensions
        flat_ray_directions = ray_directions.reshape([-1, height * width, 3])
        flat_permuted_observations = permuted_observations.reshape([-1, height * width, 3])

        # If sampling is required we perform it
        if samples_per_image > 0:
            all_selected_permutation = []
            all_ray_direction_samples = []
            all_observation_samples = []
            scenes_count = flat_ray_directions.size(0)
            # Takes samples from each input image
            for scene_idx in range(scenes_count):
                # Samples for the current image
                permutation = torch.randperm(flat_ray_directions.size()[1], device=ray_directions.device)
                selected_permutation = permutation[:samples_per_image]
                current_ray_directions_samples = flat_ray_directions[scene_idx, selected_permutation]
                current_observations_samples = flat_permuted_observations[scene_idx, selected_permutation]

                all_ray_direction_samples.append(current_ray_directions_samples)
                all_observation_samples.append(current_observations_samples)
                all_selected_permutation.append(selected_permutation)

            flat_sampled_ray_directions = torch.stack(all_ray_direction_samples, dim=0)
            flat_sampled_observations = torch.stack(all_observation_samples, dim=0)
            flat_selected_permutation = torch.stack(all_selected_permutation, dim=0)
        # If all rays should be sampled just the flattening of height and width was needed
        else:
            flat_sampled_ray_directions = flat_ray_directions
            flat_sampled_observations = flat_permuted_observations
            # If no sampling is used, then the selected indices are in natural order from 0 to height * width - 1
            flat_selected_permutation = torch.arange(flat_ray_directions.size(-2), dtype=ray_directions.dtype, device=ray_directions.device)
            flat_selected_permutation = flat_selected_permutation.repeat([flat_ray_directions.size(0), 1])

        # Transforms indices from 0 to height * width - 1 to normalized height and width positions
        flat_sampled_positions = RayHelper.permutation_indices_to_positions(flat_selected_permutation, height, width)

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        sampled_observations = TensorFolder.fold(flat_sampled_observations, initial_dimensions)
        sampled_positions = TensorFolder.fold(flat_sampled_positions, initial_dimensions)

        return sampled_ray_directions, sampled_observations, sampled_positions

    @staticmethod
    def sample_rays_at_keypoints(ray_directions: torch.Tensor, keypoints: torch.Tensor, max_samples_per_image: int) -> Tuple[torch.Tensor]:
        '''
        Samples rays at the position specified by the keypoints.
        Samples are taken at points along the skeleton identified by keypoints.
        Samples are taken in the same segment position across observations and cameras for the same sequence
        Assumes keypoints in COCO ordering

        :param ray_directions: (..., observations_count, cameras_count, height, width, 3) tensor with ray directions
        :param keypoints: (..., observations_count, cameras_count, keypoints_count, 3) tensor with keypoints normalized in [0, 1]. The last dimension is in (height, width, confidence) format
        :param max_samples_per_image: maximum number of samples to draw for each image

        :return: (..., observations_count, cameras_count, samples_per_image, 3) tensor with sampled ray directions
                 (..., observations_count, cameras_count, samples_per_image, 2) tensor with sampled positions (height, width) from the top left corner normalized in [0, 1]
                 (..., observations_count, cameras_count, samples_per_image) tensor with confidence values corresponding to the sampled points
        '''

        # Defines the pairs of keypoints ids that should be connected
        segment_ids = [
            (0, 11),
            (0, 12),
            (5, 6),
            (5, 7),
            (5, 11),
            (5, 12),
            (6, 8),
            (6, 11),
            (6, 12),
            (7, 9),
            (8, 10),
            (11, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16)
        ]

        # Computes the dimensions of the tensor
        initial_dimensions = ray_directions.size()[:-5]
        height = ray_directions.size()[-3]
        width = ray_directions.size()[-2]
        cameras_count = ray_directions.size()[-4]
        observations_count = ray_directions.size()[-5]

        # Merges width and height and flattens the initial dimensions

        flat_ray_directions, _ = TensorFolder.flatten(ray_directions, -5)
        flat_keypoints, _ = TensorFolder.flatten(keypoints, -4)
        keypoints_count = flat_keypoints.size(-2)

        # Computes the beginning and end points of all segments
        all_segments = []
        for current_segment_ids in segment_ids:
            # (elements_count, observations_count, cameras_count, 3, 2) begin and end point of the current segment
            current_segment = torch.stack((flat_keypoints[..., current_segment_ids[0], :], flat_keypoints[..., current_segment_ids[1], :]), dim=-1)
            all_segments.append(current_segment)
        # (elements_count, observations_count, cameras_count, bones_count, 3, 2) tensor with beginning and end points of all segments
        segments = torch.stack(all_segments, dim=-3)

        # Computes the minimum number of times the segments count must be replicated to surpass the maximum number of keypoints
        segments_count = segments.size(-3)
        segments_replication_factor = max_samples_per_image // segments_count
        if max_samples_per_image % segments_count != 0:
            segments_replication_factor += 1
        # Repeats the segments to surpass the maximum number of keypoints and then cuts to max_samples_per_image
        segments = segments.repeat((1, 1, 1, segments_replication_factor, 1, 1))
        segments = segments[..., :max_samples_per_image, :, :]
        # segments is (elements_count, observations_count, cameras_count, max_samples_per_image, 3, 2)

        # Computes the position on each segment where to take the sample. The sampled position is the same across observations and cameras of the same sequence
        segments_samples_fractions_size = [segments.size(0), 1, 1, segments.size(3)] # (elements_count, 1, 1, max_samples_per_image)
        segments_samples_fractions = torch.rand(size=segments_samples_fractions_size, dtype=segments.dtype, device=segments.device)
        segments_samples_fractions = segments_samples_fractions.repeat(1, observations_count, cameras_count, 1)
        # (elements_count, max_samples_per_image, 1)
        segments_samples_fractions = segments_samples_fractions.unsqueeze(-1)
        # Computes the keypoints on the segments where we want to take samples
        keypoints_to_sample = segments[..., 0] + (segments[..., 1] - segments[..., 0]) * segments_samples_fractions

        flat_keypoint_to_sample_scores = keypoints_to_sample[..., -1]
        flat_keypoint_to_sample_positions = keypoints_to_sample[..., :2]
        # Do not use correct_range since keypoints coordinates are extracted at unknown high resolution, so the mapping error is unknown but small
        flat_sampled_ray_directions = RayHelper.sample_rays_at(flat_ray_directions, flat_keypoint_to_sample_positions, correct_range=False)

        # Old implementation
        """# Rescales the keypoints from [0, 1] to the full image size
        
        flat_ray_directions = ray_directions.reshape([-1, height * width, 3])
        
        flat_rescaled_keypoints = torch.clone(flat_keypoints)
        flat_rescaled_keypoints[..., 0] *= height
        flat_rescaled_keypoints[..., 1] *= width

        # Separates keypoints from scores. Converts keypoints positions to integers for indexing
        flat_rescaled_keypoints = flat_rescaled_keypoints[..., :2].round().long()
        flat_keypoints_scores = flat_rescaled_keypoints[..., -1]

        # Computes the positions to be sampled in the images
        # (..., keypoints_count==samples_per_image)
        positions_to_sample = flat_rescaled_keypoints[..., 0] * width + flat_rescaled_keypoints[..., 1]
        unsqueezed_positions_to_sample = positions_to_sample.unsqueeze(-1).repeat((1, 1, 3))
        # (..., keypoints_count==samples_per_image, 3)

        # Samples the ray directions at the specified point
        flat_sampled_ray_directions = torch.gather(flat_ray_directions, 1, unsqueezed_positions_to_sample)
        positions_to_sample = positions_to_sample[..., 0]
        flat_sampled_positions = RayHelper.permutation_indices_to_positions(positions_to_sample, height, width)"""

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        sampled_positions = TensorFolder.fold(flat_keypoint_to_sample_positions, initial_dimensions)
        sampled_scores = TensorFolder.fold(flat_keypoint_to_sample_scores, initial_dimensions)

        return sampled_ray_directions, sampled_positions, sampled_scores

    @staticmethod
    def sample_rays_at_object(ray_directions: torch.Tensor, images: torch.Tensor, samples_per_image: torch.Tensor, bounding_box: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Samples rays from the given bounding box area and couples them with the corresponding image features

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param images: (..., features_count, height, width) tensor with images
        :param samples_per_image: number of samples to draw for each image.
        :param bounding_box: (..., 4) tensor with (left, top, right, bottom) bounding boxes
                                      bounding box values are normalized in [0, 1]

        :return: (..., samples_per_image, 3) tensor with sampled ray directions
                 (..., samples_per_image, features_count) tensor with corresponding image features
                 (..., samples_per_image, 2) tensor with sampled positions (height, width) from the top left corner normalized in [0, 1]
        '''

        # Computes observations in H,W,C order instead of C,H,W
        images_axes_order = list(range(len(images.size())))
        images_axes_order = images_axes_order[:-3] + [images_axes_order[-2], images_axes_order[-1], images_axes_order[-3]]
        permuted_images = images.permute(images_axes_order)

        # Computes the dimensions of the tensor
        initial_dimensions = ray_directions.size()[:-3]
        height = ray_directions.size()[-3]
        width = ray_directions.size()[-2]
        features_count = images.size(-3)

        # Creates the tensor holding the spatial weights
        weight_masks = torch.zeros_like(permuted_images[..., 0])
        flat_weight_masks, _ = TensorFolder.flatten(weight_masks, -2)  # (..., height, width)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_box, -1)  # (..., 4)

        # Denormalizes the boudning boxes and aligns them to the largest pixel
        flat_normalized_bounding_boxes = flat_bounding_boxes.clone()  # Avoids overwriting the original tensor
        flat_normalized_bounding_boxes[:, 0] = torch.floor(flat_normalized_bounding_boxes[:, 0] * width)
        flat_normalized_bounding_boxes[:, 2] = torch.ceil(flat_normalized_bounding_boxes[:, 2] * width)
        flat_normalized_bounding_boxes[:, 1] = torch.floor(flat_normalized_bounding_boxes[:, 1] * height)
        flat_normalized_bounding_boxes[:, 3] = torch.ceil(flat_normalized_bounding_boxes[:, 3] * height)

        scenes_count = flat_weight_masks.size(0)
        # Computes the weight mask for each image
        for scene_idx in range(scenes_count):
            current_bounding_box = flat_normalized_bounding_boxes[scene_idx]

            # Gets the margins of the bounding box
            left = int(current_bounding_box[0].item())
            top = int(current_bounding_box[1].item())
            right = int(current_bounding_box[2].item())
            bottom = int(current_bounding_box[3].item())

            # Increases the weights in the bounding box
            current_bounding_box_area = (right - left) * (bottom - top)
            # Some objects may be very thin or out of frame, eg. backplate in tennis, so area may be 0
            if current_bounding_box_area != 0:
                # Increases the weight in that region
                flat_weight_masks[scene_idx, top:bottom, left:right] += 1

        # Merges height and width in the  weight mask (..., height * width)
        flat_linearized_weight_mask = flat_weight_masks.reshape([-1, height * width])

        # Merges width and height and flattens the initial dimensions
        flat_ray_directions = ray_directions.reshape([-1, height * width, 3])
        flat_permuted_observations = permuted_images.reshape([-1, height * width, features_count])

        # Perform sampling
        all_selected_permutation = []
        all_ray_direction_samples = []
        all_observation_samples = []
        scenes_count = flat_ray_directions.size(0)
        # Takes samples from each input image
        for scene_idx in range(scenes_count):

            # Gets the current weights and normalizes them
            current_weights = flat_linearized_weight_mask[scene_idx]
            current_weights = current_weights / current_weights.sum()

            cdf = torch.cumsum(current_weights, dim=0)
            # Uniformly samples values in the cdf
            cdf_samples = torch.rand((samples_per_image,), device=current_weights.device)
            # Finds the indices corresponding to the sampled cdf position
            sample_indexes = torch.searchsorted(cdf, cdf_samples)
            sample_indexes = torch.clamp(sample_indexes, max=cdf.size(0) - 1)

            # Samples with the indices
            current_ray_directions_samples = flat_ray_directions[scene_idx, sample_indexes]
            current_observations_samples = flat_permuted_observations[scene_idx, sample_indexes]

            all_ray_direction_samples.append(current_ray_directions_samples)
            all_observation_samples.append(current_observations_samples)
            all_selected_permutation.append(sample_indexes)

        flat_sampled_ray_directions = torch.stack(all_ray_direction_samples, dim=0)
        flat_sampled_observations = torch.stack(all_observation_samples, dim=0)
        flat_selected_permutation = torch.stack(all_selected_permutation, dim=0)

        # Transforms indices from 0 to height * width - 1 to normalized height and width positions
        flat_sampled_positions = RayHelper.permutation_indices_to_positions(flat_selected_permutation, height, width)

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        sampled_observations = TensorFolder.fold(flat_sampled_observations, initial_dimensions)
        sampled_positions = TensorFolder.fold(flat_sampled_positions, initial_dimensions)

        return sampled_ray_directions, sampled_observations, sampled_positions

    @staticmethod
    def sample_rays_at(ray_directions: torch.Tensor, sampled_positions: torch.Tensor, correct_range=True, original_image_size: Tuple[int,int]=None):
        '''
        Samples directions at the given sample positions

        :param ray_directions: (..., height, width, 3) tensor with ray directions
        :param sampled_positions: (..., samples_per_image, 2) tensor with positions to sample (height, width) from the top left corner normalized in [0, 1]
        :param correct_range: if normalized coordinates are obtained by dividing pixel_idx / dimension, when a 1 pixel error originates
                              if set to True the error is corrected
        :param original_image_size: (height, width) tuple with the original size of the image from which the sampled_positions were obtained and normalized..
                                    Needed if correct_range is true
        :return: (..., samples_per_image, 3) tensor with sampled ray directions
        '''

        flat_ray_directions, initial_dimensions = TensorFolder.flatten(ray_directions, -3)
        flat_sampled_positions, _ = TensorFolder.flatten(sampled_positions, -2)

        # Corrects the range of the sampled positions. Eg. if original size is 4 then [0, 0.25, 0.5, 0.75] -> [0, 0.33, 0.66, 1.0]
        if correct_range:
            original_image_size = torch.tensor(original_image_size, dtype=ray_directions.dtype, device=ray_directions.device)
            flat_sampled_positions = flat_sampled_positions * (original_image_size / (original_image_size - 1 + 1e-8))

        flat_sampled_positions = flat_sampled_positions[..., [1, 0]]

        # Puts in (batch_size, 3, height, width) format
        flat_permuted_ray_directions = flat_ray_directions.permute([0, 3, 1, 2])
        # Puts in (batch_size, samples_per_image, 1, 2) format to simulate a 2D tensor of width 1
        flat_sampled_positions = flat_sampled_positions.unsqueeze(-2)
        # Puts in range [-1, +1] for grid_sample
        flat_sampled_positions = (flat_sampled_positions - 0.5) * 2

        flat_sampled_ray_directions = F.grid_sample(flat_permuted_ray_directions, flat_sampled_positions, align_corners=True) # Align corners = True so that directions are not considered as pixels
        # Puts in (batch_size, 3, samples_per_image, 1)
        flat_sampled_ray_directions = flat_sampled_ray_directions.squeeze(-1)
        # Puts in (batch_size, samples_per_image, 3)
        flat_sampled_ray_directions = flat_sampled_ray_directions.permute([0, 2, 1])

        sampled_ray_directions = TensorFolder.fold(flat_sampled_ray_directions, initial_dimensions)
        return sampled_ray_directions

    @staticmethod
    def sample_features_at(features: torch.Tensor, sampled_positions: torch.Tensor, mode="bilinear", correct_range=True, original_image_size: Tuple[int,int]=None):
        '''
        Samples directions at the given sample positions

        :param features: (..., features_count, height, width) tensor with features
        :param sampled_positions: (..., samples_per_image, 2) tensor with positions to sample (height, width) from the top left corner normalized in [0, 1]
        :param correct_range: if normalized coordinates are obtained by dividing pixel_idx / dimension, when a 1 pixel error originates
                              if set to True the error is corrected
        :param original_image_size: (height, width) tuple with the original size of the image from which the sampled_positions were obtained and normalized..
                                    Needed if correct_range is true
        :return: (..., samples_per_image, features_count) tensor with sampled features
        '''

        flat_features, initial_dimensions = TensorFolder.flatten(features, -3)
        flat_sampled_positions, _ = TensorFolder.flatten(sampled_positions, -2)

        # Corrects the range of the sampled positions. Eg. if original size is 4 then [0, 0.25, 0.5, 0.75] -> [0, 0.33, 0.66, 1.0]
        if correct_range:
            original_image_size = torch.tensor(original_image_size, dtype=features.dtype, device=features.device)
            flat_sampled_positions = flat_sampled_positions * (original_image_size / (original_image_size - 1 + 1e-8))

        flat_sampled_positions = flat_sampled_positions[..., [1, 0]]

        # Puts in (batch_size, samples_per_image, 1, 2) format to simulate a 2D tensor of width 1
        flat_sampled_positions = flat_sampled_positions.unsqueeze(-2)
        # Puts in range [-1, +1] for grid_sample
        flat_sampled_positions = (flat_sampled_positions - 0.5) * 2

        flat_sampled_features = F.grid_sample(flat_features, flat_sampled_positions, mode=mode, align_corners=True) # Align corners = True so that directions are not considered as pixels
        # Puts in (batch_size, 3, samples_per_image, 1)
        flat_sampled_features = flat_sampled_features.squeeze(-1)
        # Puts in (batch_size, samples_per_image, features_count)
        flat_sampled_features = flat_sampled_features.permute([0, 2, 1])

        sampled_features = TensorFolder.fold(flat_sampled_features, initial_dimensions)
        return sampled_features

    @staticmethod
    def sample_original_region_from_patch_samples(observations: torch.Tensor, sampled_positions: torch.Tensor, stride: int):
        '''
        Samples directions at the given sample positions

        :param observations: (..., features_count, height, width) tensor with features
        :param sampled_positions: (..., samples_per_image, 2) tensor with positions (height, width) from the top left corner normalized in [0, 1] representing
                                                              the grid of positions in the original image sampled to obtain the patch
                                                              samples_per_image = patch_size * patch_size
        :param stride: stride value used to produce the patch of samples.

        :return: (..., features_count, patch_size * stride, patch_size * stride) tensor with the original image region corresponding
                                                                                 to patch samples
        '''

        original_image_height = observations.size(-2)
        original_image_width = observations.size(-1)
        squared_patch_size = sampled_positions.size(-2)

        # Gets the size of the patch in feature space and in image space
        feature_space_patch_size = int(math.sqrt(squared_patch_size))
        image_space_patch_size = feature_space_patch_size * stride

        original_image_size = (original_image_height, original_image_width)
        original_image_size = torch.tensor(original_image_size, dtype=observations.dtype, device=observations.device)

        flat_observations, initial_dimensions = TensorFolder.flatten(observations, -3)
        flat_sampled_positions, _ = TensorFolder.flatten(sampled_positions, -2)

        elements_count = flat_observations.size(0)

        # Obtains original sampled coordinates
        flat_sampled_positions = (flat_sampled_positions * original_image_size).round()
        # Obtains the top left corners of the block of pixel corresponding to each sample
        flat_sampled_positions = (flat_sampled_positions / stride).long() * stride
        # The first and the last sample correspond to the corners of the patch to sample

        # (elements_count, 2)
        top_left_coordinate = flat_sampled_positions[:, 0]
        bottom_right_coordinate = flat_sampled_positions[:, -1] + stride

        grid_side = torch.tensor(range(image_space_patch_size), dtype=observations.dtype, device=observations.device)
        grid_rows, grid_columns = torch.meshgrid(grid_side, grid_side)
        # (image_space_patch_size, image_space_patch_size, 2)
        original_region_grid = torch.stack([grid_rows, grid_columns], dim=-1)
        # (elements_count, image_space_patch_size, image_space_patch_size, 2)
        original_region_grid.unsqueeze(0).repeat(elements_count, 1, 1, 1)
        # Offsets the grid to cover its corresponding location in each image
        original_region_grid = original_region_grid + top_left_coordinate.unsqueeze(1).unsqueeze(1)

        # Puts the range in [0, 1]
        # Divided by size - 1 since the last index must map to 1 in grid_sample
        original_region_grid = original_region_grid / (original_image_size - 1)

        # Transforms range [0, 1] -> [-1, 1]
        original_region_grid = (original_region_grid - 0.5) * 2
        # Swaps rows with columns as expected by grid_sample
        original_region_grid = original_region_grid[..., [1, 0]]

        # Pixel correspondence is exact, so we can use nearest which in this case is equivalent to bilinear
        flat_sampled_original_regions = F.grid_sample(flat_observations, original_region_grid, mode="nearest", align_corners=True)

        original_regions = TensorFolder.fold(flat_sampled_original_regions, initial_dimensions)
        return original_regions

    @staticmethod
    def permutation_indices_to_positions(permutation_indices: torch.Tensor, height: int, width: int) -> torch.Tensor:
        '''
        Transforms a tensor with sample permutation indices in [0, height * width - 1] in a tensor of height and width positions

        :param permutation_indices: (..., samples_per_image) tensor with permutation indices
        :param height: height of the observations tensor
        :param width: width of the observations tensor
        :return: (..., samples_per_image, 2) tensor with height and width positions normalized in [0, 1] relative to the
                                             top left corner
        '''

        # Computes rows and columns
        rows_tensor = permutation_indices // width
        columns_tensor = permutation_indices % width

        # Normalizes the output.
        rows_tensor = rows_tensor / height
        columns_tensor = columns_tensor / width

        output_tensor = torch.stack([rows_tensor, columns_tensor], dim=-1)
        return output_tensor

    @staticmethod
    def transform_points(points: torch.Tensor, transformation_matrix: torch.Tensor, rotation=True, translation=True):
        '''
        Transforms points according to the given transformation matrix
        :param points: (..., 3) tensor of points
        :param transformation_matrix: (..., 4, 4) tensor of transformation matrices
        :param rotation: true if points should be rotated
        :param translation: true if points should be translated
        :return:
        '''

        transformed_points = points

        # Applies rotation (matrix multiplication)
        if rotation:
            transformed_points = torch.sum(transformed_points.unsqueeze(-2) * transformation_matrix[..., :3, :3], -1)

        # Applies translation
        if translation:
            transformed_points = transformed_points + transformation_matrix[..., :3, -1]

        return transformed_points

    @staticmethod
    def transform_rays(ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor, transformation_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Transforms the rays according to the given transformation matrix. Directions are rotated

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param focal_normals: (..., 3) tensor with focal normals
        :param transformation_matrix: (..., 4, 4) tensor with transformation matrix for each image
        :return: (..., 3) tensor with transformed ray origins
                 (..., samples_per_image, 3) tensor with transformed ray directions
                 (..., 3) tensor with transformed focal normals
        '''

        transformed_ray_origins = RayHelper.transform_points(ray_origins, transformation_matrix)
        transformed_focal_normals = RayHelper.transform_points(focal_normals, transformation_matrix, translation=False)

        samples_per_image = ray_directions.size()[-2]
        initial_dimensions = ray_directions.size()[:-2]
        # Puts the transformation matrix in the same format as the ray directions by adding the samples_per_image dimension
        transformation_matrix = transformation_matrix.unsqueeze(-3).repeat(([1] * len(initial_dimensions)) + [samples_per_image, 1, 1])

        transformed_ray_directions = RayHelper.transform_points(ray_directions, transformation_matrix, translation=False)

        return transformed_ray_origins, transformed_ray_directions, transformed_focal_normals

    @staticmethod
    def create_ray_positions(ray_origins: torch.Tensor, ray_directions: torch.Tensor, z_near: torch.Tensor, z_far: torch.Tensor, positions_count: int,
                             perturb: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Creates positions_count points along each specified ray between z_near and z_far

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param z_near: (..., [samples_per_image]) minimum distance for the generated points from the ray origin measured perpendicularly to the focal plane
                                                  if the sample_per_image dimension is present specifies the distance for each ray
        :param z_far: (..., [samples_per_image]) maximum distance for the generated points from the ray origin measured perpendicularly to the focal plane
                                                 if the sample_per_image dimension is present specifies the distance for each ray
        :param positions_count: number of points to sample along each ray
        :param perturb: True if sampled points are to be perturbed
        :return: (..., samples_per_image, positions_count, 3) tensor with sampled positions along each ray
                 (..., samples_per_image, positions_count) tensor with distances of each sampled points from the focal plane, measured perpendicularly to it
        '''

        if not torch.is_tensor(z_near):
            z_near = torch.ones(ray_origins.size()[:-1], dtype=torch.float32).cuda() * z_near
        if not torch.is_tensor(z_far):
            z_far = torch.ones(ray_origins.size()[:-1], dtype=torch.float32).cuda() * z_far

        # Gets uniform positions
        ray_positions_t = torch.linspace(0.0, 1.0, positions_count).cuda()

        # Translates between near and far
        ray_positions_t = z_near.unsqueeze(-1) * (1.0 - ray_positions_t) + z_far.unsqueeze(-1) * (ray_positions_t)

        # If z_near and z_far are defined for each origin we need to add the samples_per_image dimension
        if len(z_near.size()) == len(ray_origins.size()) - 1:

            # Creates space for the samples_per_image_dimension
            ray_positions_t = ray_positions_t.unsqueeze(-2)
            # Adds the samples_per_image_dimension
            dimensions_to_add = list(ray_directions.size()[:-1])
            ray_positions_t = ray_positions_t.repeat([1] * (len(dimensions_to_add) - 1) + [dimensions_to_add[-1], 1])


        # Moves the points randomly in their intervals
        if perturb:
            mid_points = (ray_positions_t[..., 1:] + ray_positions_t[..., :-1]) / 2
            upper = torch.cat([mid_points, ray_positions_t[..., -1:]], dim=-1)
            lower = torch.cat([ray_positions_t[..., :1], mid_points], dim=-1)

            # stratified samples in those intervals
            ray_positions_t_rand = torch.rand(ray_positions_t.size()).cuda()

            ray_positions_t = lower + (upper - lower) * ray_positions_t_rand

        # Generates the ray positions. Unsqueeze creates the dimension for the samples per image, the positions_count points and for the 3 spatial dimensions
        ray_positions = ray_origins.unsqueeze(-2).unsqueeze(-2) + ray_directions.unsqueeze(-2) * ray_positions_t.unsqueeze(-1)

        return ray_positions, ray_positions_t

    @staticmethod
    def transform_ray_positions(ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor,
                                ray_positions: torch.Tensor, transformation_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Transforms the rays according to the given transformation matrix
        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param focal_normals: (..., 3) tensor with focal normals
        :param ray_positions: (..., samples_per_image, positions_count, 3) tensor with sampled positions along each ray
        :param transformation_matrix: (..., 4, 4) tensor with transformation matrix for each image
        :return: (..., 3) tensor with transformed ray origins. Only translation is applied
                 (..., samples_per_image, 3) tensor with transformed ray directions. Only rotation is applied
                 (..., 3) tensor with transformed focal normals. Only rotation is applied
                 (..., positions_count, 3) tensor with transformed positions along each ray
        '''

        transformed_ray_origins = RayHelper.transform_points(ray_origins, transformation_matrix)
        transformed_focal_normals = RayHelper.transform_points(focal_normals, transformation_matrix, translation=False)

        positions_count = ray_positions.size(-2)
        samples_per_image = ray_positions.size(-3)

        # Adds the samples_per_image dimension to the transformation matrix
        transformation_matrix = transformation_matrix.unsqueeze(-3).repeat([1] * (len(transformation_matrix.size()) - 2) + [samples_per_image, 1, 1])

        # Transforms directions
        transformed_ray_directions = RayHelper.transform_points(ray_directions, transformation_matrix, translation=False)

        # Adds the positions_count dimension to the transformation matrix
        transformation_matrix = transformation_matrix.unsqueeze(-3)
        transformation_matrix = transformation_matrix.repeat([1] * (len(transformation_matrix.size()) - 2) + [positions_count, 1, 1])

        transformed_ray_positions = RayHelper.transform_points(ray_positions, transformation_matrix)

        return transformed_ray_origins, transformed_ray_directions, transformed_focal_normals, transformed_ray_positions

    @staticmethod
    def create_ray_positions_weighted(ray_origins: torch.Tensor, ray_directions: torch.Tensor, positions_count: int,
                                      reference_ray_positions_t: torch.Tensor, weights: torch.Tensor, perturb: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Creates positions_count points along each specified ray according to a weighting functions

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param positions_count: number of points to sample along each ray
        :param reference_ray_positions_t: (..., reference_positions_count) tensor with distances of each reference sampled point from the focal plane, measured perpendicularly to it
        :param weights: (..., reference_positions_count) tensor with weights to assign to each reference poisition for sampling
        :param perturb: True if sampled points are to be perturbed
        :return: (..., samples_per_image, positions_count + reference_positions_count, 3) tensor with sampled positions along each ray
                 (..., samples_per_image, positions_count + reference_positions_count) tensor with distances of each sampled points from the focal plane, measured perpendicularly to it
        '''

        # Computes mid points
        reference_mid_values = (reference_ray_positions_t[..., 1:] + reference_ray_positions_t[..., :-1]) / 2
        # Resamples positions according to their weight
        ray_positions_t = RayHelper.sample_pdf(reference_mid_values, weights[..., 1:-1], positions_count, perturb)
        ray_positions_t = ray_positions_t.detach()

        # Merges the reference positions with the new ones and obtains the rays
        merged_ray_positions_t, _ = torch.sort(torch.cat([reference_ray_positions_t, ray_positions_t], dim=-1), dim=-1)
        ray_positions = ray_origins.unsqueeze(-2).unsqueeze(-2) + ray_directions.unsqueeze(-2) * merged_ray_positions_t.unsqueeze(-1)  # (..., positions_count + reference_positions_count, 3)

        return ray_positions, merged_ray_positions_t

    @staticmethod
    def sample_pdf(bin_delimiters: torch.Tensor, weights: torch.Tensor, positions_count: int, perturb: bool) -> torch.Tensor:
        '''
        Computes samples form a weighted distribution

        :param bin_delimiters: (..., reference_positions - 1) tensor with center points along which to perform sampling
                                                    sampled points lie between the first and the last bin. Bins
                                                    are the portions between each value in this tensor
        :param weights: (..., reference_positions - 2) weights to assign to each bin. Since bins are one less than the
                                                       bin delimiting values, there is one less weight
        :param positions_count: number of points to sample
        :param perturb: True if samples have to be perturbed
        :return: (..., positions_count) tensor with new sample positions
        '''

        # Get pdf
        weights += 1e-5  # prevent nans
        pdf = weights / torch.sum(weights, dim=-1, keepdims=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        # Take uniform samples
        if not perturb:
            cdf_samples = torch.linspace(0., 1., positions_count).cuda()

            # Makes of the same dimension as the cdf
            dimensions_to_add = list(cdf.shape[:-1])
            for _ in range(len(dimensions_to_add)):
                cdf_samples = cdf_samples.unsqueeze(0)
            cdf_samples = cdf_samples.repeat(dimensions_to_add + [1])
        # Take random samples
        else:
            cdf_samples = torch.rand(list(cdf.shape[:-1]) + [positions_count]).cuda()

        # Finds the indices that correspond to the positions of u in cdf
        cdf_sample_indexes = torch.searchsorted(cdf, cdf_samples, right=True)
        # Clamps to avoid out of bounds
        below = torch.clamp(cdf_sample_indexes - 1, min=0)  # Indexes of values greater or equal
        above = torch.clamp(cdf_sample_indexes, max=cdf.size(-1) - 1)  # Indexes of immediately greater values

        # Trick to get in gather both below and above values at the same time
        cdf_sample_indexes_stacked = torch.stack([below, above], dim=-1)

        # Retrieves the values of the cdf at the below and above positions.
        # below reside in idx 0 of the last dimension, above in the idx 1
        cdf_g = torch.gather(torch.stack([cdf] * 2, dim=-1), -2, cdf_sample_indexes_stacked)
        # same for bins
        bins_g = torch.gather(torch.stack([bin_delimiters] * 2, dim=-1), -2, cdf_sample_indexes_stacked)

        normalization = (cdf_g[..., 1] - cdf_g[..., 0])
        # Values that are too small are replaced by 1
        normalization = torch.where(normalization < 1e-5, torch.ones_like(normalization), normalization)
        t = (cdf_samples - cdf_g[..., 0]) / normalization
        positions_t = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return positions_t


if __name__ == "__main__":

    def test1():
        batch_size = 2
        observations_count = 4
        cameras_count = 5
        height = 50
        width = 100
        focal = 100
        positions_count = 91

        samples_per_image = 0

        observations = torch.zeros((batch_size, observations_count, cameras_count, 3, height, width)).cuda()

        ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays([batch_size, observations_count, cameras_count], height, width, focal)
        sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays(ray_directions, observations, samples_per_image)

        transformation = PoseParameters([np.pi/2, 0, 0], [0, 0, 0])
        transformation_matrix = transformation.as_homogeneous_matrix_torch().cuda()

        transformation_matrix = transformation_matrix.unsqueeze(0).unsqueeze(0).repeat([batch_size, observations_count, cameras_count, 1, 1])

        transformed_ray_origins, transformed_ray_directions, transformed_focal_normals = RayHelper.transform_rays(ray_origins, sampled_directions, focal_normals, transformation_matrix)

        ray_positions, ray_positions_t = RayHelper.create_ray_positions(transformed_ray_origins, transformed_ray_directions, 10, 100, positions_count, perturb=False)

        transformed_ray_origins, transformed_ray_directions, transformed_focal_normals, transformed_ray_positions = RayHelper.transform_ray_positions(transformed_ray_origins, transformed_ray_directions, transformed_focal_normals, ray_positions, transformation_matrix)

        weights = torch.rand(ray_positions_t.size()).cuda()
        weights[..., -2] *= 100000

        resampled_positions, resampled_positions_t = RayHelper.create_ray_positions_weighted(transformed_ray_origins, transformed_ray_directions, positions_count, ray_positions_t, weights, True)

    def test2():
        batch_size = 1
        observations_count = 1
        cameras_count = 1
        height = 50
        width = 100
        focal = 100

        objects_count = 2
        weights = [1, 5]

        samples_per_image = 100

        observations = torch.zeros((batch_size, observations_count, cameras_count, 3, height, width)).cuda()
        ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays([batch_size, observations_count, cameras_count], height, width, focal)

        bounding_boxes = torch.zeros((batch_size, observations_count, cameras_count, 4, objects_count))
        bounding_boxes[..., 0, 0] = 0
        bounding_boxes[..., 1, 0] = 0
        bounding_boxes[..., 2, 0] = 0.5
        bounding_boxes[..., 3, 0] = 0.5

        bounding_boxes[..., 0, 1] = 0.5
        bounding_boxes[..., 1, 1] = 0.5
        bounding_boxes[..., 2, 1] = 1
        bounding_boxes[..., 3, 1] = 1

        sampled_directions, sampled_observations = RayHelper.sample_rays_weighted(ray_directions, observations, samples_per_image, bounding_boxes, weights)
        pass

    def test3():
        batch_size = 1
        height = 2
        width = 2

        keypoints_count = 5

        ray_directions = torch.ones((batch_size, height, width, 3))
        ray_directions[0, 0, 0, 0] = -1.0
        ray_directions[0, 0, 0, 1] = 1.0
        ray_directions[0, 0, 1, 0] = 1.0
        ray_directions[0, 0, 1, 1] = 1.0
        ray_directions[0, 1, 0, 0] = -1.0
        ray_directions[0, 1, 0, 1] = -1.0
        ray_directions[0, 1, 1, 0] = -1.0
        ray_directions[0, 1, 1, 1] = 1.0

        sample_positions = torch.zeros((batch_size, keypoints_count, 3))
        sample_positions[0, 0, 0] = 0.0
        sample_positions[0, 0, 1] = 0.0
        sample_positions[0, 0, 2] = 0.5

        sample_positions[0, 1, 0] = 0.0
        sample_positions[0, 1, 1] = 1.0
        sample_positions[0, 1, 2] = 0.6


        sample_positions[0, 2, 0] = 1.0
        sample_positions[0, 2, 1] = 0.0
        sample_positions[0, 2, 2] = 0.7

        sample_positions[0, 3, 0] = 1.0
        sample_positions[0, 3, 1] = 1.0
        sample_positions[0, 3, 2] = 0.8

        sample_positions[0, 4, 0] = 0.95
        sample_positions[0, 4, 1] = 0.95
        sample_positions[0, 4, 2] = 0.9

        sampled_rays = RayHelper.sample_rays_at_keypoints(ray_directions, sample_positions, keypoints_count)
        pass

    def test4():
        batch_size = 1
        height = 4
        width = 8

        downsample_factor = 2

        features = torch.tensor(list(range(width // downsample_factor)), dtype=torch.float)
        # (bs, 1, 1, 4)
        features = features.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        features = features.repeat(1, 1, height // downsample_factor, 1)
        features[0, 0, 1] += 10

        sample_positions = torch.tensor([[0.5, 0.0], [0.5, 0.125], [0.5, 0.25], [0.5, 0.375], [0.5, 0.5], [0.5, 0.625], [0.5, 0.75], [0.5, 0.875]])
        # (bs, 4, 2)
        sample_positions = sample_positions.unsqueeze(0)

        sampled_features = RayHelper.sample_features_at(features, sample_positions, mode="nearest", correct_range=True, original_image_size=(height, width))

        pass

    def test5():
        batch_size = 1
        height = 8
        width = 8

        downsample_factor = 2
        data = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 3, 8],
            [0, 0, 0, 0, 1, 0, 1, 2],
            [0, 0, 0, 0, 1, 4, 2, 3],
            [0, 0, 0, 0, 1, 0, 1, 4],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        features = torch.tensor(data, dtype=torch.float)
        # (bs, 1, 8, 8)
        features = features.unsqueeze(0).unsqueeze(0)

        sample_positions = torch.tensor([[3/8, 5/8], [3/8, 7/8], [5/8, 5/8], [5/8, 7/8]])
        # (bs, 4, 2)
        sample_positions = sample_positions.unsqueeze(0)

        sampled_features = RayHelper.sample_original_region_from_patch_samples(features, sample_positions, stride=downsample_factor)

        pass

    def test6():
        batch_size = 5
        height = 4
        width = 8

        strides = [2, 4]
        directions= torch.zeros((batch_size, height, width, 3))
        observations = torch.zeros((batch_size, 3, height, width))

        directions, observations, indices = RayHelper.sample_all_rays_strided_grid(directions, observations, strides)
        folded_directions = RayHelper.fold_strided_grid_samples(directions, strides, (height, width), dim=-2)

        pass


    test6()

    a = torch.ones((2, 3, 4, 5))
    my_tuple = (0, 0, slice(1, 3), 0)
    res = a[my_tuple]

    a[0] = 0

    g = torch.ones((2, 3), dtype=torch.bool)
    g[0, 0] = False

    b = a[g]


    pass