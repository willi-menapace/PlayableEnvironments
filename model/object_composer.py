import importlib
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.object_ids_helper import ObjectIDsHelper
from utils.configuration import Configuration
from utils.lib_3d.bounding_box import BoundingBox
from utils.lib_3d.pose_parameters import PoseParameters
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_broadcaster import TensorBroadcaster
from utils.tensor_folder import TensorFolder


class ObjectComposer(nn.Module):

    def __init__(self, config):
        super(ObjectComposer, self).__init__()

        self.config = config

        # Creates the models representing each object in the composition
        models_coarse = self.create_object_models(fine=False)
        models_fine = self.create_object_models(fine=True)
        self.object_models_coarse = nn.ModuleList(models_coarse)
        self.object_models_fine = nn.ModuleList(models_fine)

        # Whether to apply activation to the outputs of the models
        self.apply_activation = self.config["model"]["apply_activation"]
        if self.object_models_coarse[0].model_config["nerf_model"]["output_features"] != 3 and self.apply_activation:
            raise Exception("The application of activations to the nerf output is requested, but the model seem not to output colors directly. Please make sure this is the behavior you desire")

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

    def create_object_models(self, fine: bool) -> List[nn.Module]:
        '''
        Creates the model representing each object in the composition
        :param fine: True if the models to be created are fine models
        :return: list of created object models
        '''

        object_models = []
        # Creates the model for each object as specified in the configuration
        for current_object_config in self.config["model"]["object_models"]:
            # Do not create the fine model if we do not use it
            if fine and "use_fine" in current_object_config and current_object_config["use_fine"] == False:
                current_model = None
            else:
                model_class = current_object_config["architecture"]
                current_model = getattr(importlib.import_module(model_class), 'model')(self.config, current_object_config)
            object_models.append(current_model)

        return object_models

    def set_step(self, current_step: int):
        '''
        Sets the current step to the specified value
        :param current_step:
        :return:
        '''

        # Sets the step in all object models
        for current_object_model in self.object_models_coarse:
            current_object_model.set_step(current_step)
        for current_object_model in self.object_models_fine:
            if current_object_model is not None:
                current_object_model.set_step(current_step)

    def compute_object_z_bounds(self, ray_origins: torch.Tensor, focal_normals: torch.Tensor, bounding_box: BoundingBox) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes the z_near and z_far values such that [ray_origin + focal_normals * z_near, ray_origin + focal_normals * z_far] represents
        the segment in R^3 perpendicular to the focal plane such that it is the smallest that comprises all the
        projections of the bounding box points on the axis perpendicular to the focal plane

        :param ray_origins: (..., 3) tensor with ray origins
        :param focal_normals: (..., 3) tensor with focal normals
        :param bounding_box: bounding box object of interest
        :return: (...), (...) tensors with z_near and z_far values
        '''

        # Obtains the points that compose the bounding box
        corner_points = bounding_box.get_corner_points()

        # Translates the coordinate system to be centered on the camera origin
        # Also adds the ... dimensions to corner points and the 8 dimension to ray origins
        corner_points = corner_points - ray_origins.unsqueeze(-2)
        # Adds the 8 dimension to the focal normals
        focal_normals = focal_normals.unsqueeze(-2)

        # Dot product on the last dimension
        # cos(normals, corners) * ||corners||
        projections = torch.sum(corner_points * focal_normals, dim=-1)

        # Computes the closest and the furthest projection
        z_near = torch.min(projections, dim=-1)[0]
        z_far = torch.max(projections, dim=-1)[0]

        return z_near, z_far

    def compute_raywise_object_z_bounds(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, bounding_box: BoundingBox, object_validity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Computes the z_near and z_far values for each ray such that [ray_origin + ray_direction * z_near, ray_origin + ray_direction * z_far] represents
        the portion of the ray in R^3 that intersects the bounding box

        If the object is not present in the scene returns z_near = z_far at 0.0

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param bounding_box: bounding box object of interest
        :param object_validity (...) boolean tensor with True if the object is present in the scene
        :return: (..., samples_per_image), (..., samples_per_image) tensors with z_near and z_far values
        '''

        eps = 1e-6

        # Obtains the points that compose the lower and higher part of the bounding box
        # (2, 3)
        corner_points = bounding_box.get_corner_points()[[0, 6]]

        # Subtracts the origin (..., 2, 3)
        corner_points = corner_points - ray_origins.unsqueeze(-2)
        # (..., samples_per_image, 2, 3)
        corner_points = corner_points.unsqueeze(-3)
        # (..., samples_per_image, 2, 3)
        z_bounds = corner_points / (ray_directions.unsqueeze(-2) + eps)

        # Since ray directions may be negative we may have to invert lower and upper bounds
        z_near, _ = z_bounds.min(dim=-2)
        z_far, _ = z_bounds.max(dim=-2)

        # Finds the largest interval that satisfies containment in the bounding box limit for all 3 dimensions
        # if z_near > z_max no such interval exists
        z_near, _ = z_near.max(dim=-1)
        z_far, _ = z_far.min(dim=-1)

        # Broadcasts to (..., samples_per_images)
        object_validity = object_validity.unsqueeze(-1)
        _, object_validity = torch.broadcast_tensors(z_far, object_validity)

        # For rays that do not satisfy constraints or refer to objects that are not present put z_near and z_far close
        # to the camera so that they won't interfere with rendering. If they went into another object's bounding box
        # they may set transparency where there is not
        mask = torch.logical_or((z_far <= z_near), (object_validity == False))
        z_far[mask] = 0.0
        z_near[mask] = 0.0

        return z_near, z_far

    def compute_position_distances(self, ray_positions_t: torch.Tensor, ray_directions: torch.Tensor) -> torch.Tensor:
        '''
         Computes the euclidean distance between each position in each ray direction

        :param ray_positions_t: (..., samples_per_image, positions_count) tensor with distance of each sampled points
                                                                          from the focal plane, measured perpendicularly to it
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :return: (..., samples_per_image, positions_count) tensor with euclidean distance between each sample point on
                                                           a ray and its next one. Last distance is posed to infinity
        '''

        # Compute the distance between points on the ray perpendicular to the focal plane
        first_distances = ray_positions_t[..., 1:] - ray_positions_t[..., :-1]

        # Add the distance of the last sample which is infinity
        # The dimension size is always 1 since we add 1 position at the end
        last_distances_size = list(first_distances.size())
        last_distances_size[-1] = 1

        last_distances = torch.ones(last_distances_size, dtype=torch.float32, device=ray_positions_t.device) * 1e10
        distances = torch.cat([first_distances, last_distances], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        distances = distances * torch.linalg.norm(ray_directions[..., None, :], dim=-1)
        return distances

    def compute_alphas(self, raw_alphas: torch.Tensor, position_distances: torch.Tensor, perturb: bool):
        '''
        Computes the alpha values for each position taking into account the distance between samples

        :param raw_alphas: (..., positions_count) tensor with ray alpha values
        :param position_distances: (..., positions_count) tensor with euclidean distance between each sample point on
                                                                             a ray and its next one
        :param perturb: True if noise should be applied to the raw alphas

        :return: (..., positions_count) tensor with alpha values
        '''

        # Regularize network during training (prevents floater artifacts).
        if perturb:
            noise = torch.randn(raw_alphas.size(), device=raw_alphas.device)
            raw_alphas = raw_alphas + noise

        return 1.0 - torch.exp(-F.relu(raw_alphas) * position_distances)

    def compute_weights(self, alphas: torch.Tensor) -> torch.Tensor:
        '''
        Computes the weight of each position, i.e. the probability that light is absorbed at the current position

        :param alphas: (..., positions_count) tensor with alpha values
        :return: (..., positions_count) tensor with weight for each position
        '''

        # Emulates tensorflow exclusive=True cumprod where the first returned element is 1 instead of x_0
        alphas_to_shift = 1. - alphas + 1e-10
        shifted_alphas = torch.cat([torch.ones_like(alphas_to_shift[..., 0:1]), alphas_to_shift[..., :-1]], dim=-1)

        # Probability that light is not absorbed in the previous positions and it is absorbed at the current one
        weights = alphas * torch.cumprod(shifted_alphas, dim=-1)

        return weights

    def copy_tensor_list(self, tensor_list: List[torch.Tensor]):
        tensor_list = [tensor.clone() for tensor in tensor_list]
        return tensor_list

    def fix_all_object_overlaps(self, ray_origins: torch.Tensor, all_raw_alphas: List[torch.Tensor], all_ray_positions_t: List[torch.Tensor],
                                all_ray_positions: List[torch.Tensor], all_ray_displacements: List[torch.Tensor], all_ray_divergences: List[torch.Tensor]):
        '''
        Fixes the portion of ray samples for static objects that fall within the bounding box of a dynamic object to have null contribution
        Content of the original tensors is not altered
        Content of the input lists is not altered
        The returned fixed ray_positions_t are no longer sorted

        :param ray_origins: see compose method
        :param all_raw_alphas: see compose method
        :param all_ray_positions_t: see compose method
        :param all_ray_positions: see compose method
        :param all_ray_displacements: see compose method
        :param all_ray_divergences: see compose method

        :return: fixed and unordered versions of all_raw_alphas, all_ray_positions_t, all_ray_positions, all_ray_displacements, all_ray_divergences
        '''

        static_objects_count = self.object_id_helper.static_objects_count
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        # Makes copies of the tensors to ensure the original ones are not altered
        all_fixed_raw_alphas = self.copy_tensor_list(all_raw_alphas)
        all_fixed_ray_positions_t = self.copy_tensor_list(all_ray_positions_t)
        all_fixed_ray_positions = self.copy_tensor_list(all_ray_positions)
        all_fixed_ray_displacements = self.copy_tensor_list(all_ray_displacements)
        all_fixed_ray_divergences = self.copy_tensor_list(all_ray_divergences)

        flat_ray_origins, initial_dimensions = TensorFolder.flatten(ray_origins, -1)

        # Fixes each static object with each dynamic object
        for static_object_idx in range(static_objects_count):
            # Gets the for the static object
            current_raw_alphas = all_fixed_raw_alphas[static_object_idx]
            current_ray_positions_t = all_fixed_ray_positions_t[static_object_idx]
            current_ray_positions = all_fixed_ray_positions[static_object_idx]
            current_ray_displacements = all_fixed_ray_displacements[static_object_idx]
            current_ray_divergences = all_fixed_ray_divergences[static_object_idx]
            current_original_ray_positions_t = all_ray_positions_t[static_object_idx]  # Will not be altered, so can be taken from original

            # Flattens the tensors
            current_raw_alphas, _ = TensorFolder.flatten(current_raw_alphas, -1)
            current_ray_positions_t, _ = TensorFolder.flatten(current_ray_positions_t, -1)
            current_ray_positions, _ = TensorFolder.flatten(current_ray_positions, -2)
            current_ray_displacements, _ = TensorFolder.flatten(current_ray_displacements, -2)
            current_ray_divergences, _ = TensorFolder.flatten(current_ray_divergences, -1)
            current_original_ray_positions_t, _ = TensorFolder.flatten(current_original_ray_positions_t, -1)

            for offsetted_dynamic_object_idx in range(dynamic_objects_count):
                dynamic_object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(offsetted_dynamic_object_idx)

                # Gets the tensors for the dynamic object and flattens them
                current_overlap_ray_positions_t = all_ray_positions_t[dynamic_object_idx]  # Will not be altered, so can be taken from original
                current_overlap_ray_positions_t, _ = TensorFolder.flatten(current_overlap_ray_positions_t, dimensions=-1)

                # Fixes the overlap between the static and the dynamic object
                self.fix_object_overlap(current_raw_alphas, current_ray_positions_t, current_original_ray_positions_t, current_ray_positions,
                                        current_ray_displacements, current_ray_divergences, flat_ray_origins, current_overlap_ray_positions_t)

            # Folds the tensors
            current_raw_alphas = TensorFolder.fold(current_raw_alphas, initial_dimensions)
            current_ray_positions_t = TensorFolder.fold(current_ray_positions_t, initial_dimensions)
            current_ray_positions = TensorFolder.fold(current_ray_positions, initial_dimensions)
            current_ray_displacements = TensorFolder.fold(current_ray_displacements, initial_dimensions)
            current_ray_divergences = TensorFolder.fold(current_ray_divergences, initial_dimensions)

            # Inserts the results back in the list
            all_fixed_raw_alphas[static_object_idx] = current_raw_alphas
            all_fixed_ray_positions_t[static_object_idx] = current_ray_positions_t
            all_fixed_ray_positions[static_object_idx] = current_ray_positions
            all_fixed_ray_displacements[static_object_idx] = current_ray_displacements
            all_fixed_ray_divergences[static_object_idx] = current_ray_divergences

        return all_fixed_raw_alphas, all_fixed_ray_positions_t, all_fixed_ray_positions, all_fixed_ray_displacements, all_fixed_ray_divergences

    def fix_object_overlap(self, raw_alphas: torch.Tensor, ray_positions_t: torch.Tensor, original_ray_positions_t: torch.Tensor, ray_positions: torch.Tensor,
                                 ray_displacements: torch.Tensor, ray_divergences: torch.Tensor,
                                 ray_origins: torch.Tensor, overlap_ray_positions_t: torch.Tensor):
        '''
        Fixes the portion of object samples that overlap with a set of other samples to have null contribution to the final result
        Poses alphas to negative, positions_t, displacements and divergences to 0, positions to origins (correspond to positions_t = 0)
        Input tensors are fixed in place (raw_alphas, ray_positions_t, ray_positions, ray_displacements, ray_divergences)

        :param raw_alphas: (elements_count, positions_count) tensor with output alphas
        :param ray_positions_t: (elements_count, positions_count) tensors with distance of each value from the focal plane, measured perpendicularly to it. Its values may have been already altered
        :param original_ray_positions_t: (elements_count, positions_count) See ray_positions_t, but its values must never have been fixed for object overlap. That is, Its element along the innermost position must be sorted
        :param ray_positions: (elements_count, positions_count, 3) tensors with ray positions
        :param ray_displacements: (elements_count, positions_count, 3) tensors with ray displacements associated to each position
        :param ray_divergences: (elements_count, positions_count) tensors with divergences for the displacements rayfield at each position
        :param ray_origins: (elements_count, 3) tensor with the origin of each ray
        :param overlap_ray_positions_t: (elements_count, positions_count) tensors with distance of each value from the focal plane, measured perpendicularly to it.
                                                                          All values that correspond to points between its min and max that will be fixed for overlap.
        :return:
        '''

        if len(raw_alphas.size()) != 2:
            raise Exception("Input seem not to have been flattened")

        elements_count = raw_alphas.size(0)
        positions_count = raw_alphas.size(1)

        # Gets first and last elements defining the range for the overlap
        overlap_ray_positions_t = overlap_ray_positions_t[:, (0, positions_count - 1)]
        # Finds the beginning and end indexes for the interval. Uses original ray positions since the ray_positions_t may not be sorted
        overlap_intervals = torch.searchsorted(original_ray_positions_t, overlap_ray_positions_t)

        """flat_indexes_to_set = []
        for interval_idx in range(elements_count):
            current_interval = overlap_intervals_cpu[interval_idx]
            begin_idx = int(current_interval[0])
            end_idx = int(current_interval[1])  # End index not to be set
            for inner_idx in range(begin_idx, end_idx):
                flat_idx = inner_idx + interval_idx * positions_count
                flat_indexes_to_set.append(flat_idx)
                #mask_cpu[interval_idx, inner_idx] = False

        flat_indexes_to_set = torch.as_tensor(flat_indexes_to_set, dtype=torch.int64, device=raw_alphas.device)
        trues = torch.ones_like(flat_indexes_to_set, dtype=torch.bool, device=raw_alphas.device)

        # Inserts trues into the boolean matrix at the location of the indexes
        flat_mask = torch.ones((elements_count * positions_count,), device=raw_alphas.device, dtype=torch.bool)
        flat_mask.scatter_(0, flat_indexes_to_set, trues)

        # Folds the mask
        mask = flat_mask.reshape((elements_count, positions_count))"""


        # Computes the indexes in flat mask that must be set
        # Those are the elements that fall inside the overlapping interval
        overlap_intervals_cpu = overlap_intervals.detach().cpu().numpy()  # Overlaps are used on cpu, so all the tensor is transferred to cpu in the first place
        mask_cpu = np.zeros((elements_count, positions_count), dtype=np.bool)  # Using numpy instead of torch cpu tensors seems slightly faster
        for interval_idx in range(elements_count):
            current_interval = overlap_intervals_cpu[interval_idx]
            begin_idx = int(current_interval[0])
            end_idx = int(current_interval[1])  # End index not to be set
            # If the interval is not empty
            if begin_idx != end_idx:
                mask_cpu[interval_idx, begin_idx:end_idx] = True

        # Puts the mask in the device
        mask = torch.from_numpy(mask_cpu).to(raw_alphas.device)


        """
        # LEGACY CODE, SLOWER DUE DO GPU TENSOR INDEXING
        # Computes the indexes in flat mask that must be set
        # Those are the elements that fall inside the overlapping interval
        flat_indexes_to_set = []
        for interval_idx in range(elements_count):
            current_interval = overlap_intervals[interval_idx]
            begin_idx = current_interval[0].item()
            end_idx = current_interval[1].item()  # End index not to be set
            for inner_idx in range(begin_idx, end_idx):
                flat_idx = inner_idx + interval_idx * positions_count
                flat_indexes_to_set.append(flat_idx)
        flat_indexes_to_set = torch.as_tensor(flat_indexes_to_set, dtype=torch.int64, device=raw_alphas.device)
        trues = torch.ones_like(flat_indexes_to_set, dtype=torch.bool, device=raw_alphas.device)

        # Inserts trues into the boolean matrix at the location of the indexes
        flat_mask.scatter_(0, flat_indexes_to_set, trues)

        # Folds the mask
        mask = flat_mask.reshape((elements_count, positions_count))
        """

        # Makes ray origins the same size as the ray_positions (elements_count, positions_count, 3)
        positional_ray_origins = ray_origins.unsqueeze(1).repeat((1, positions_count, 1))
        # Makes the mask the same size as the ray_positions (elements_count, positions_count, 3)
        positional_mask = mask.unsqueeze(-1).repeat(1, 1, 3)

        # Alphas become an arbitrary negative value
        raw_alphas[mask] = raw_alphas[mask] * 0.0 - 10.0
        # Positions are set to the origin (0)
        ray_positions_t[mask] *= 0.0
        ray_positions[positional_mask] = positional_ray_origins[positional_mask]
        # Displacements and divervences become 0
        ray_displacements[mask] *= 0.0
        ray_divergences[mask] *= 0.0

    def compose(self, ray_origins: torch.Tensor, all_raw_features: List[torch.Tensor], all_raw_alphas: List[torch.Tensor],
                all_ray_positions_t: List[torch.Tensor], all_ray_positions: List[torch.Tensor],
                all_ray_displacements: List[torch.Tensor], all_ray_divergences: List[torch.Tensor]):
        '''
        Compose the results of multiple objects into a single one

        :param ray_origins: list of (..., 3) tensors with ray origins
        :param all_raw_features: list of (..., positions_count_i, output_features_count) tensors with output features
        :param all_raw_alphas: list of (..., positions_count_i) tensors with output alphas
        :param all_ray_positions_t: list of (..., positions_count_i) tensors with distance of each value from the focal plane, measured perpendicularly to it
        :param all_ray_positions: list of (..., positions_count_i, 3) tensors with ray positions
        :param all_ray_displacements: list of (..., positions_count_i, 3) tensors with ray displacements associated to each position
        :param all_ray_divergences: list of (..., positions_count_i) tensors with divergences for the displacements rayfield at each position

        :return: (..., positions_count, output_features_count) tensor with composed output features
                 (..., positions_count) tensor with composed output alphas
                 (..., positions_count) tensor with distance of each value from the focal plane, measured perpendicularly to it
                 (..., positions_count, 3) tensor with ray positions
                 (..., positions_count, 3) tensor with ray displacements associated to each position
                 (..., positions_count) tensor with divergences for the displacements rayfield at each position
        '''

        # Fixes overlap between static and dynamic objects to make null the contribution of static object values that fall within the box of a dynamic object
        if self.config["model"]["fix_object_overlaps"]:
            fix_results = self.fix_all_object_overlaps(ray_origins, all_raw_alphas, all_ray_positions_t, all_ray_positions, all_ray_displacements, all_ray_divergences)
            all_raw_alphas, all_ray_positions_t, all_ray_positions, all_ray_displacements, all_ray_divergences = fix_results

        # Concatenates in a single tensor
        all_raw_features = torch.cat(all_raw_features, dim=-2)
        all_raw_alphas = torch.cat(all_raw_alphas, dim=-1)
        all_ray_positions_t = torch.cat(all_ray_positions_t, dim=-1)
        all_ray_positions = torch.cat(all_ray_positions, dim=-2)
        all_ray_displacements = torch.cat(all_ray_displacements, dim=-2)
        all_ray_divergences = torch.cat(all_ray_divergences, dim=-1)

        # Obtains sorted indices and sorts the other arrays
        all_ray_positions_t, sorting_indices = torch.sort(all_ray_positions_t, dim=-1)

        # Performs indexing according to sort order
        all_raw_alphas = torch.gather(all_raw_alphas, dim=-1, index=sorting_indices)
        all_ray_divergences = torch.gather(all_ray_divergences, dim=-1, index=sorting_indices)
        # Puts the indices in the same dimensions as the features
        sorting_indices_features, _ = torch.broadcast_tensors(sorting_indices.unsqueeze(-1), all_raw_features)
        all_raw_features = torch.gather(all_raw_features, dim=-2, index=sorting_indices_features)
        sorting_indices_positions, _ = torch.broadcast_tensors(sorting_indices.unsqueeze(-1), all_ray_positions)
        all_ray_positions = torch.gather(all_ray_positions, dim=-2, index=sorting_indices_positions)
        all_ray_displacements = torch.gather(all_ray_displacements, dim=-2, index=sorting_indices_positions)

        return all_raw_features, all_raw_alphas, all_ray_positions_t, all_ray_positions, all_ray_displacements, all_ray_divergences

    def visualize_scene(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor,
                        ray_positions: torch.Tensor, bounding_box: BoundingBox, transformation_matrix_w2o: torch.Tensor,
                        to_world_system=True):
        '''
        Visualizes the first scene described by the given tensors

        :param ray_origins: see forward_object
        :param ray_directions: see forward_object
        :param focal_normals: see forward_object
        :param ray_positions: (..., samples_per_image, positions_count, 3) tensor with ray directions
        :param bounding_box: bounding box of the object to visualize
        :param transformation_matrix_w2o: see forward_object
        :param to_world_system: whether the input coordinates are in object system and need to be converted to world system
        :return:
        '''

        from utils.lib_3d.scene_viewer import SceneViewer

        # If coordinates are in object system we translate them to world system
        if to_world_system:
            ray_origins, ray_directions, focal_normals, ray_positions =  RayHelper.transform_ray_positions(ray_origins, ray_directions,
                                                                         focal_normals, ray_positions, transformation_matrix_w2o.inverse())

        # Extracts the values for the first scene
        flat_ray_origins = TensorFolder.flatten(ray_origins, dimensions=-2)[0][0]  # The first indexes the flat tensor, the second indexes the first tensor dimesnion
        flat_ray_directions = TensorFolder.flatten(ray_directions, dimensions=-3)[0][0]
        flat_focal_normals = TensorFolder.flatten(focal_normals, dimensions=-2)[0][0]
        flat_ray_positions = TensorFolder.flatten(ray_positions, dimensions=-4)[0][0]
        flat_transformation_matrix_w2o = TensorFolder.flatten(transformation_matrix_w2o, dimensions=-2)[0][0]

        flat_transformation_matrix_o2w = flat_transformation_matrix_w2o.inverse()
        scene_viewer = SceneViewer()
        scene_viewer.add_bounding_box(bounding_box, flat_transformation_matrix_o2w)
        scene_viewer.add_rays(flat_ray_origins, flat_ray_directions, flat_focal_normals, flat_ray_positions)

        scene_viewer.render_views()

    def forward_object(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor,
                       transformation_matrix_w2o: torch.Tensor, model_coarse: nn.Module, model_fine: nn.Module,
                       style: torch.Tensor, deformation: torch.Tensor, object_in_scene: torch.Tensor, perturb: bool,
                       video_indexes: torch.Tensor=None, canonical_pose: bool=False) -> Dict[str, Tuple[torch.Tensor]]:
        '''

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param focal_normals: (..., 3) tensor with focal normals
        :param transformation_matrix_w2o: (..., 4, 4) transformation matrix from world to object coordinate system
        :param model_coarse: Model to use for obtaning coarse values
        :param model_fine: Model to use to obtain fine values. None if fine network is not to be used
        :param style: (..., style_features_count) features representing style of each object
        :param deformation: (..., deformation_features_count) features representing deformation of each object
        :param object_in_scene: (...) boolean tensor with True if the object is present in the scene
        :param perturb: true if perturbations should be applied during computations
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset. Indexes are inferred if None
        :param canonical_pose: if True renders the object in the canonical pose

        :return: Dictionary with keys "fine" (only if fine network is present) and "coarse" containing each
                 (..., samples_per_image, positions_count, output_features_count) tensor with ray feature outputs
                 (..., samples_per_image, positions_count) tensor with raw alpha values
                 (..., samples_per_image, positions_count) tensor with distance of each value from the focal plane, measured perpendicularly to it
                 (..., samples_per_image, positions_count, 3) tensor with ray position
                 (..., samples_per_image, positions_count, 3) tensor with ray displacements for each position
                 (..., samples_per_image, positions_count) tensor with divergence of the rayfield for each position
                 dictionary with extra outputs
                     (..., sampers_per_image, positions_count, dim)
        '''

        # Obtains the coordinates in the system of the current object
        ray_origins, ray_directions, focal_normals = RayHelper.transform_rays(ray_origins, ray_directions,
                                                                              focal_normals, transformation_matrix_w2o)

        # Computes the z_near and z_far values for the current object
        z_near, z_far = self.compute_raywise_object_z_bounds(ray_origins, ray_directions, model_coarse.bounding_box, object_in_scene)
        z_near = torch.clamp(z_near, min=model_coarse.model_config["z_near_min"], max=model_coarse.model_config["z_far_max"])
        z_far = torch.clamp(z_far, min=model_coarse.model_config["z_near_min"], max=model_coarse.model_config["z_far_max"])

        # Creates the positions for the coarse model
        positions_count_coarse = model_coarse.model_config["positions_count_coarse"]
        coarse_ray_positions, coarse_ray_positions_t = RayHelper.create_ray_positions(ray_origins, ray_directions, z_near,
                                                                                      z_far, positions_count_coarse, perturb)

        if False:
            self.visualize_scene(ray_origins, ray_directions, focal_normals, coarse_ray_positions, model_coarse.bounding_box, transformation_matrix_w2o, to_world_system=True)

        # Adds a dimension for samples_per_image
        style = style.unsqueeze(-2)
        deformation = deformation.unsqueeze(-2)
        if video_indexes is not None:
            video_indexes = video_indexes.unsqueeze(-1)

        # Adds a dimension for the samples per image
        samples_per_image = ray_directions.size(-2)
        expanded_ray_origins = TensorBroadcaster.add_dimension(ray_origins, samples_per_image, -2)

        # Forwards through the coarse model
        coarse_raw_features, coarse_raw_alphas, coarse_ray_displacements, coarse_extra_outputs = model_coarse(coarse_ray_positions, expanded_ray_origins, ray_directions, style, deformation, video_indexes=video_indexes, canonical_pose=canonical_pose)
        # If the object is not in the scene, we set its opacity to 0.
        # No positions should be sampled inside its bounding box, but if the camera is in the origin then points in the model could be sampled
        coarse_raw_alphas[torch.logical_not(object_in_scene)] = model_coarse.empty_space_alpha
        if self.apply_activation:
            coarse_raw_features = torch.sigmoid(coarse_raw_features)

        # Computes coarse distances, alphas and weights
        coarse_position_distances = self.compute_position_distances(coarse_ray_positions_t, ray_directions)
        coarse_alphas = self.compute_alphas(coarse_raw_alphas, coarse_position_distances, perturb)
        coarse_weights = self.compute_weights(coarse_alphas)

        # Computes the divergence of the displacements rayfield
        coarse_ray_divergence = self.compute_approximate_divergence(coarse_ray_positions, coarse_ray_displacements)

        results = {
            "coarse": (coarse_raw_features, coarse_raw_alphas, coarse_ray_positions_t, coarse_ray_positions, coarse_ray_displacements, coarse_ray_divergence, coarse_extra_outputs),
        }

        # Performs the computations for the fine model only if the fine model is to be used
        if model_fine is not None:
            # Computes fine positions
            positions_count_fine = model_fine.model_config["positions_count_fine"]
            fine_ray_positions, fine_ray_positions_t = RayHelper.create_ray_positions_weighted(ray_origins, ray_directions, positions_count_fine, coarse_ray_positions_t, coarse_weights, perturb)
            # Forwards through the fine model
            fine_raw_features, fine_raw_alphas, fine_ray_displacements, fine_extra_outputs = model_fine(fine_ray_positions, expanded_ray_origins, ray_directions, style, deformation, video_indexes=video_indexes, canonical_pose=canonical_pose)
            # If the object is not in the scene, we set its opacity to 0.
            # No positions should be sampled inside its bounding box, but if the camera is in the origin then points in the model could be sampled
            fine_raw_alphas[torch.logical_not(object_in_scene)] = model_fine.empty_space_alpha
            if self.apply_activation:
                fine_raw_features = torch.sigmoid(fine_raw_features)

            fine_ray_divergence = self.compute_approximate_divergence(fine_ray_positions, fine_ray_displacements)

            results["fine"] = (fine_raw_features, fine_raw_alphas, fine_ray_positions_t, fine_ray_positions, fine_ray_displacements, fine_ray_divergence, fine_extra_outputs)

        return results

    def compute_approximate_divergence(self, ray_positions: torch.Tensor, ray_displacements: torch.Tensor):
        '''
        From FFJORD codebase. Computes the approximate divergence of the flow field
        :param ray_positions: (..., 3) tensor with ray positions
        :param ray_displacements: (..., 3) tensor with the displacement for each position

        :return: (...) tensor with divergence for each element, 0 if not in training mode
        '''

        # No computing graph if evaluating, so no gradient can be computed
        # Do not compute divergence also if parameters were not used to compute the ray displacements
        if not self.training or not ray_displacements.requires_grad:
            return torch.zeros_like(ray_displacements[..., 0])

        # avoids explicitly computing the Jacobian
        e = torch.randn_like(ray_displacements, device=ray_displacements.get_device())
        e_dydx = torch.autograd.grad(ray_displacements, ray_positions, e, create_graph=True)[0]
        e_dydx_e = e_dydx * e
        approx_tr_dydx = e_dydx_e.sum(dim=-1)
        return approx_tr_dydx

    def compute_expected_positions(self, ray_positions: torch.Tensor, ray_displacements: torch.Tensor, weights: torch.Tensor, eps=1e-8) -> torch.Tensor:
        '''
        Computes the expected position of the first surface encountered by a ray

        :param ray_positions: (..., positions_count, 3) tensor with ray positions
        :param ray_displacements: (..., positions_count, 3) tensor with ray displacements
        :param weights: (..., positions_count) tensor with weights associated to each sample
        :return: (..., 3) tensor with with the expected position of the first surface encountered by the ray
        '''

        # Do not alter the weights to reduce losses on expected positions
        weights = weights.detach()

        # Computes the expected position of the first surface encountered by the ray
        bent_ray_positions = ray_positions + ray_displacements
        # Computes the expected position as the weighted average of the position over the ray
        expected_positions = (bent_ray_positions * weights.unsqueeze(-1)).sum(dim=-2)
        expected_positions = expected_positions / (weights.unsqueeze(-1).sum(dim=-2) + eps)

        return expected_positions

    def forward_expected_positions(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor, transformation_matrix_w2o: torch.Tensor,
                style: torch.Tensor, deformation: torch.Tensor, object_in_scene: torch.Tensor, object_id: int, perturb: bool, video_indexes: torch.Tensor=None, canonical_pose: bool=False) -> Dict:
        '''
        Computes output values associated to each input ray by composing the underlying objects

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param focal_normals: (..., 3) tensor with focal normals
        :param transformation_matrix_w2o: (..., 4, 4) transformation matrix from world to object coordinate system
        :param style: (..., style_features_count) features representing style of each object
        :param deformation: (..., deformation_features_count) features representing deformation of each object
        :param object_in_scene: (...) boolean tensor with True if the object is present in the scene
        :param object_id: id of the object
        :param perturb: true if perturbations should be applied during computations
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset. Indexes are inferred if None

        :return: Dictionary with the following fields

                 Tuple of:
                 (..., samples_per_image, 3) tensor with the expected position of the first surface encountered by each ray
                 (..., samples_per_image) tensor with opacity for the object at each ray

                 Dictionary is divided into coarse and fine results
        '''

        # Id of the model associated to the current object instance
        object_model_idx = self.object_id_helper.model_idx_by_object_idx(object_id)

        model_coarse = self.object_models_coarse[object_model_idx]
        model_fine = self.object_models_fine[object_model_idx]

        # Obtains the coordinates in the system of the current object
        ray_origins, ray_directions, focal_normals = RayHelper.transform_rays(ray_origins, ray_directions, focal_normals, transformation_matrix_w2o)

        # Computes the z_near and z_far values for the current object
        z_near, z_far = self.compute_raywise_object_z_bounds(ray_origins, ray_directions, model_coarse.bounding_box, object_in_scene)
        z_near = torch.clamp(z_near, min=model_coarse.model_config["z_near_min"], max=model_coarse.model_config["z_far_max"])
        z_far = torch.clamp(z_far, min=model_coarse.model_config["z_near_min"], max=model_coarse.model_config["z_far_max"])

        # Creates the positions for the coarse model
        positions_count_coarse = model_coarse.model_config["positions_count_coarse"]
        coarse_ray_positions, coarse_ray_positions_t = RayHelper.create_ray_positions(ray_origins, ray_directions, z_near, z_far, positions_count_coarse, perturb)

        # Adds a dimension for samples_per_image
        style = style.unsqueeze(-2)
        deformation = deformation.unsqueeze(-2)
        if video_indexes is not None:
            video_indexes = video_indexes.unsqueeze(-1)

        # Adds a dimension for the samples per image
        samples_per_image = ray_directions.size(-2)
        expanded_ray_origins = TensorBroadcaster.add_dimension(ray_origins, samples_per_image, -2)

        # Forwards through the coarse model
        coarse_raw_features, coarse_raw_alphas, coarse_ray_displacements, coarse_extra_outputs = model_coarse(coarse_ray_positions, expanded_ray_origins, ray_directions, style, deformation, video_indexes=video_indexes, canonical_pose=canonical_pose)
        # If the object is not in the scene, we set its opacity to 0.
        # No positions should be sampled inside its bounding box, but if the camera is in the origin then points in the model could be sampled
        coarse_raw_alphas[torch.logical_not(object_in_scene)] = model_coarse.empty_space_alpha
        if self.apply_activation:
            coarse_raw_features = torch.sigmoid(coarse_raw_features)

        # Computes coarse distances, alphas and weights
        coarse_position_distances = self.compute_position_distances(coarse_ray_positions_t, ray_directions)
        coarse_alphas = self.compute_alphas(coarse_raw_alphas, coarse_position_distances, perturb)
        coarse_weights = self.compute_weights(coarse_alphas)

        coarse_opacity = coarse_weights.sum(dim=-1)
        # Computes the expected position of the first surfare encountered by the ray
        coarse_expected_positions = self.compute_expected_positions(coarse_ray_positions, coarse_ray_displacements, coarse_weights)

        results = {
            "coarse": (coarse_expected_positions, coarse_opacity),
        }

        # Performs the computations for the fine model only if the fine model is to be used
        if model_fine is not None:
            # Computes fine positions
            positions_count_fine = model_fine.model_config["positions_count_fine"]
            fine_ray_positions, fine_ray_positions_t = RayHelper.create_ray_positions_weighted(ray_origins, ray_directions, positions_count_fine, coarse_ray_positions_t, coarse_weights, perturb)
            # Forwards through the fine model
            fine_raw_features, fine_raw_alphas, fine_ray_displacements, fine_extra_outputs = model_fine(fine_ray_positions, expanded_ray_origins, ray_directions, style, deformation, video_indexes=video_indexes, canonical_pose=canonical_pose)
            # If the object is not in the scene, we set its opacity to 0.
            # No positions should be sampled inside its bounding box, but if the camera is in the origin then points in the model could be sampled
            fine_raw_alphas[torch.logical_not(object_in_scene)] = model_fine.empty_space_alpha
            if self.apply_activation:
                fine_raw_features = torch.sigmoid(fine_raw_features)

            # Computes coarse distances, alphas and weights
            fine_position_distances = self.compute_position_distances(fine_ray_positions_t, ray_directions)
            fine_alphas = self.compute_alphas(fine_raw_alphas, fine_position_distances, perturb)
            fine_weights = self.compute_weights(fine_alphas)

            fine_opacity = fine_weights.sum(dim=-1)
            # Computes the expected position of the first surface encountered by the ray
            fine_expected_positions = self.compute_expected_positions(fine_ray_positions, fine_ray_displacements, fine_weights)

            results["fine"] = (fine_expected_positions, fine_opacity)

        return results

    def integrate(self, raw_features: torch.Tensor, raw_alphas: torch.Tensor, ray_directions: torch.Tensor,
                  ray_positions_t: torch.Tensor, ray_positions: torch.Tensor, ray_displacements: torch.Tensor,
                  ray_divergences: torch.Tensor, perturb: bool) -> Dict[str, torch.Tensor]:
        '''
        Integrates the values present in each ray

        :param raw_features: (..., positions_count, output_features_count) tensor with ray feature outputs
        :param raw_alphas: (..., positions_count) tensor with raw alpha values
        :param ray_directions: (..., 3) tensor with ray directions
        :param ray_positions_t: (..., positions_count) tensor with distance of each value from the focal plane, measured perpendicularly to it
        :param ray_positions: (..., positions_count, 3) tensor with ray positions
        :param ray_displacements: (..., positions_count, 3) tensor with ray displacments for each position
        :param ray_divergences: (..., positions_count,) tensor with displacements rayfield divergences at each position
        :param perturb: true if perturbations should be applied during computations
        :return: Dictionary with the following fields
                 (..., output_features_count) tensor of integrated features along the input rays
                 (...,) tensor with the opacity of each ray. Values lower than 1 indicate background visibility
                 (...,) tensor with the accumulation of weights along each ray
                 (..., positions_count) tensor of weights for each ray
                 (...,) tensor of depth for each ray
                 (...,) tensor of disparity (inverse depth) for each ray
                 (...,) tensor of integrated magnitude of ray displacements
                 (...,) tensor of alpha-weighted divergence of the ray displacements
        '''

        # Computes distances, alphas and weights
        position_distances = self.compute_position_distances(ray_positions_t, ray_directions)
        alphas = self.compute_alphas(raw_alphas, position_distances, perturb)
        weights = self.compute_weights(alphas)

        # Integrates the features along the positions count dimension
        integrated_features = torch.sum(weights.unsqueeze(-1) * raw_features, dim=-2)

        # Estimated depth map is expected distance. Alternatively it can be computed as the median value
        depth = torch.sum(weights * ray_positions_t, dim=-1)

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        # Values lower than 1 indicate visibility of the background
        opacity = torch.sum(weights, dim=-1)

        # Disparity map is inverse depth.
        disparity = 1. / torch.clamp(depth / opacity, min=1e-10)

        # Computes divergence for each ray. Uses mean instead of sum as in the original implementation
        divergence = torch.abs(ray_divergences)
        integrated_divergence = torch.mean(alphas.detach() * divergence, dim=-1)

        # Computes displacements magnitude. Uses mean instead of sum as in the original implementation
        integrated_displacements_magnitude = torch.mean(weights.detach() * torch.norm(ray_displacements, dim=-1), dim=-1)

        results = {
            "integrated_features": integrated_features,
            "opacity": opacity,
            "weights": weights,
            "depth": depth,
            "disparity": disparity,
            "integrated_displacements_magnitude": integrated_displacements_magnitude,
            "integrated_divergence": integrated_divergence
        }

        return results

    def forward(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor, transformation_matrix_w2o: torch.Tensor,
                style: torch.Tensor, deformation: torch.Tensor, object_in_scene: torch.Tensor, perturb: bool, video_indexes: torch.Tensor=None, canonical_pose: bool=False) -> Dict:
        '''
        Computes output values associated to each input ray by composing the underlying objects

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param focal_normals: (..., 3) tensor with focal normals
        :param transformation_matrix_w2o: (..., 4, 4, objects_count) transformation matrix from world to object coordinate system
        :param style: (..., style_features_count, objects_count) features representing style of each object
        :param deformation: (..., deformation_features_count, objects_count) features representing deformation of each object
        :param object_in_scene: (..., objects_count) boolean tensor with True if the object is present in the scene
        :param perturb: true if perturbations should be applied during computations
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset. Indexes are inferred if None
        :param canonical_pose: if True renders the object in the canonical pose

        :return: Dictionary with the following fields
                 (..., samples_per_image, output_features_count) tensor of integrated features along the input rays
                 (..., samples_per_image,) tensor with the accumulation of weights along each ray
                 (..., samples_per_image, positions_count) tensor of weights for each ray
                 (..., samples_per_image,) tensor of depth for each ray
                 (..., samples_per_image,) tensor of disparity (inverse depth) for each ray
                 dictionary of
                     (..., samples_per_image, positions_count, dim) tensors representing extra outputs
                 Dictionary is divided into coarse and fine results
        '''

        samples_per_image = ray_directions.size(-2)

        objects_count = self.object_id_helper.objects_count
        # Checks that the correct number of transformation matrices is supplied
        if transformation_matrix_w2o.size(-1) != objects_count:
            raise Exception(f"Transformation matrix must specifies transformations for"
                            f"({transformation_matrix_w2o.size(-1)}) objects instead of ({objects_count})")

        # Computes values for each object along the rays
        all_object_results = []
        for object_idx in range(objects_count):

            # Id of the model associated to the current object instance
            object_model_idx = self.object_id_helper.model_idx_by_object_idx(object_idx)

            current_transformation_matrix_w2o = transformation_matrix_w2o[..., object_idx]

            current_model_coarse = self.object_models_coarse[object_model_idx]
            current_model_fine = self.object_models_fine[object_model_idx]

            current_object_style = style[..., object_idx]
            current_object_deformation = deformation[..., object_idx]

            current_object_in_scene = object_in_scene[..., object_idx]

            current_results = self.forward_object(ray_origins, ray_directions, focal_normals, current_transformation_matrix_w2o,
                                                  current_model_coarse, current_model_fine, current_object_style, current_object_deformation,
                                                  current_object_in_scene, perturb, video_indexes=video_indexes, canonical_pose=canonical_pose)

            all_object_results.append(current_results)

        # Adds the samples_per_image dimension to ray_origins
        expanded_ray_origins = ray_origins.unsqueeze(-2)
        expanded_ray_origins = expanded_ray_origins.repeat([1] * (len(expanded_ray_origins.size()) - 2) + [samples_per_image, 1])

        results = {}
        # Composes the results for the different objects
        # Iterates over coarse and fine
        for model_type in all_object_results[0].keys():
            results[model_type] = {}

            all_raw_features = []
            all_raw_alphas = []
            all_ray_positions_t = []
            all_ray_positions = []
            all_ray_displacements = []
            all_ray_divergences = []

            # Concatenates the results of each object
            for object_idx, current_result in enumerate(all_object_results):

                current_raw_features = current_result[model_type][0]
                current_ray_alphas = current_result[model_type][1]
                current_ray_positions_t = current_result[model_type][2]
                current_ray_positions = current_result[model_type][3]
                current_ray_displacements = current_result[model_type][4]
                current_ray_divergences = current_result[model_type][5]
                current_extra_outputs = current_result[model_type][6]

                all_raw_features.append(current_raw_features)
                all_raw_alphas.append(current_ray_alphas)
                all_ray_positions_t.append(current_ray_positions_t)
                all_ray_positions.append(current_ray_positions)
                all_ray_displacements.append(current_ray_displacements)
                all_ray_divergences.append(current_ray_divergences)

                # Integrates the current object
                current_object_integration_results = self.integrate(current_raw_features, current_ray_alphas, ray_directions, current_ray_positions_t, current_ray_positions, current_ray_displacements, current_ray_divergences, perturb)
                results[model_type][f"object_{object_idx}"] = current_object_integration_results
                results[model_type][f"object_{object_idx}"]["extra_outputs"] = current_extra_outputs

            # Composes and integrates the global scene
            composed_raw_features, composed_raw_alphas, composed_ray_positions_t, composed_ray_positions, composed_ray_displacements, composed_ray_divergences = self.compose(expanded_ray_origins, all_raw_features, all_raw_alphas, all_ray_positions_t, all_ray_positions, all_ray_displacements, all_ray_divergences)
            current_integration_results = self.integrate(composed_raw_features, composed_raw_alphas, ray_directions, composed_ray_positions_t, composed_ray_positions, composed_ray_displacements, composed_ray_divergences, perturb)
            results[model_type]["global"] = current_integration_results

        # Tensor for fixing pytorch backward hook bug. Multiple dimenisons to allow concatenation in subsequent steps in code
        results["pytorch_hook"] = torch.zeros((1, 1, 1, 1, 1, 1, 1, 1, 1), device=composed_raw_features.device)

        return results


if __name__ == "__main__":

    batch_size = 2
    observations_count = 4
    cameras_count = 5
    height = 50
    width = 100
    focal = 100
    samples_per_image = 10
    positions_count = 91

    perturb = True

    config_path = "configs/01_moving_box_3d_v2_static.yaml"

    configuration = Configuration(config_path)
    # configuration.check_config()
    # configuration.create_directory_structure()

    config = configuration.get_config()

    observations = torch.zeros((batch_size, observations_count, cameras_count, 3, height, width)).cuda()

    ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays([batch_size, observations_count, cameras_count], height, width, focal)
    sampled_directions, sampled_observations = RayHelper.sample_rays(ray_directions, observations, samples_per_image)

    transformation = PoseParameters([0, 0, 0], [0, 0, 0])
    transformation_matrix = transformation.as_homogeneous_matrix_torch().cuda()
    transformation_matrix = transformation_matrix.unsqueeze(0).unsqueeze(0).repeat([batch_size, observations_count, cameras_count, 1, 1]).unsqueeze(-1)


    object_composer = ObjectComposer(config).cuda()
    results = object_composer(ray_origins, sampled_directions, focal_normals, transformation_matrix, True)

    print(results)
    pass