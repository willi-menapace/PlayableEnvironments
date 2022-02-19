import importlib
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.camera_parameters_storage import CameraParametersStorage
from model.object_composer import ObjectComposer
from model.utils.object_ids_helper import ObjectIDsHelper
from utils.lib_3d.bounding_box import BoundingBox
from utils.lib_3d.pose_parameters import PoseParameters
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_batchifier import TensorBatchifier
from utils.tensor_folder import TensorFolder
from utils.torch_time_meter import TorchTimeMeter


class EnvironmentModel(nn.Module):

    def __init__(self, config):
        '''
        Initializes the environment model

        :param config: the configuration file
        '''
        super(EnvironmentModel, self).__init__()

        self.config = config
        self.focal_length_multiplier = config["data"]["focal_length_multiplier"]

        # Whether to use the weighted ray sampling strategy
        self.use_weighted_sampling = config["model"]["use_weighted_sampling"]
        self.sampling_weights = config["model"]["sampling_weights"]

        self.enable_camera_parameters_offsets = self.config["model"]["enable_camera_parameters_offsets"]
        self.training_cameras_count = len(self.config["training"]["batching"]["allowed_cameras"])

        self.camera_parameters_offsets = CameraParametersStorage(self.config["model"]["camera_parameters_memory_size"], self.training_cameras_count)

        # Creates the object composer
        self.object_composer = ObjectComposer(config)

        # Creates parameters encoders for each object
        object_encoder_models = self.create_object_parameters_encoders()
        self.object_parameters_encoders = nn.ModuleList(object_encoder_models)

        # Creates style encoders for each object
        object_encoder_models = self.create_object_encoders()
        self.object_encoders = nn.ModuleList(object_encoder_models)

        # If the image decoder is to be used, instantiate it along with the grid sampler
        self.use_image_decoder = "image_decoder" in self.config["model"]
        if self.use_image_decoder:
            self.image_decoder = self.create_image_decoder()
            self.grid_sampler = self.create_grid_sampler()

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        self.time_meter = TorchTimeMeter(name="environment_model_perf", mode="sum", enabled=False)

        # Sets the current step to be 0. Will be updated by calls of set_step
        self.current_step = 0


    def get_object_encoder_parameters(self):
        return self.object_encoders.parameters()

    def get_camera_offsets_parameters(self):
        return self.camera_parameters_offsets.parameters()

    def get_main_parameters(self, additional_excluded_parameters=None):
        '''
        Gets all parameters that are not part of the encoders or of the camera offsets
        :param additional_excluded_parameters: set of additional parameter names to exclude. For override purposes
        :return:
        '''

        selected_parameters = []
        excluded_parameters_names = set(["object_encoders."+name for name, param in self.object_encoders.named_parameters()] +
                                        ["camera_parameters_offsets."+name for name, param in self.camera_parameters_offsets.named_parameters()])

        if additional_excluded_parameters is not None:
            excluded_parameters_names = excluded_parameters_names.union(additional_excluded_parameters)

        for current_name, current_parameter in self.named_parameters():
            if current_name not in excluded_parameters_names:
                selected_parameters.append(current_parameter)

        return selected_parameters

    def create_object_parameters_encoders(self) -> List[nn.Module]:
        '''
        Creates parameters encoder model for each object
        :return: list of created models
        '''

        object_models = []
        # Creates the model for each object as specified in the configuration
        for current_object_config in self.config["model"]["object_parameters_encoder"]:
            model_class = current_object_config["architecture"]

            current_model = getattr(importlib.import_module(model_class), 'model')(self.config, current_object_config)
            object_models.append(current_model)

        return object_models

    def create_object_encoders(self) -> List[nn.Module]:
        '''
        Creates style encoder model for each object
        :return: list of created models
        '''

        object_models = []
        # Creates the model for each object as specified in the configuration
        for current_object_config in self.config["model"]["object_encoders"]:
            model_class = current_object_config["architecture"]

            current_model = getattr(importlib.import_module(model_class), 'model')(self.config, current_object_config)
            object_models.append(current_model)

        return object_models

    def create_image_decoder(self) -> nn.Module:
        '''
        Creates the image decoder
        :return: the image decoder
        '''

        model_config = self.config["model"]["image_decoder"]
        model_class = model_config["architecture"]

        # Creates the model
        model = getattr(importlib.import_module(model_class), 'model')(self.config, model_config)
        return model

    def create_grid_sampler(self) -> nn.Module:
        '''
        Creates the grid sampler
        :return: the grid sampler
        '''

        model_config = self.config["model"]["grid_sampler"]
        model_class = model_config["architecture"]

        # Creates the model
        model = getattr(importlib.import_module(model_class), 'model')(self.config, model_config)
        return model

    def set_step(self, current_step: int):
        '''
        Sets the current step to the specified value
        :param current_step:
        :return:
        '''

        # Records the current step
        self.current_step = current_step

        # Sets the step in all object models
        self.object_composer.set_step(current_step)

    def compute_rotation_translation_o2w(self, observations: torch.Tensor, transformation_c2w: PoseParameters, camera_rotations: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor):
        '''
        Computes the rotation and translation parameters from object to world for each object in the scene

        :param observations: (..., cameras_count, 3, height, width) tensor with observations
        :param transformation_c2w: (..., cameras_count, 4, 4) parameters with transformation camera to world
        :param camera_rotations: (..., cameras_count, 3) tensor with camera rotations
        :param focals: (..., cameras_count) tensor with focal length associated to each camera
        :param bounding_boxes: (..., cameras_count, 4, dynamic_objects_count) tensor with normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., cameras_count, dynamic_objects_count) tensor with True is the corresponding dynamic object instance is present in the scene
        :return: (..., 3, objects_count) tensor with rotation parameters from object to world for each object
                 (..., 3, objects_count) tensor with translation parameters from object to world for each object
        '''

        # Gets the transformation matrix
        transformation_matrix_w2c = transformation_c2w.get_inverse_homogeneous_matrix()

        object_rotation_parameters_o2w = []
        object_translation_parameters_o2w = []
        # Computes the parameters for each object instance of each object category
        for object_model_idx in range(self.object_id_helper.object_models_count):
            current_object_parameters_encoder = self.object_parameters_encoders[object_model_idx]

            # If the current object category is static
            if self.object_id_helper.is_static(object_model_idx):
                current_object_rotation_parameters_o2w, current_object_translation_parameters_o2w = current_object_parameters_encoder(observations)
            # If the current object category is dynamic
            else:
                # Selects the bounding boxes for the current object
                start_object_idx, end_object_idx = self.object_id_helper.dynamic_object_idx_range_by_model_idx(object_model_idx)
                current_bounding_boxes = bounding_boxes[..., start_object_idx:end_object_idx]
                current_bounding_boxes_validity = bounding_boxes_validity[..., start_object_idx:end_object_idx]
                # Obtains translation parameters for each object in the scene
                current_object_rotation_parameters_o2w, current_object_translation_parameters_o2w = current_object_parameters_encoder(observations, transformation_matrix_w2c, camera_rotations, focals, current_bounding_boxes, current_bounding_boxes_validity)
            object_rotation_parameters_o2w.append(current_object_rotation_parameters_o2w)
            object_translation_parameters_o2w.append(current_object_translation_parameters_o2w)
        # Concatenates the parameters for all objects along the object dimension
        object_rotation_parameters_o2w = torch.cat(object_rotation_parameters_o2w, dim=-1)
        object_translation_parameters_o2w = torch.cat(object_translation_parameters_o2w, dim=-1)

        return object_rotation_parameters_o2w, object_translation_parameters_o2w

    def compute_transformation_matrix_w2o_o2w(self, object_rotation_parameters_o2w: torch.Tensor, object_translation_parameters_o2w: torch.Tensor):
        '''
        Computes the transformation matrices from world to object for each object in the scene

        :param object_rotation_parameters_o2w: (..., 3, objects_count) tensor with rotation parameters from object to world for each object
        :param object_translation_parameters_o2w: (..., 3, objects_count) tensor with translation parameters from object to world for each object
        :return: (..., cameras_count, 4, 4, objects_count) tensor with transformation matrices from world to object coordinates for each object
                 (..., cameras_count, 4, 4, objects_count) tensor with transformation matrices from object to world coordinates for each object
        '''

        all_transformation_matrices_w2o = []
        all_transformation_matrices_o2w = []
        objects_count = self.object_id_helper.objects_count
        for object_idx in range(objects_count):
            # For each object obtains its w2o transformation matrix
            current_pose_parameters = PoseParameters(object_rotation_parameters_o2w[..., object_idx], object_translation_parameters_o2w[..., object_idx])
            current_transformation_matrix_w2o = current_pose_parameters.get_inverse_homogeneous_matrix()
            current_transformation_matrix_o2w = current_pose_parameters.as_homogeneous_matrix_torch()
            all_transformation_matrices_w2o.append(current_transformation_matrix_w2o)
            all_transformation_matrices_o2w.append(current_transformation_matrix_o2w)

        all_transformation_matrices_w2o = torch.stack(all_transformation_matrices_w2o, dim=-1)
        all_transformation_matrices_w2o = all_transformation_matrices_w2o.unsqueeze(-4)  # Adds a dimension for the cameras count
        all_transformation_matrices_o2w = torch.stack(all_transformation_matrices_o2w, dim=-1)
        all_transformation_matrices_o2w = all_transformation_matrices_o2w.unsqueeze(-4)  # Adds a dimension for the cameras count

        return all_transformation_matrices_w2o, all_transformation_matrices_o2w

    def compute_object_bounding_boxes(self, transformation_matrix_o2w: torch.Tensor, transformation_matrix_w2c: torch.Tensor,
                                      focals: torch.Tensor, height: int, width: int) -> torch.Tensor:
        '''
        For each object, compute its bounding boxes in the image plane of each camera

        :param transformation_matrix_o2w: (..., [cameras_count,] 4, 4, objects_count) tensor with transformation matrix for each object from object to world coordinates
        :param transformation_matrix_w2c: (..., cameras_count, 4, 4) tensor with transformation matrix for each object from world to each
                                                                     camera coordinates
        :param focals: (..., cameras_count) focal length associated to each camera
        :param height: height in pixels of the image plane
        :param width: width in pixels of the image plane
        :return: (..., cameras_count, 4, objects_count) tensor with left, top, right, bottom coordinates for bounding boxes of
                                                        each object in the image plane of each camera. Coordinates refer to
                                                        the top left corner of the image plane and are normalized in [0, 1]
                 (..., cameras_count, bounding_box_points, 2, objects_count) tensor with x, y coordinates for projected 3d bounding box points of
                                                                             each object in the image plane of each camera. Coordinates refer to
                                                                             the top left corner of the image plane and are normalized in [0, 1]
        '''

        # If the optional camera dimension is present in transformation_matrix_o2w, we eliminate it because
        # o2w transformations are independent of cameras
        if len(transformation_matrix_o2w.size()) > len(transformation_matrix_w2c.size()):
            transformation_matrix_o2w = transformation_matrix_o2w[..., 0, :, :, :]

        all_object_bounding_boxes = []
        all_object_projected_bounding_box_points = []
        # Finds the bounding box of each object instance
        objects_count = self.object_id_helper.objects_count
        for object_idx in range(objects_count):

            # Id of the model associated to the current object instance
            object_model_idx = self.object_id_helper.model_idx_by_object_idx(object_idx)

            # Computes transformation matrices for the current object
            current_transformation_matrix_o2w = transformation_matrix_o2w[..., object_idx]

            # Computes the points of the current bounding box
            current_bounding_box = self.object_composer.object_models_coarse[object_model_idx].bounding_box
            current_object_points = current_bounding_box.get_edge_points()

            # Creates a dimension for the number of points in each bounding box
            current_transformation_matrix_o2w = current_transformation_matrix_o2w.unsqueeze(-3)
            # Transforms the points into world coordinates
            current_world_points = RayHelper.transform_points(current_object_points, current_transformation_matrix_o2w)
            # Creates a dimension for the number of cameras
            current_world_points = current_world_points.unsqueeze(-3)
            # Creates a dimension for the number of points in each bounding box
            current_transformation_matrix_w2c = transformation_matrix_w2c.unsqueeze(-3)
            current_camera_points = RayHelper.transform_points(current_world_points, current_transformation_matrix_w2c)

            # Adds dimensions to account for the 8 and 3 dimensions of the bounding box points
            unsqueezed_focals = focals.unsqueeze(-1).unsqueeze(-1)

            # Projects the points on the image plane. - Accounts for the camera pointing in the -z dimension
            # (..., cameras_count, 8, 3)
            projected_points = -current_camera_points[..., :2] / current_camera_points[..., 2:3] * unsqueezed_focals
            projected_points[..., 1] *= -1  # y coordinates grow going down in the image plane due to matrix indexing

            # Patches points behind the camera so that they won't be taken into account when producing the bounding box
            # WARNING: Assumes at least one point is in front of the camera, otherwise the top may be lower than bottom
            # and left may be rightmost than right
            patch_mask = current_camera_points[..., 2:3] > 0
            patch_mask = torch.cat([patch_mask, patch_mask], dim=-1)
            projected_points_patched_max = projected_points.clone()
            projected_points_patched_max[patch_mask] = 1e20
            projected_points_patched_min = projected_points.clone()
            projected_points_patched_min[patch_mask] = -1e20

            # Computes bounding boxes positions
            left = torch.min(projected_points_patched_max[..., 0], dim=-1)[0]
            right = torch.max(projected_points_patched_min[..., 0], dim=-1)[0]
            top = torch.min(projected_points_patched_max[..., 1], dim=-1)[0]
            bottom = torch.max(projected_points_patched_min[..., 1], dim=-1)[0]

            current_bounding_boxes = torch.stack([left, top, right, bottom], dim=-1)
            all_object_bounding_boxes.append(current_bounding_boxes)
            all_object_projected_bounding_box_points.append(projected_points)

        all_object_bounding_boxes = torch.stack(all_object_bounding_boxes, dim=-1)
        all_object_projected_bounding_box_points = torch.stack(all_object_projected_bounding_box_points, dim=-1)

        # Transforms the coordinate reference point from the image plane center to the top left center and normalizes
        all_object_bounding_boxes[..., 0, :] = (all_object_bounding_boxes[..., 0, :] + (width / 2)) / width
        all_object_bounding_boxes[..., 2, :] = (all_object_bounding_boxes[..., 2, :] + (width / 2)) / width
        all_object_bounding_boxes[..., 1, :] = (all_object_bounding_boxes[..., 1, :] + (height / 2)) / height
        all_object_bounding_boxes[..., 3, :] = (all_object_bounding_boxes[..., 3, :] + (height / 2)) / height

        all_object_projected_bounding_box_points[..., 0, :] = (all_object_projected_bounding_box_points[..., 0, :] + (width / 2)) / width
        all_object_projected_bounding_box_points[..., 1, :] = (all_object_projected_bounding_box_points[..., 1, :] + (height / 2)) / height

        # Clamps values outsize the image plane
        all_object_bounding_boxes = torch.clamp(all_object_bounding_boxes, min=0.0, max=1.0)
        all_object_projected_bounding_box_points = torch.clamp(all_object_projected_bounding_box_points, min=0.0, max=1.0)
        return all_object_bounding_boxes, all_object_projected_bounding_box_points

    def compute_object_axes_projection(self, transformation_matrix_o2w: torch.Tensor, transformation_matrix_w2c: torch.Tensor, focals: torch.Tensor, height: int, width: int) -> torch.Tensor:
        '''
        For each object, compute its axes in the image plane of each camera

        :param transformation_matrix_o2w: (..., [cameras_count,] 4, 4, objects_count) tensor with transformation matrix for each object from object to world coordinates
        :param transformation_matrix_w2c: (..., cameras_count, 4, 4) tensor with transformation matrix for each object from world to each
                                                                     camera coordinates
        :param focals: (..., cameras_count) focal length associated to each camera
        :param height: height in pixels of the image plane
        :param width: width in pixels of the image plane
        :return: (..., cameras_count, 4, 2, objects_count) tensor with x, y coordinates for projected 3d points origin, x_axis, y_axis, z_axis of
                                                                             each object in the image plane of each camera. Coordinates refer to
                                                                             the top left corner of the image plane and are normalized in [0, 1]
        '''

        # If the optional camera dimension is present in transformation_matrix_o2w, we eliminate it because
        # o2w transformations are independent of cameras
        if len(transformation_matrix_o2w.size()) > len(transformation_matrix_w2c.size()):
            transformation_matrix_o2w = transformation_matrix_o2w[..., 0, :, :, :]

        all_object_projected_bounding_box_points = []
        # Finds the bounding box of each object instance
        objects_count = self.object_id_helper.objects_count
        for object_idx in range(objects_count):

            # Id of the model associated to the current object instance
            object_model_idx = self.object_id_helper.model_idx_by_object_idx(object_idx)

            # Computes transformation matrices for the current object
            current_transformation_matrix_o2w = transformation_matrix_o2w[..., object_idx]

            # Computes the points to be projected
            current_object_points = torch.tensor([
                (0.0, 0.0, 0.0),  # Origin
                (1.0, 0.0, 0.0),  # x-axis
                (0.0, 1.0, 0.0),  # y-axis
                (0.0, 0.0, 1.0),  # z-axis
            ], device=transformation_matrix_o2w.device)

            # Creates a dimension for the number of points in each bounding box
            current_transformation_matrix_o2w = current_transformation_matrix_o2w.unsqueeze(-3)
            # Transforms the points into world coordinates
            current_world_points = RayHelper.transform_points(current_object_points, current_transformation_matrix_o2w)
            # Creates a dimension for the number of cameras
            current_world_points = current_world_points.unsqueeze(-3)
            # Creates a dimension for the number of points in each bounding box
            current_transformation_matrix_w2c = transformation_matrix_w2c.unsqueeze(-3)
            current_camera_points = RayHelper.transform_points(current_world_points, current_transformation_matrix_w2c)

            # Adds dimensions to account for the 8 and 3 dimensions of the bounding box points
            unsqueezed_focals = focals.unsqueeze(-1).unsqueeze(-1)

            # Projects the points on the image plane. - Accounts for the camera pointing in the -z dimension
            # (..., cameras_count, 8, 3)
            projected_points = -current_camera_points[..., :2] / current_camera_points[..., 2:3] * unsqueezed_focals
            projected_points[..., 1] *= -1  # y coordinates grow going down in the image plane due to matrix indexing

            # Patches points behind the camera so that they won't be taken into account when producing the bounding box
            # WARNING: Assumes at least one point is in front of the camera, otherwise the top may be lower than bottom
            # and left may be rightmost than right
            patch_mask = current_camera_points[..., 2:3] > 0
            patch_mask = torch.cat([patch_mask, patch_mask], dim=-1)
            projected_points_patched_max = projected_points.clone()
            projected_points_patched_max[patch_mask] = 1e20
            projected_points_patched_min = projected_points.clone()
            projected_points_patched_min[patch_mask] = -1e20

            all_object_projected_bounding_box_points.append(projected_points)

        all_object_projected_bounding_box_points = torch.stack(all_object_projected_bounding_box_points, dim=-1)

        all_object_projected_bounding_box_points[..., 0, :] = (all_object_projected_bounding_box_points[..., 0, :] + (width / 2)) / width
        all_object_projected_bounding_box_points[..., 1, :] = (all_object_projected_bounding_box_points[..., 1, :] + (height / 2)) / height

        # Clamps values outsize the image plane
        return all_object_projected_bounding_box_points

    def compute_object_encodings(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                                 bounding_boxes: torch.Tensor, reconstructed_bounding_boxes: torch.Tensor, global_frame_indexes: torch.Tensor,
                                 video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, shuffle_style: bool):
        '''

        :param observations: (..., observations_count, cameras_count, 3, height, width) observations
        :param camera_rotations: (..., observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (..., observations_count, cameras_count, 3) measured camera_translation
        :param bounding_boxes: (..., observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param reconstructed_bounding_boxes: (..., observations_count, cameras_count, 4, objects_count) normalized reconstructed bounding boxes in [0, 1] for each object instance
        :param global_frame_indexes: (..., observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (..., observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset
        :param shuffle_style: True if style codes should be shuffled between observations at different temporal points

        :return: (..., observations_count, style_features_count, objects_count) tensor with style encoding
                 (..., observations_count, deformation_features_count, objects_count) tensor with deformation encoding
                 list of objects_count (..., observations_count, 1, 1, features_height, features_width) tensor with attention map. Refers only to the first camera
                 list of objects_count (..., observations_count, 1, 3, crop_height, crop_width) tensor with cropped image. Refers only to the first camera
        '''

        # Adds the observations_count dimension to video_indexes
        video_indexes = video_indexes.unsqueeze(-1)
        video_indexes, _ = torch.broadcast_tensors(video_indexes, video_frame_indexes)

        # Computes the style and deformation encodings for each object
        style = []
        deformation = []
        object_attention = []
        object_crops = []
        for object_idx in range(self.object_id_helper.objects_count):

            # Computes the id of the model associated to the current instance
            object_model_idx = self.object_id_helper.model_idx_by_object_idx(object_idx)

            is_static = self.object_id_helper.is_static(object_model_idx)
            # Since static objects do not move, their reconstructed bounding box is exact and they do not have
            # annotations in the dataset, we use the reconstructed bounding box
            if is_static:
                current_bounding_box = reconstructed_bounding_boxes[..., object_idx]
            # If the object if dynamic, use the bounding box in the datasaet
            else:
                current_bounding_box = bounding_boxes[..., self.object_id_helper.dynamic_object_idx_by_object_idx(object_idx)]

            current_object_style_encoder = self.object_encoders[object_model_idx]
            current_object_style, current_object_deformations, current_object_attention, current_object_crops = current_object_style_encoder(
                observations, current_bounding_box, camera_rotations, camera_translations, global_frame_indexes, video_frame_indexes, video_indexes)
            # Shuffles style codes at different temporal points (along observations_count dimension) if required
            # Continue to shuffle until a permutation that is not the linear order is found
            if shuffle_style:
                permutation_size = current_object_style.size()[-2]
                bad_permutation = torch.tensor(list(range(permutation_size)), device=observations.device, dtype=torch.int64)
                keep_permuting = True
                while keep_permuting:
                    permutation = torch.randperm(permutation_size, device=observations.device)  # Camera dimension not present in style encodings
                    if not torch.all(bad_permutation == permutation):
                        keep_permuting = False
                current_object_style = current_object_style[..., permutation, :]

            style.append(current_object_style)
            deformation.append(current_object_deformations)
            object_attention.append(current_object_attention)
            object_crops.append(current_object_crops)
        style = torch.stack(style, dim=-1)
        deformation = torch.stack(deformation, dim=-1)

        return style, deformation, object_attention, object_crops

    def batchified_composer_call(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor,
                                 transformation_matrix_w2o: torch.Tensor, style: torch.Tensor, deformation: torch.Tensor,
                                 object_in_scene: torch.Tensor, perturb: bool, samples_per_image_batching: int=0,
                                 video_indexes: torch.Tensor=None, canonical_pose: bool=False):
        '''
        Performs a call to the object composer by splitting the rays in batches

        :param ray_origins: see composer documentation
        :param ray_directions: see composer documentation
        :param focal_normals: see composer documentation
        :param transformation_matrix_w2o: see composer documentation
        :param style: see composer documentation
        :param deformation: see composer documentation
        :param object_in_scene: see composer documentation
        :param perturb: see composer documentation
        :param samples_per_image_batching: maximum number of samples that should be forwarded with each batch
                                           if 0 no batching is used
        :param video_indexes: see composer documentation
        :param canonical_pose: see composer documentation

        :return:  see composer documentation
        '''

        dimension_idx = len(ray_directions.size()) - 2
        samples_per_image = ray_directions.size(dimension_idx)

        # If no batching is required create a batch big as all the input images
        if samples_per_image_batching == 0:
            samples_per_image_batching = samples_per_image

        # Divides the ray directions in batches along the samples_per_image direction
        batchified_ray_directions = TensorBatchifier.batchify(ray_directions, dim=-2, batch_size=samples_per_image_batching)

        all_results = []
        for idx, current_ray_directions in enumerate(batchified_ray_directions):
            #print(f"{idx}/{len(batchified_ray_directions)}")
            # Forwards the values to the composer
            current_results = self.object_composer(ray_origins, current_ray_directions, focal_normals, transformation_matrix_w2o, style, deformation, object_in_scene, perturb, video_indexes=video_indexes, canonical_pose=canonical_pose)
            all_results.append(current_results)

        # If batching has been used, merges all results
        if len(current_results) > 0:
            merged_results = self.merge_dictionaries(all_results, dimension=dimension_idx)
        # Otherwise directly returns the result
        else:
            merged_results = all_results[0]

        return merged_results

    def merge_dictionaries(self, dictionaries: List[Dict], dimension: int):
        '''

        :param dictionaries: List of dictionaries contanining tensors to merge.
                             All dictionaries must have the same keys and contain either tensors or dictionaries
        :param dimension: dimension on which to concatenate corresponding tensors
        :return: dictionary with the same keys as the original dictioanries and with all tensors concatenated on the
                 specified dimension
        '''

        merged_results = {}
        for key in list(dictionaries[0].keys()):
            if key == "pytorch_hook":  # A fake key may be present to work around pytorch hook bug with dictionaries. We drop it
                continue
            if torch.is_tensor(dictionaries[0][key]):
                merged_list = [current_dictionary[key] for current_dictionary in dictionaries]  # Gathers all tensors
                merged_tensor = torch.cat(merged_list, dim=dimension)  # Concatenates along the batchified dimension
                merged_results[key] = merged_tensor

            # Must be a dictionary
            else:
                merged_results[key] = self.merge_dictionaries([current_dictionary[key] for current_dictionary in dictionaries], dimension)
        return merged_results

    def fold_dictionary(self, dictionary: Dict, height: int, width: int):
        '''
        Folds all tensors in the dictionary with a dimension equal to the product of height and width into tensors
        with that dimension splitted into a height and width dimension
        :param dictionary: The dictionary to fold
        :param height: target height
        :param width: target width
        :return: the original dictionary with its matching entries folded
        '''

        target_dimension_size = height * width
        for key in dictionary:
            current_element = dictionary[key]
            # If element is dict perform recursive call
            if type(current_element) is dict:
                dictionary[key] = self.fold_dictionary(current_element, height, width)
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
                    reshaped_element = current_element.reshape(current_tensor_sizes[:dimension_idx] + [height, width] + current_tensor_sizes[dimension_idx + 1:])
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

        :param observations: see forward
        :param camera_rotations: see forward
        :param camera_translations: see forward
        :param focals: see forward
        :param bounding_boxes: see forward
        :param bounding_boxes_validity: see forward
        :param global_frame_indexes: see forward
        :param video_frame_indexes: see forward
        :param video_indexes: see forward
        :param perturb: if true applies perturbation to samples
        :param samples_per_image_batching: maximum number of samples that should be forwarded with each batch.
                                           if 0 no batching is used
        :param upsample_factor: see forward
        :param canonical_pose: see forward

        :return: see forward, but fields with a samples_per_image dimension are expressed with a height and width instead
        '''

        # Performs computation of the values for all image pixels
        results_flat = self(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity,
                            global_frame_indexes, video_frame_indexes, video_indexes, 0, perturb, samples_per_image_batching,
                            upsample_factor=upsample_factor, canonical_pose=canonical_pose)
        height = int(observations.size(-2) * upsample_factor)
        width = int(observations.size(-1) * upsample_factor)

        # Folds the tensors to recover height and width
        folded_results = self.fold_dictionary(results_flat, height, width)

        return folded_results

    def render_full_frame_from_scene_encoding(self, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                                            focals: torch.Tensor, image_size: Tuple[int, int], object_rotation_parameters_o2w: torch.Tensor, object_translation_parameters_o2w: torch.Tensor,
                                            object_style: torch.Tensor, object_deformation: torch.Tensor, object_in_scene: torch.Tensor,
                                            perturb: bool, samples_per_image_batching: int = 1000, upsample_factor: float = 1.0, canonical_pose: bool=False) -> Dict[str, torch.Tensor]:
        '''
        Renders a full frames starting from the given observations and camera parameters

        :param camera_rotations: see forward
        :param camera_translations: see forward
        :param focals: see forward
        :param image_size: see forward
        :param object_rotation_parameters_o2w: see forward
        :param object_translation_parameters_o2w: see forward
        :param object_style: see forward
        :param object_deformation: see forward
        :param object_in_scene: see forward
        :param perturb: if true applies perturbation to samples
        :param samples_per_image_batching: maximum number of samples that should be forwarded with each batch.
                                           if 0 no batching is used
        :param upsample_factor: see forward
        :param canonical_pose: see forward

        :return: see forward, but fields with a samples_per_image dimension are expressed with a height and width instead
        '''

        # Performs computation of the values for all image pixels
        results_flat = self(camera_rotations, camera_translations, focals, image_size, object_rotation_parameters_o2w, object_translation_parameters_o2w,  object_style, object_deformation, object_in_scene, 0, perturb, samples_per_image_batching, upsample_factor=upsample_factor, canonical_pose=canonical_pose, mode="scene_encodings")
        height = int(image_size[0] * upsample_factor)
        width = int(image_size[1] * upsample_factor)

        # Folds the tensors to recover height and width
        folded_results = self.fold_dictionary(results_flat, height, width)

        return folded_results

    def compute_ray_object_distances(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, transformation_matrix_o2w: torch.Tensor) -> torch.Tensor:
        '''
        Computes the distance between each object center and each ray

        :param ray_origins: (..., cameras_count, 3) tensor with ray origins
        :param ray_directions: (..., cameras_count, samples_per_image, 3) tensor with ray directions
        :param transformation_matrix_o2w: (..., 4, 4, objects_count) tensor with transformation matrices from object to world coordinates
        :return: (..., cameras_count, samples_per_image, objects_count) tensor with distances from each object to each ray
        '''

        # Adds the samples per image dimension
        ray_origins = ray_origins.unsqueeze(-2)  # (..., cameras_count, samples_per_image, 3)
        # Normalizes the ray directions
        normalized_ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

        # Retrieves the bounding boxes for each object
        bounding_boxes = []
        for object_idx in range(self.object_id_helper.objects_count):
            object_model_idx = self.object_id_helper.model_idx_by_object_idx(object_idx)
            bounding_boxes.append(self.object_composer.object_models_coarse[object_model_idx].bounding_box)
        objects_count = len(bounding_boxes)

        # Computes distances for each object
        all_distances = []
        for object_idx in range(objects_count):
            current_transformation_matrix_o2w = transformation_matrix_o2w[..., object_idx]
            # Gets the object center in object coordinates
            current_object_center = bounding_boxes[object_idx].get_center_offset(device=ray_origins.device)

            # Computes the center of objects in world coordinates
            current_object_center = RayHelper.transform_points(current_object_center, current_transformation_matrix_o2w)

            # Adds the cameras_count and samples_per_image dimensions
            current_object_center = current_object_center.unsqueeze(-2).unsqueeze(-2)  # (..., cameras_count, samples_per_image, 3)

            # (a - p)
            camera_to_object_vector = ray_origins - current_object_center

            # (a - p) . n
            on_ray_projection_length = torch.sum(camera_to_object_vector * normalized_ray_directions, dim=-1)

            # ((a - p) . n)n
            on_ray_projection = on_ray_projection_length.unsqueeze(-1) * normalized_ray_directions

            # (a - p) - ((a - p) . n)n
            current_distance = camera_to_object_vector - on_ray_projection
            # Computes L2 squared distance
            #print("Warning: using L1 ray distance")
            #current_distance = torch.abs(current_distance).sum(-1)
            current_distance = current_distance.pow(2).sum(-1)
            all_distances.append(current_distance)

        all_distances = torch.stack(all_distances, dim=-1)
        return all_distances

    def compute_decoded_image(self, composition_results: Dict, sampled_positions: torch.Tensor):
        '''
        Decodes the samples computed from the compositor into an image
        Adds the decoded image under the "decoded_observations" key to the same location in the input dictionary where integrated nerf features
        are present. Supports only the use of coarse features. Fine features must not be present.

        Inserts the decoded image into the composition results with a shape of
        (..., observations_count, cameras_count, output_features_count, height, width) tensor with decoded images

        :param composition_results: Dictionary of results from the object composer
                                    Refer to the object composer documentation for its description
        :param sampled_positions: (..., observations_count, cameras_count, samples_per_image, 2) tensor with samples positions
        :return:
        '''

        if not self.use_image_decoder:
            raise Exception("Image decoding was requested, but the use of the image decoder was not configured")

        if "fine" in composition_results:
            raise Exception("Image decoding is being used only on the coarse features, but fine features are being computed anyway. Please disable the fine nerf models.")

        self.time_meter.start("grid_sampling")

        integrated_features = composition_results["coarse"]["global"]["integrated_features"]
        sampled_grid = self.grid_sampler(integrated_features, sampled_positions)

        self.time_meter.end("grid_sampling")
        self.time_meter.start("image_decoding")

        decoded_images = self.image_decoder(sampled_grid)

        self.time_meter.end("image_decoding")

        composition_results["coarse"]["global"]["decoded_images"] = decoded_images

    def forward(self, *args, mode="observations", **kwargs):
        '''

        :param args: Positional arguments to pass to the target forward mode
        :param mode: Mode to use for the forward
                     "observations" reconstructs a scene from its observations
                     "scene_encodings" reconstructs a scene from its encoding
                     "observations_scene_encoding_only" reconstructs only scene encodings from its observations
                     "pose_consistency" performs a forward pass for estimation of pose consistency
        :param kwargs: Keyword arguments to pass to the target forward mode
        :return: Output of the chosen forward mode
        '''

        return_value = None
        if mode == "observations":
            return_value = self.forward_from_observations(*args, **kwargs)
        elif mode == "scene_encodings":
            return_value = self.forward_from_scene_encoding(*args, **kwargs)
        elif mode == "observations_scene_encoding_only":
            return_value = self.forward_scene_encoding_from_observations(*args, **kwargs)
        elif mode == "pose_consistency":
            return_value = self.forward_pose_consistency(*args, **kwargs)
        elif mode == "keypoint_consistency":
            return_value = self.forward_keypoint_consistency(*args, **kwargs)
        else:
            raise Exception(f"Unknown forward mode '{mode}'")

        return return_value

    def forward_scene_encoding_from_observations(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, shuffle_style: bool = False) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param observations: (..., observations_count, cameras_count, 3, height, width) observations
        :param camera_rotations: (..., observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (..., observations_count, cameras_count, 3) measured camera_translation
        :param focals: (..., observations_count, cameras_count) measured camera focal lengths
        :param bounding_boxes: (..., observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the dynamic object is present in the scene
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param shuffle_style: True if style codes should be shuffled between observations at different temporal points

        :return: Dictionary with scene encodings fields as described in forward from observations
        '''

        if self.enable_camera_parameters_offsets:
            # Obtains camera parameter offsets for the current frames
            camera_rotation_offsets, camera_translation_offsets, focal_offsets = self.camera_parameters_offsets(global_frame_indexes)
            # Applies offsets to the camera parameters
            camera_rotations = camera_rotations + camera_rotation_offsets
            camera_translations = camera_translations + camera_translation_offsets
            focals = focals + camera_rotation_offsets

        # Corrects for change in focals due to image rescaling
        rescaled_focals = focals * self.focal_length_multiplier

        # Extracts the observation dimensions
        observation_sizes = list(observations.size())
        height = observation_sizes[-2]
        width = observation_sizes[-1]

        # Computes the transformation matrices for each image
        transformation_c2w = PoseParameters(camera_rotations, camera_translations)
        # Computes the position of each object and the transformation parameters from world to object for each of them. Camera parameters should not receive gradients from losses derived from object positions
        object_rotation_parameters_o2w, object_translation_parameters_o2w = self.compute_rotation_translation_o2w(observations, transformation_c2w.detach(), camera_rotations, rescaled_focals.detach(), bounding_boxes, bounding_boxes_validity)
        all_transformation_matrices_w2o, all_transformation_matrices_o2w = self.compute_transformation_matrix_w2o_o2w(object_rotation_parameters_o2w, object_translation_parameters_o2w)

        # Computes the bounding boxes for each object. Camera parameters should not receive gradients from losses derived from bounding boxes
        transformation_matrix_w2c = transformation_c2w.get_inverse_homogeneous_matrix()
        reconstructed_bounding_boxes, reconstructed_3d_bounding_box_points = self.compute_object_bounding_boxes(all_transformation_matrices_o2w, transformation_matrix_w2c.detach(), rescaled_focals.detach(), height, width)

        # Computes the style and deformation encodings for each object
        style, deformation, object_attention, object_crops = self.compute_object_encodings(observations, camera_rotations, camera_translations, bounding_boxes, reconstructed_bounding_boxes, global_frame_indexes, video_frame_indexes, video_indexes, shuffle_style)

        # Creates a tensor that for each object instance, both static and dynamic, indicates whether it is present in the
        # scene. Static objects are always present. Dynamic objects are present if they are detected in at least one camera
        bounding_boxes_validity_static = torch.ones_like(bounding_boxes_validity[..., 0:1], dtype=torch.bool)
        bounding_boxes_validity_static = torch.cat([bounding_boxes_validity_static] * self.object_id_helper.static_objects_count, dim=-1)
        object_in_scene = torch.cat([bounding_boxes_validity_static, bounding_boxes_validity], dim=-1)
        object_in_scene, _ = object_in_scene.max(dim=-2, keepdim=True)

        scene_encoding = {
            "camera_rotations": camera_rotations,
            "camera_translations": camera_translations,
            "focals": focals,
            "object_rotation_parameters": object_rotation_parameters_o2w,
            "object_translation_parameters": object_translation_parameters_o2w,
            "object_style": style,
            "object_deformation": deformation,
            "object_in_scene": object_in_scene[..., 0, :]
        }

        return scene_encoding

    def printtime(self, start, end, label):
        end.record()
        torch.cuda.synchronize()

        print(f"{label}: {start.elapsed_time(end)}")

    def forward_from_observations(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, samples_per_image: int, perturb: bool,
                samples_per_image_batching: int = 0, shuffle_style: bool = False, upsample_factor: float = 1.0,
                patch_size: int = 0, patch_stride: int = 0, align_grid: bool = True, canonical_pose: bool=False) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param observations: (..., observations_count, cameras_count, 3, height, width) observations
        :param camera_rotations: (..., observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (..., observations_count, cameras_count, 3) measured camera_translation
        :param focals: (..., observations_count, cameras_count) measured camera focal lengths
        :param bounding_boxes: (..., observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the dynamic object is present in the scene
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param samples_per_image: number of rays to use for each image. If 0, use a ray for each pixel
        :param perturb: if true applies perturbation to samples
        :param samples_per_image_batching: maximum number of samples that should be forwarded with each batch.
                                           if 0 no batching is used
        :param shuffle_style: True if style codes should be shuffled between observations at different temporal points
        :param patch_size: size of each side of the patch to sample. Must be a multiple of 2
        :param patch_stride: stride between pixels sampled in the batch. Can be a list of strides to sample at multiple resolutions
        :param align_grid: if True align the rays corresponding to the patch to the corresponding feature center
        :param canonical_pose: if True renders the object in the canonical pose

        :return: Dictionary with the fields returned by ObjectComposer.
                 An additional "observations" field is added with tensor of size
                 (batch_size, observations_count, cameras_count, samples_per_image, 3)
                 representing ground truth observations

                 Additional fields may be present
                 "object_rotation_parameters" (batch_size, observations_count, 3, objects_count)
                 "ray_object_distances" (batch_size, observations_count, cameras_count, samples_per_image, objects_count)
                 "reconstructed_bounding_boxes" (batch_size, observations_count, cameras_count, 4, objects_count)
                 "object_translation_parameters" (batch_size, observations_count, 3, objects_count)
                 "object_attention" list of objects_count (batch_size, observations_count, 1, 1, features_height, features_width)
                 "object_crops" list of objects_count (batch_size, observations_count, 1, 3, crop_height, crop_width)
                 "scene_encoding" dictionary with parameters describing the current scene
        '''

        self.time_meter.start("camera_correction")

        if self.enable_camera_parameters_offsets:
            # Obtains camera parameter offsets for the current frames
            camera_rotation_offsets, camera_translation_offsets, focal_offsets = self.camera_parameters_offsets(global_frame_indexes)
            # Applies offsets to the camera parameters
            camera_rotations = camera_rotations + camera_rotation_offsets
            camera_translations = camera_translations + camera_translation_offsets
            focals = focals + focal_offsets

        self.time_meter.end("camera_correction")
        self.time_meter.start("initial_upsampling")

        # Corrects for change in focals due to image rescaling
        rescaled_focals = focals * self.focal_length_multiplier

        # If rendering must happen at higher resolution, upsample the observations
        if upsample_factor != 1.0:
            height = observations.size(-2)
            width = observations.size(-1)
            target_height = int(height * upsample_factor)
            target_width = int(width * upsample_factor)

            # Performs the upsampling
            flat_observations, observation_dimensions = TensorFolder.flatten(observations, -3)
            flat_observations = F.upsample(flat_observations, (target_height, target_width), mode="bilinear")
            observations = TensorFolder.fold(flat_observations, observation_dimensions)

        self.time_meter.end("initial_upsampling")
        self.time_meter.start("camera_rays_creation")

        # Extracts the observation dimensions
        observation_sizes = list(observations.size())
        height = observation_sizes[-2]
        width = observation_sizes[-1]
        initial_observation_sizes = observation_sizes[:-3]

        # Creates the rays for each camera
        ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays(initial_observation_sizes, height, width, rescaled_focals * upsample_factor)

        self.time_meter.end("camera_rays_creation")
        self.time_meter.start("object_localization")

        # Computes the transformation matrices for each image
        transformation_c2w = PoseParameters(camera_rotations, camera_translations)
        # Computes the position of each object and the transformation parameters from world to object for each of them. Camera parameters should not receive gradients from losses derived from object positions
        object_rotation_parameters_o2w, object_translation_parameters_o2w = self.compute_rotation_translation_o2w(observations, transformation_c2w.detach(), camera_rotations, rescaled_focals.detach() * upsample_factor, bounding_boxes, bounding_boxes_validity)
        all_transformation_matrices_w2o, all_transformation_matrices_o2w = self.compute_transformation_matrix_w2o_o2w(object_rotation_parameters_o2w, object_translation_parameters_o2w)

        self.time_meter.end("object_localization")
        self.time_meter.start("bounding_boxes")

        # Computes the bounding boxes for each object. Camera parameters should not receive gradients from losses derived from bounding boxes
        transformation_matrix_w2c = transformation_c2w.get_inverse_homogeneous_matrix()
        reconstructed_bounding_boxes, reconstructed_3d_bounding_box_points = self.compute_object_bounding_boxes(all_transformation_matrices_o2w, transformation_matrix_w2c.detach(), rescaled_focals.detach() * upsample_factor, height, width)
        projected_axes = self.compute_object_axes_projection(all_transformation_matrices_o2w, transformation_matrix_w2c.detach(), rescaled_focals.detach() * upsample_factor, height, width)

        self.time_meter.end("bounding_boxes")
        self.time_meter.start("ray_sampling")

        # Performs sampling of the rays if needed
        if patch_size != 0 and samples_per_image != 0:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays_strided_patch(ray_directions, observations, patch_size, patch_stride, reconstructed_bounding_boxes, self.sampling_weights, align_grid=align_grid)
        # If strided sampling is being used and the full image is requested
        elif patch_stride and samples_per_image == 0:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_all_rays_strided_grid(ray_directions, observations, patch_stride)
        elif self.use_weighted_sampling:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays_weighted(ray_directions, observations, samples_per_image, reconstructed_bounding_boxes, self.sampling_weights)
        else:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays(ray_directions, observations, samples_per_image)

        self.time_meter.end("ray_sampling")
        self.time_meter.start("ray_transformation")

        # Transforms the rays, bringing them to world coordinates
        transformation_matrix_c2w = transformation_c2w.as_homogeneous_matrix_torch()
        transformation_results = RayHelper.transform_rays(ray_origins, sampled_directions, focal_normals, transformation_matrix_c2w)
        transformed_ray_origins, transformed_ray_directions, transformed_focal_normals = transformation_results

        self.time_meter.end("ray_transformation")
        self.time_meter.start("object_encoding")

        # Computes the style and deformation encodings for each object
        style, deformation, object_attention, object_crops = self.compute_object_encodings(observations, camera_rotations, camera_translations, bounding_boxes, reconstructed_bounding_boxes, global_frame_indexes, video_frame_indexes, video_indexes, shuffle_style)

        self.time_meter.end("object_encoding")
        self.time_meter.start("ray_object_distances")

        # Creates a dimension for cameras_count
        per_camera_style = style.unsqueeze(-3)
        per_camera_deformation = deformation.unsqueeze(-3)

        # Compute distances from rays to predicted object centers. For matrices removes the dimension for cameras
        ray_object_distances = self.compute_ray_object_distances(transformed_ray_origins, transformed_ray_directions,
                                                                 all_transformation_matrices_o2w[..., 0, :, :, :])

        self.time_meter.end("ray_object_distances")
        self.time_meter.start("object_in_scene")

        # Creates a tensor that for each object instance, both static and dynamic, indicates whether it is present in the
        # scene. Static objects are always present. Dynamic objects are present if they are detected in at least one camera
        bounding_boxes_validity_static = torch.ones_like(bounding_boxes_validity[..., 0:1], dtype=torch.bool)
        bounding_boxes_validity_static = torch.cat([bounding_boxes_validity_static] * self.object_id_helper.static_objects_count, dim=-1)
        object_in_scene = torch.cat(([bounding_boxes_validity_static] * self.object_id_helper.static_objects_count) + [bounding_boxes_validity], dim=-1)
        object_in_scene, _ = object_in_scene.max(dim=-2, keepdim=True)

        self.time_meter.end("object_in_scene")
        self.time_meter.start("composer")

        # Composes the rays
        expanded_video_indexes, _ = torch.broadcast_tensors(video_indexes.unsqueeze(-1).unsqueeze(-1), transformed_ray_origins[..., 0])
        composition_results = self.batchified_composer_call(transformed_ray_origins, transformed_ray_directions, transformed_focal_normals,
                                                            all_transformation_matrices_w2o, per_camera_style, per_camera_deformation,
                                                            object_in_scene, perturb, samples_per_image_batching, expanded_video_indexes,
                                                            canonical_pose=canonical_pose)

        self.time_meter.end("composer")
        self.time_meter.start("image_decoder")

        # Decodes the image from the samples if requested
        if self.use_image_decoder:
            self.compute_decoded_image(composition_results, sampled_positions)

        self.time_meter.end("image_decoder")
        self.time_meter.print_summary()

        composition_results["observations"] = sampled_observations
        composition_results["positions"] = sampled_positions
        composition_results["object_rotation_parameters"] = object_rotation_parameters_o2w
        composition_results["object_translation_parameters"] = object_translation_parameters_o2w
        composition_results["ray_object_distances"] = ray_object_distances
        composition_results["reconstructed_bounding_boxes"] = reconstructed_bounding_boxes
        composition_results["reconstructed_3d_bounding_boxes"] = reconstructed_3d_bounding_box_points
        composition_results["projected_axes"] = projected_axes
        composition_results["object_attention"] = object_attention
        composition_results["object_crops"] = object_crops

        scene_encoding = {
            "camera_rotations": camera_rotations,
            "camera_translations": camera_translations,
            "focals": focals,
            "object_rotation_parameters": object_rotation_parameters_o2w,
            "object_translation_parameters": object_translation_parameters_o2w,
            "object_style": style,
            "object_deformation": deformation,
            "object_in_scene": object_in_scene[..., 0, :]
        }

        composition_results["scene_encoding"] = scene_encoding

        return composition_results

    def forward_from_scene_encoding(self, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, image_size: Tuple[int, int], object_rotation_parameters_o2w: torch.Tensor, object_translation_parameters_o2w: torch.Tensor,
                object_style: torch.Tensor, object_deformation: torch.Tensor, object_in_scene: torch.Tensor, samples_per_image: int,
                perturb: bool, samples_per_image_batching: int = 0, upsample_factor: float = 1.0, patch_size: int = 0, patch_stride: int = 0, canonical_pose: bool=False) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param camera_rotations: (..., observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (..., observations_count, cameras_count, 3) measured camera_translation
        :param focals: (..., observations_count, cameras_count) measured camera focal lengths
        :param image_size (height, width) size of the image in pixels
        :param object_rotation_parameters_o2w: (..., observations_count, 3, objects_count) tensor with rotation parameters from object to world for each object
        :param object_translation_parameters_o2w: (..., observations_count, 3, objects_count) tensor with translation parameters from object to world for each object
        :param object_style (..., observations_count, style_features_count, objects_count) tensor with style encoding
        :param object_deformation (..., observations_count, deformation_features_count, objects_count) tensor with deformation encoding
        :param object_in_scene (..., observations_count, objects_count) boolean tensor with True if the object is present in the scene

        :param samples_per_image: number of rays to use for each image. If 0, use a ray for each pixel
        :param perturb: if true applies perturbation to samples
        :param samples_per_image_batching: maximum number of samples that should be forwarded with each batch.
                                           if 0 no batching is used
        :param patch_size: size of each side of the patch to sample. Must be a multiple of 2
        :param patch_stride: stride between pixels sampled in the batch. Can be a list of strides to sample at multiple resolutions

        :param canonical_pose: if True renders the object in the canonical pose

        :return: Dictionary with the fields returned by ObjectComposer.
                 An additional "observations" field is added with tensor of size
                 (batch_size, observations_count, cameras_count, samples_per_image, 3)
                 representing ground truth observations

                 Additional fields may be present
                 "object_rotation_parameters" (batch_size, observations_count, 3, objects_count)
                 "ray_object_distances" (batch_size, observations_count, cameras_count, samples_per_image, objects_count)
                 "reconstructed_bounding_boxes" (batch_size, observations_count, cameras_count, 4, objects_count)
                 "object_translation_parameters" (batch_size, observations_count, 3, objects_count)
                 "object_attention" (batch_size, observations_count, 1, 1, features_height, features_width, objects_count)
                 "object_crops" (batch_size, observations_count, 1, 3, crop_height, crop_width, objects_count)
                 "scene_encoding" dictionary with parameters describing the current scene
        '''

        # Corrects for change in focals due to image rescaling
        rescaled_focals = focals * self.focal_length_multiplier

        # Computes the dimensions of the image accounting for the upsample factor
        height = int(image_size[0] * upsample_factor)
        width = int(image_size[1] * upsample_factor)
        initial_sizes = list(camera_rotations.size())[:-1]

        observations_size = initial_sizes + [3, height, width]
        fake_observations = torch.zeros(observations_size, dtype=torch.float, device=camera_rotations.device)

        with torch.no_grad():  # No gradient required for this computation
            # Creates the rays for each camera
            ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays(initial_sizes, height, width, rescaled_focals * upsample_factor)

        # Computes the transformation matrices for each image
        transformation_c2w = PoseParameters(camera_rotations, camera_translations)
        all_transformation_matrices_w2o, all_transformation_matrices_o2w = self.compute_transformation_matrix_w2o_o2w(object_rotation_parameters_o2w, object_translation_parameters_o2w)

        # Computes the bounding boxes for each object
        transformation_matrix_w2c = transformation_c2w.get_inverse_homogeneous_matrix()
        reconstructed_bounding_boxes, reconstructed_3d_bounding_box_points = self.compute_object_bounding_boxes(all_transformation_matrices_o2w, transformation_matrix_w2c, rescaled_focals * upsample_factor, height, width)
        projected_axes = self.compute_object_axes_projection(all_transformation_matrices_o2w, transformation_matrix_w2c.detach(), rescaled_focals.detach(), height, width)

        # Performs sampling of the rays if needed
        # Performs sampling of the rays if needed
        if patch_size != 0 and samples_per_image != 0:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays_strided_patch(ray_directions, fake_observations, patch_size, patch_stride, reconstructed_bounding_boxes, self.sampling_weights, align_grid=True)
        # If patched rendering is being used and the full image is requested
        elif patch_stride and samples_per_image == 0:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_all_rays_strided_grid(ray_directions, fake_observations, patch_stride)
        elif self.use_weighted_sampling:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays_weighted(ray_directions, fake_observations, samples_per_image, reconstructed_bounding_boxes, self.sampling_weights)
        else:
            sampled_directions, sampled_observations, sampled_positions = RayHelper.sample_rays(ray_directions, fake_observations, samples_per_image)

        # Transforms the rays, bringing them to world coordinates
        transformation_matrix_c2w = transformation_c2w.as_homogeneous_matrix_torch()
        transformation_results = RayHelper.transform_rays(ray_origins, sampled_directions, focal_normals, transformation_matrix_c2w)
        transformed_ray_origins, transformed_ray_directions, transformed_focal_normals = transformation_results

        # Creates a dimension for cameras_count
        per_camera_style = object_style.unsqueeze(-3)
        per_camera_deformation = object_deformation.unsqueeze(-3)

        # Adds the camera dimension
        object_in_scene = object_in_scene.unsqueeze(-2)

        # Composes the rays
        composition_results = self.batchified_composer_call(transformed_ray_origins, transformed_ray_directions, transformed_focal_normals,
                                                            all_transformation_matrices_w2o, per_camera_style, per_camera_deformation,
                                                            object_in_scene, perturb, samples_per_image_batching, canonical_pose=canonical_pose)

        # Decodes the image from the samples if requested
        if self.use_image_decoder:
            self.compute_decoded_image(composition_results, sampled_positions)

        composition_results["object_rotation_parameters"] = object_rotation_parameters_o2w
        composition_results["object_translation_parameters"] = object_translation_parameters_o2w
        composition_results["reconstructed_bounding_boxes"] = reconstructed_bounding_boxes
        composition_results["reconstructed_3d_bounding_boxes"] = reconstructed_3d_bounding_box_points
        composition_results["projected_axes"] = projected_axes

        scene_encoding = {
            "camera_rotations": camera_rotations,
            "camera_translations": camera_translations,
            "focals": focals,
            "object_rotation_parameters": object_rotation_parameters_o2w,
            "object_translation_parameters": object_translation_parameters_o2w,
            "object_style": object_style,
            "object_deformation": object_deformation,
            "object_in_scene": object_in_scene[..., 0, :]
        }

        composition_results["scene_encoding"] = scene_encoding

        return composition_results

    def merge_expected_position_results(self, expected_position_results: List[Dict]) -> Dict:
        '''
        Merges the expected position results in a single dictionary
        Each key in the new dictionary is a list with the corresponding items in the original dictionaries with order preserved

        :param expected_position_results: List of dictionaries containing the expected position results
        :return: Merged expected position results. Keys are the same as in the original dictionaries. Each item is a list
        '''

        merged_results = {}
        # Merges each key
        for key in expected_position_results[0].keys():
            merged_results[key] = []
            for current_results in expected_position_results:
                merged_results[key].append(current_results[key])

        return merged_results

    def invert_expected_position_results(self, all_expected_position_results: List[Dict]):
        '''
        Inverts network type (eg. "fine", "coarse") with object id hierarchical orders into the dictionary
        :param all_expected_position_results: results for each object, internally divided by network type
        :return: Dictionary with network type keys that map to dictionaries with object id keys
        '''

        inverted_results = {}
        # Adds the keys for the network type
        for key in all_expected_position_results[0].keys():
            inverted_results[key] = {}

        # Adds the values of each object
        for dynamic_object_id, current_results in enumerate(all_expected_position_results):
            for key in current_results:
                inverted_results[key][f"dynamic_object_{dynamic_object_id}"] = current_results[key]

        return inverted_results

    def forward_pose_consistency(self, optical_flow: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, object_style: torch.Tensor, object_deformation: torch.Tensor,
                object_rotation_parameters_o2w: torch.Tensor, object_translation_parameters_o2w: torch.Tensor,
                samples_per_image: int, perturb: bool) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param optical_flow: (..., observations_count, cameras_count, 2, height, width) tensor with optical_flow normalized in [0, 1]. 1 corresponds to image height or width
                                                                                        the first optical flow channel must correspond to the height dimension, the second to the width dimension
        :param camera_rotations: (..., observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (..., observations_count, cameras_count, 3) measured camera_translation
        :param focals: (..., observations_count, cameras_count) measured camera focal lengths
        :param bounding_boxes: (..., observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the dynamic object is present in the scene
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param object_style (..., observations_count, style_features_count, objects_count) tensor with style encoding
        :param object_deformation (..., observations_count, deformation_features_count, objects_count) tensor with deformation encoding
        :param object_rotation_parameters_o2w (..., observations_count, 3, objects_count) tensor with object rotation parameters
        :param object_translation_parameters_o2w (..., observations_count, 3, objects_count) tensor with object translation parameters
        :param samples_per_image: number of rays to use for each image. If 0, use a ray for each pixel
        :param perturb: if true applies perturbation to samples

        :return: Dictionary with the fields returned by ObjectComposer for pose consistency.
                 Dictionary divided into "coarse" and optionally "fine".
                 Each dictionary contains a key for each dynamic object with its corresponding results in the form of
                      [
                          (..., observations_count - 1, cameras_count, samples_per_image, 3) tensor with expected positions for the preceding image
                          (..., observations_count - 1, cameras_count, samples_per_image, 3) tensor with expected positions of the corresponding points in the successive image
                      ]

                 Additional fields may be present
        '''

        self.time_meter.start("camera_correction")

        if self.enable_camera_parameters_offsets:
            # Obtains camera parameter offsets for the current frames
            camera_rotation_offsets, camera_translation_offsets, focal_offsets = self.camera_parameters_offsets(global_frame_indexes)
            # Applies offsets to the camera parameters
            camera_rotations = camera_rotations + camera_rotation_offsets
            camera_translations = camera_translations + camera_translation_offsets
            focals = focals + focal_offsets

        self.time_meter.end("camera_correction")
        self.time_meter.start("focal_rescaling")

        # Corrects for change in focals due to image rescaling
        rescaled_focals = focals * self.focal_length_multiplier

        self.time_meter.end("focal_rescaling")
        self.time_meter.start("camera_rays_creation")

        # Extracts the observation dimensions
        observation_sizes = list(optical_flow.size())
        height = observation_sizes[-2]
        width = observation_sizes[-1]
        initial_observation_sizes = observation_sizes[:-3]

        # Adds a dimesnion for the camera
        object_style = object_style.unsqueeze(-3)
        object_deformation = object_deformation.unsqueeze(-3)

        # Adds dimensions for the observations_count and for the cameras
        if video_indexes is not None:
            video_indexes = video_indexes.unsqueeze(-1)
            video_indexes = video_indexes.unsqueeze(-1)

        # Creates the rays for each camera
        ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays(initial_observation_sizes, height, width, rescaled_focals)

        self.time_meter.end("camera_rays_creation")
        self.time_meter.start("world_object_matrices_computation")

        # Computes the transformation matrices for each image
        transformation_c2w = PoseParameters(camera_rotations, camera_translations)
        transformation_matrix_c2w = transformation_c2w.as_homogeneous_matrix_torch()
        # Computes the transformation parameters from world to object for each of them.
        all_transformation_matrices_w2o, all_transformation_matrices_o2w = self.compute_transformation_matrix_w2o_o2w(object_rotation_parameters_o2w, object_translation_parameters_o2w)

        self.time_meter.end("world_object_matrices_computation")

        all_object_results = []

        # Computes the sample positions for each object
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count
        for dynamic_object_idx in range(dynamic_objects_count):
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)

            self.time_meter.start("previous_positions_extraction")
            # Extracts the source positions for the optical flow
            current_bounding_box = bounding_boxes[..., dynamic_object_idx]
            current_transformation_matrix_w2o = all_transformation_matrices_w2o[..., object_idx]
            current_object_style = object_style[..., object_idx]
            current_object_deformation = object_deformation[..., object_idx]
            current_bounding_boxes_validity = bounding_boxes_validity[..., dynamic_object_idx]
            previous_ray_origins = ray_origins[..., :-1, :, :]
            previous_ray_directions = ray_directions[..., :-1, :, :, :, :]
            previous_focal_normals = focal_normals[..., :-1, :, :]
            previous_transformation_matrix_w2o = current_transformation_matrix_w2o[..., :-1, :, :, :]
            previous_object_style = current_object_style[..., :-1, :, :]
            previous_object_deformation = current_object_deformation[..., :-1, :, :]
            previous_bounding_boxes_validity = current_bounding_boxes_validity[..., :-1, :]
            previous_optical_flow = optical_flow[..., :-1, :, :, :, :]
            previous_bounding_box = current_bounding_box[..., :-1, :, :]
            previous_transformation_matrix_c2w = transformation_matrix_c2w[..., :-1, :, :, :]

            self.time_meter.end("previous_positions_extraction")
            self.time_meter.start("previous_positions_sampling")

            # Samples the images at the location of the object
            previous_sampled_directions, previous_sampled_optical_flow, previous_sampled_positions = RayHelper.sample_rays_at_object(previous_ray_directions, previous_optical_flow, samples_per_image, previous_bounding_box)

            self.time_meter.end("previous_positions_sampling")
            self.time_meter.start("next_positions_extraction")

            # For each sample, computes the positions where to sample in the successive frame
            next_sampled_positions = previous_sampled_optical_flow + previous_sampled_positions
            # Perform sampling at the new positions
            next_ray_origins = ray_origins[..., 1:, :, :]
            next_ray_directions = ray_directions[..., 1:, :, :, :, :]
            next_focal_normals = focal_normals[..., 1:, :, :]
            next_transformation_matrix_w2o = current_transformation_matrix_w2o[..., 1:, :, :, :]
            next_object_style = current_object_style[..., 1:, :, :]
            next_object_deformation = current_object_deformation[..., 1:, :, :]
            next_bounding_boxes_validity = current_bounding_boxes_validity[..., 1:, :]
            # Do not use correct_range since optical flow is extracted at unknown high resolution, so the mapping error is unknown but small
            next_sampled_directions = RayHelper.sample_rays_at(next_ray_directions, next_sampled_positions, correct_range=False)
            next_transformation_matrix_c2w = transformation_matrix_c2w[..., 1:, :, :, :]

            self.time_meter.end("next_positions_extraction")
            self.time_meter.start("expected_positions_computation")

            # Transforms the rays, bringing them to world coordinates. Transformation for ray origins, and focal normals could be done just once before computing previous and next
            previous_transformation_results = RayHelper.transform_rays(previous_ray_origins, previous_sampled_directions, previous_focal_normals, previous_transformation_matrix_c2w)
            previous_transformed_ray_origins, previous_transformed_sampled_ray_directions, previous_transformed_focal_normals = previous_transformation_results

            next_transformation_results = RayHelper.transform_rays(next_ray_origins, next_sampled_directions, next_focal_normals, next_transformation_matrix_c2w)
            next_transformed_ray_origins, next_transformed_sampled_ray_directions, next_transformed_focal_normals = next_transformation_results

            previous_expected_position_results = self.object_composer.forward_expected_positions(previous_transformed_ray_origins, previous_transformed_sampled_ray_directions, previous_transformed_focal_normals, previous_transformation_matrix_w2o,
                                                                                            previous_object_style, previous_object_deformation, object_in_scene=previous_bounding_boxes_validity,
                                                                                            object_id=object_idx, perturb=perturb, video_indexes=video_indexes)

            next_expected_position_results = self.object_composer.forward_expected_positions(next_transformed_ray_origins, next_transformed_sampled_ray_directions, next_transformed_focal_normals, next_transformation_matrix_w2o,
                                                                                           next_object_style, next_object_deformation, object_in_scene=next_bounding_boxes_validity,
                                                                                           object_id=object_idx, perturb=perturb, video_indexes=video_indexes)

            merged_expected_position_results = self.merge_expected_position_results([previous_expected_position_results, next_expected_position_results])

            all_object_results.append(merged_expected_position_results)

            self.time_meter.end("expected_positions_computation")

        results = self.invert_expected_position_results(all_object_results)

        self.time_meter.print_summary()

        # A bug in pytorch makes it necessary to have at least one tensor in the top level of the returned dictionary
        # for hook registration
        results["pytorch_backward_hook"] = results["coarse"]["dynamic_object_0"][0]

        return results

    def forward_keypoint_consistency(self, observations: torch.Tensor, camera_rotations: torch.Tensor, camera_translations: torch.Tensor,
                                      focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, global_frame_indexes: torch.Tensor,
                                      video_frame_indexes: torch.Tensor, video_indexes: torch.Tensor, object_style: torch.Tensor, object_deformation: torch.Tensor,
                                      object_rotation_parameters_o2w: torch.Tensor, object_translation_parameters_o2w: torch.Tensor,
                                      keypoints: torch.Tensor, keypoints_validity: torch.Tensor, max_samples_per_image: int, perturb: bool) -> Dict[str, torch.Tensor]:
        '''
        Forwards a batch of data through the model

        :param observations: (..., observations_count, cameras_count, 3, height, width) tensor with observations
        :param camera_rotations: (..., observations_count, cameras_count, 3) measured camera_rotation
        :param camera_translations: (..., observations_count, cameras_count, 3) measured camera_translation
        :param focals: (..., observations_count, cameras_count) measured camera focal lengths
        :param bounding_boxes: (..., observations_count, cameras_count, 4, dynamic_objects_count) normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the dynamic object is present in the scene
        :param global_frame_indexes: (bs, observations_count) tensor of integers representing the global indexes corresponding to the frames
        :param video_frame_indexes: (bs, observations_count) tensor of integers representing indexes in the original videos corresponding to the frames
        :param video_indexes: (bs) tensor of integers representing indexes of each video in the dataset
        :param object_style (..., observations_count, style_features_count, objects_count) tensor with style encoding
        :param object_deformation (..., observations_count, deformation_features_count, objects_count) tensor with deformation encoding
        :param object_rotation_parameters_o2w (..., observations_count, 3, objects_count) tensor with object rotation parameters
        :param object_translation_parameters_o2w (..., observations_count, 3, objects_count) tensor with object translation parameters
        :param keypoints: (..., observations_count, cameras_count, keypoints_count, 3, dynamic_objects_count) normalized keypoints in [0, 1] for each dynamic object instance.
                                                                                                              The third dimension is given by (height, width, confidence_score) and the prediction confidence
        :param keypoints_validity: (..., observations_count, cameras_count, dynamic_objects_count) boolean tensor with True if the keypoints detection for the dynamic object are present
        :param max_samples_per_image: maximum number of rays to use for each image. The smallest between keypoints_count and samples_per_image is applied
        :param perturb: if true applies perturbation to samples

        :return: Dictionary with a tuple of
                 the fields returned by ObjectComposer for pose consistency, the confidence score for each prediction.
                 Dictionary divided into "coarse" and optionally "fine".
                 Each dictionary contains a key for each dynamic object with its corresponding results in the form of
                      (
                          (..., observations_count, cameras_count, samples_per_image, 3) tensor with expected positions for the preceding image
                          (..., observations_count, cameras_count, samples_per_image) tensor with confidence scores for each sample
                          (..., observations_count, cameras_count, samples_per_image) tensor with opacity of the object for each sample
                          (..., observations_count, cameras_count, samples_per_image, 2) tensor with the sampled positions in the image plane, normalized in [0, 1] in (height, width) order
                      )

                 Additional fields may be present
        '''

        # TODO initialization of the linear layers in NERF as in original paper

        self.time_meter.start("camera_correction")

        if self.enable_camera_parameters_offsets:
            # Obtains camera parameter offsets for the current frames
            camera_rotation_offsets, camera_translation_offsets, focal_offsets = self.camera_parameters_offsets(global_frame_indexes)
            # Applies offsets to the camera parameters
            camera_rotations = camera_rotations + camera_rotation_offsets
            camera_translations = camera_translations + camera_translation_offsets
            focals = focals + focal_offsets

        self.time_meter.end("camera_correction")
        self.time_meter.start("focal_rescaling")

        # Corrects for change in focals due to image rescaling
        rescaled_focals = focals * self.focal_length_multiplier

        self.time_meter.end("focal_rescaling")
        self.time_meter.start("camera_rays_creation")

        # Extracts the observation dimensions
        observation_sizes = list(observations.size())
        height = observation_sizes[-2]
        width = observation_sizes[-1]
        initial_observation_sizes = observation_sizes[:-3]

        # Adds a dimesnion for the camera
        object_style = object_style.unsqueeze(-3)
        object_deformation = object_deformation.unsqueeze(-3)

        # Adds dimensions for the observations_count and for the cameras
        if video_indexes is not None:
            video_indexes = video_indexes.unsqueeze(-1)
            video_indexes = video_indexes.unsqueeze(-1)

        # Creates the rays for each camera
        ray_directions, ray_origins, focal_normals = RayHelper.create_camera_rays(initial_observation_sizes, height, width, rescaled_focals)

        self.time_meter.end("camera_rays_creation")
        self.time_meter.start("world_object_matrices_computation")

        # Computes the transformation matrices for each image
        transformation_c2w = PoseParameters(camera_rotations, camera_translations)
        transformation_matrix_c2w = transformation_c2w.as_homogeneous_matrix_torch()
        # Computes the transformation parameters from world to object for each of them.
        all_transformation_matrices_w2o, all_transformation_matrices_o2w = self.compute_transformation_matrix_w2o_o2w(object_rotation_parameters_o2w, object_translation_parameters_o2w)

        self.time_meter.end("world_object_matrices_computation")

        all_object_results = []

        # Computes the sample positions for each object
        dynamic_objects_count = self.object_id_helper.dynamic_objects_count
        for dynamic_object_idx in range(dynamic_objects_count):
            object_idx = self.object_id_helper.object_idx_by_dynamic_object_idx(dynamic_object_idx)

            self.time_meter.start("current_object_extraction")
            # Extracts data for the current object
            current_transformation_matrix_w2o = all_transformation_matrices_w2o[..., object_idx]
            current_keypoints = keypoints[..., dynamic_object_idx]
            current_object_style = object_style[..., object_idx]
            current_object_deformation = object_deformation[..., object_idx]
            current_bounding_boxes_validity = bounding_boxes_validity[..., dynamic_object_idx]

            self.time_meter.end("current_object_extraction")
            self.time_meter.start("current_object_sampling")

            # Samples the images at the location of the object
            sampled_directions, sampled_positions, sampled_keypoint_confidences = RayHelper.sample_rays_at_keypoints(ray_directions, current_keypoints, max_samples_per_image)

            self.time_meter.end("current_object_sampling")

            self.time_meter.start("expected_positions_computation")

            # Transforms the rays, bringing them to world coordinates. Transformation for ray origins, and focal normals could be done just once before computing previous and next
            transformation_results = RayHelper.transform_rays(ray_origins, sampled_directions, focal_normals, transformation_matrix_c2w)
            transformed_ray_origins, transformed_sampled_ray_directions, transformed_focal_normals = transformation_results

            expected_position_results = self.object_composer.forward_expected_positions(transformed_ray_origins, transformed_sampled_ray_directions, transformed_focal_normals, current_transformation_matrix_w2o,
                                                                                            current_object_style, current_object_deformation, object_in_scene=current_bounding_boxes_validity,
                                                                                            object_id=object_idx, perturb=perturb, video_indexes=video_indexes)

            # Couples the results with confidence scores
            for key in list(expected_position_results):
                expected_positions = expected_position_results[key][0]
                opacity = expected_position_results[key][1]
                expected_position_results[key] = (expected_positions, sampled_keypoint_confidences, opacity, sampled_positions)

            all_object_results.append(expected_position_results)

            self.time_meter.end("expected_positions_computation")

        results = self.invert_expected_position_results(all_object_results)

        self.time_meter.print_summary()

        # A bug in pytorch makes it necessary to have at least one tensor in the top level of the returned dictionary
        # for hook registration
        results["pytorch_backward_hook"] = results["coarse"]["dynamic_object_0"][0]

        return results


def model(config):
    return EnvironmentModel(config)
