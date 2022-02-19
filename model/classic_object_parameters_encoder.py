import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.configuration import Configuration
from utils.lib_3d.pose_parameters import PoseParameters
from utils.lib_3d.ray_helper import RayHelper
from utils.tensor_displayer import TensorDisplayer
from utils.tensor_folder import TensorFolder


class ClassicObjectParametersEncoder(nn.Module):
    '''
    Computes the geometry of the scene using classical computer vision techniques
    '''

    def __init__(self, config, model_config):
        '''
        Initializes the model representing the geometry of the scene

        :param config: the configuration file
        '''
        super(ClassicObjectParametersEncoder, self).__init__()

        self.config = config
        self.model_config = model_config
        self.objects_count = model_config["objects_count"]

        # Axis whose translation component is always zero
        # By default the Z axis is used (tennis)
        self.zero_axis = 2
        if "zero_axis" in model_config:
            self.zero_axis = model_config["zero_axis"]

        # (objects_count, 3, 2)
        # Creates tensors for the ranges of rotation and translation
        self.register_buffer("translation_range", torch.tensor(model_config["translation_range"], dtype=torch.float32))
        # (objects_count, 3, 2)
        self.register_buffer("rotation_range", torch.tensor(model_config["rotation_range"], dtype=torch.float32))

    """
    def compute_ground_projection(self, observations: torch.Tensor, transformation_matrix_w2c: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor):
        '''
        Projects the observations for the same scene coming from different cameras into a common space by projecting
        each observation onto a common region on the ground xz plane

        :param observations: (..., cameras_count, 3, height, width) tensor with observations
        :param transformation_matrix_w2c: (..., cameras_count, 4, 4) tensor with transformation
                                          from world to camera coordinates
        :param focals: (..., cameras_count) tensor of focal lengths to use for the projection
        :param bounding_boxes: (..., cameras_count, 4) tensor (left, top, right, bottom) of normalized bounding boxes in [0, 1]
        :return: (..., cameras_count, 3, ground_projection_height, ground_projection_width)
                 tensor with observations projected on the ground
        '''

        # Retrieves dimensions of the observations
        observations_height = observations.size(-2)
        observations_width = observations.size(-1)

        flat_focals, _ = TensorFolder.flatten(focals, -1)

        # Retrieves the ranges
        x_range = self.ground_projection_range[0]
        secondary_axis_range = self.ground_projection_range[1]
        projection_height, projection_width = self.ground_projection_resolution

        # Computes coordinates on the ground that must be mapped in the original images
        secondary_axis_coordinates, x_coordinates = torch.meshgrid([torch.linspace(*secondary_axis_range, projection_height), torch.linspace(*x_range, projection_width)])
        secondary_axis_coordinates = secondary_axis_coordinates.to(observations.device)
        x_coordinates = x_coordinates.to(observations.device)
        zero_axis_coordinates = torch.zeros_like(secondary_axis_coordinates) # All points are sampled on the ground
        # Defines on which plane to perform the projection of the image
        if self.ground_truth_projection_axes == "xy":
            coordinates_list = [x_coordinates, secondary_axis_coordinates, zero_axis_coordinates]
        else:
            assert(self.ground_truth_projection_axes == "xz")
            coordinates_list = [x_coordinates, zero_axis_coordinates, secondary_axis_coordinates]
        # (ground_projection_height, ground_projection_width, 3) x, y, z coordinates of each point on the ground
        world_coordinate_mesh = torch.stack(coordinates_list, dim=-1)
        flat_world_coordinate_mesh = world_coordinate_mesh.reshape((-1, 3))

        # Adds all the dimensions up to cameras_count in the transformation matrices
        dimensions_to_add = len(transformation_matrix_w2c.size()) - 2
        for _ in range(dimensions_to_add):
            flat_world_coordinate_mesh = flat_world_coordinate_mesh.unsqueeze(0)
        # Adds the dimension for the (ground_projection_height * ground_projection_width) points
        transformation_matrix_w2c = transformation_matrix_w2c.unsqueeze(-3)
        # Points in the coordinate system of each camera (..., cameras_count, ground_projection_height * ground_projection_width, 3)
        flat_camera_coordinate_mesh = RayHelper.transform_points(flat_world_coordinate_mesh, transformation_matrix_w2c)
        # Adds dimensions for broadcasting with the (ground_projection_height * ground_projection_width, 3) dimensions
        flat_focals = flat_focals.unsqueeze(-1).unsqueeze(-1)
        # Projects the points on the focal plane of each camera. - is used because cameras face negative z
        flat_image_plane_coordinate_mesh = -flat_camera_coordinate_mesh / flat_camera_coordinate_mesh[..., -1:] * flat_focals
        flat_image_plane_coordinate_mesh = flat_image_plane_coordinate_mesh[..., :2]
        # Negates the y since in the tensors y increases downwards
        flat_image_plane_coordinate_mesh[..., 1] *= -1
        # Shifts the coordinates because tensors are centered on the top left and not in the middle
        # Normalizes coordinates between -1 and +1
        flat_image_plane_coordinate_mesh[..., 0] /= observations_width / 2
        flat_image_plane_coordinate_mesh[..., 1] /= observations_height / 2

        # Separates again height and width
        flat_coordinate_mesh_shape = [-1, projection_height, projection_width, 2]
        flat_image_plane_coordinate_mesh = flat_image_plane_coordinate_mesh.reshape(flat_coordinate_mesh_shape)

        # Denormalizes bounding box coordinates and flattens them
        bounding_boxes = bounding_boxes.clone()
        bounding_boxes[..., [0, 2]] *= observations_width
        bounding_boxes[..., [1, 3]] *= observations_height
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -1)

        # Flattens the observations
        flat_observations, observation_removed_dimensions = TensorFolder.flatten(observations, -3)

        # Masks the observations in the areas outside the bounding boxes
        flat_observations = self.compute_masked_observations(flat_observations, flat_bounding_boxes)

        # Computes the projections
        projections = F.grid_sample(flat_observations, flat_image_plane_coordinate_mesh, )
        # Stacks the channels of the different cameras and reintruduces the initial dimensions
        projections = projections.reshape(observation_removed_dimensions + [3, projection_height, projection_width])

        return projections
    """

    def forward(self, observations: torch.Tensor, transformation_matrix_w2c: torch.Tensor, camera_rotations: torch.Tensor, focals: torch.Tensor, bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor, apply_ranges=True) -> Tuple[torch.Tensor]:
        '''
        Obtains the translations o2w of each scene object represented in the observations

        :param observations: (..., cameras_count, 3, height, width) observations
        :param transformation_matrix_w2c: (..., cameras_count, 4, 4) tensor with transformation
                                          from world to camera coordinates
        :param camera_rotations: (..., cameras_count, 3) tensor with camera rotations
        :param focals: (..., cameras_count) tensor with focal length associated to each camera
        :param bounding_boxes: (..., cameras_count, 4, dynamic_objects_count) tensor with normalized bounding boxes in [0, 1] for each dynamic object instance
        :param bounding_boxes_validity: (..., cameras_count, dynamic_objects_count) tensor with True is the corresponding dynamic object instance is present in the scene
        :param apply_ranges: If true applies the ranges specified in the configuration file that modify translations and rotations

        :return: (..., 3, objects_count) tensor with rotation parameters from object to world for each object
                 (..., 3, objects_count) tensor with translation parameters from object to world for each object
        '''

        eps = 1e-6

        observations = observations[..., :1, :, :, :]
        transformation_matrix_w2c = transformation_matrix_w2c[..., :1, :, :]
        focals = focals[..., :1]
        bounding_boxes = bounding_boxes[..., :1, :, :]
        bounding_boxes_validity = bounding_boxes_validity[..., :1, :]
        if random.randint(0, 100) == 0:
            print("Warning: using only the first camera for extracting object parameters")

        # Retrieves dimensions of the observations
        observations_height = observations.size(-2)
        observations_width = observations.size(-1)

        # Expresses the bounding boxes in pixels
        bounding_boxes = bounding_boxes.clone()
        bounding_boxes[..., [0, 2], :] *= observations_width
        bounding_boxes[..., [1, 3], :] *= observations_height

        # Removes the dimensions before cameras_count
        flat_observations, initial_observations_dimensions = TensorFolder.flatten(observations, -4)
        flat_transformation_matrix_w2c, _ = TensorFolder.flatten(transformation_matrix_w2c, -3)
        flat_focals, _ = TensorFolder.flatten(focals, -1)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -3)
        flat_bounding_boxes_validity, _ = TensorFolder.flatten(bounding_boxes_validity, -2)

        # Computes the transformation matrix from camera to world
        flat_transformation_matrix_c2w = flat_transformation_matrix_w2c[:, 0].inverse()

        objects_count = flat_bounding_boxes.size(-1)
        # Checks the number of objects
        if objects_count != self.objects_count:
            print(f"Warning: The encoder should encode {self.objects_count} objects, but {objects_count} were passed")

        all_object_rotations = []
        all_object_translations = []
        for object_idx in range(objects_count):
            elements_count = flat_bounding_boxes.size(0)
            current_bounding_boxes = flat_bounding_boxes[..., 0, :, object_idx]

            # Computes the validity of an object. An object is valid if it has been detected in at least one camera
            current_bounding_boxes_validity = flat_bounding_boxes_validity[..., 0, object_idx]

            ray_origins = torch.zeros((elements_count, 3), dtype=torch.float, device=observations.device)
            # Center of the bounding box minus the image center
            feet_image_x_coordinates = ((current_bounding_boxes[:, 0] + current_bounding_boxes[:, 2]) / 2) - (observations_width / 2)
            # Bottom of the boudning box minus the image center. - accounts for y axis going up in camera coordinates
            feet_image_y_coordinates = -(current_bounding_boxes[:, 3] - (observations_height / 2))
            # Position of the camera plane. - accounts for the camera facing the -z direction
            feet_image_z_coordinates = -flat_focals[:, 0]

            ray_directions = torch.stack([feet_image_x_coordinates, feet_image_y_coordinates, feet_image_z_coordinates], dim=-1)

            # Transforms origins and directions in camera coordinates
            ray_origins = RayHelper.transform_points(ray_origins, flat_transformation_matrix_c2w, rotation=True, translation=True)
            ray_directions = RayHelper.transform_points(ray_directions, flat_transformation_matrix_c2w, rotation=True, translation=False)

            # Number of times the direction vector must be repeated to touch the ground
            n = -ray_origins[:, self.zero_axis] / (ray_directions[:, self.zero_axis] + eps)
            # Repeats the vector n times to find the position on the ground
            current_object_translations = ray_origins + n.unsqueeze(-1) * ray_directions
            current_object_translations[..., self.zero_axis] *= 0.0  # Forces the axis to be zeroed to be 0
            if apply_ranges:
                current_object_translations[..., self.zero_axis] += ((self.translation_range[object_idx, self.zero_axis, 0] + self.translation_range[object_idx, self.zero_axis, 1]) / 2)  # Increases the zero axis by the average amount for the translation specified for that axis in the configuration file

            # Parameters for objects not present in the scene are 0
            current_object_translations[current_bounding_boxes_validity == False, :] *= 0.0

            # Computes the rotation for the object as the average rotation
            if apply_ranges:
                rotation_min = self.rotation_range[object_idx, :, 0]
                rotation_max = self.rotation_range[object_idx, :, 1]
            else:
                rotation_min = 0.0
                rotation_max = 0.0
            current_object_rotations = torch.ones_like(current_object_translations)
            current_object_rotations = current_object_rotations * ((rotation_max + rotation_min) / 2)

            all_object_rotations.append(current_object_rotations)
            all_object_translations.append(current_object_translations)

        all_object_rotations = torch.stack(all_object_rotations, dim=-1)
        all_object_translations = torch.stack(all_object_translations, dim=-1)

        # Reintroduces the original flattened dimensions
        all_object_rotations = TensorFolder.fold(all_object_rotations, initial_observations_dimensions)
        all_object_translations = TensorFolder.fold(all_object_translations, initial_observations_dimensions)

        # Enables gradient computation
        all_object_rotations.requires_grad = True
        all_object_translations.requires_grad = True

        return all_object_rotations, all_object_translations


def model(config, model_config):
    '''
    Instantiates a nerf model with the given parameters

    :param config:
    :param model_config:
    :return:
    '''

    return ClassicObjectParametersEncoder(config, model_config)

if __name__ == "__main__":

    height = 50
    width = 100
    cameras_count = 1

    config_path = "configs/tennis/01_tennis_v5_ray_bending_style_nerf_bb_1.0_dm_0.0_div_1.0_al_0.00001_spi_150_bs_16.yaml"

    configuration = Configuration(config_path)
    # configuration.check_config()
    # configuration.create_directory_structure()

    config = configuration.get_config()

    scene_encoder = ClassicObjectParametersEncoder(config, config["model"]["object_parameters_encoder"][1])

    observations = torch.ones((1, 3, height, width), dtype=torch.float32).cuda()
    bounding_boxes = torch.as_tensor([[0.5, 0.5, 1.0, 1.0]]).cuda()
    focals = torch.as_tensor([700]).cuda(0)
    pose = PoseParameters([-np.pi/2, 0, 0], [0, 3, +0]) # This is c2w
    transformation_matrix = pose.get_inverse_homogeneous_matrix().cuda() # So we invert it to get w2c
    transformation_matrix = transformation_matrix.unsqueeze(0)

    projections = scene_encoder.compute_ground_projection(observations, transformation_matrix, focals, bounding_boxes)
    TensorDisplayer.show_image_tensor(projections[0])
    pass


