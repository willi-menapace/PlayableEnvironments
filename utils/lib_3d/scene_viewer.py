import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image

from utils.lib_3d.bounding_box import BoundingBox
from utils.lib_3d.pose_parameters import PoseParameters


class SceneViewer:
    '''
    Helper class for the visualization of 3D scenes
    '''

    def __init__(self, ambient_light=[0.2, 0.2, 0.2], bg_color=[0.0, 0.0, 0.0], axes=True):
        '''
        Creates a scene viewer
        :param ambient_light: the ambient light in the scene
        :param bg_color: the background color of the scene
        '''

        self.scene = pyrender.Scene(ambient_light=ambient_light, bg_color=bg_color)

        if axes:
            self.add_axes()

    def add_bounding_box(self, bounding_box: BoundingBox, transformation_matrix_o2w: torch.Tensor):
        '''
        Adds a bounding box to the scene

        :param bounding_box: the bounding box to add
        :param transformation_matrix_o2w: (4, 4) transformation matrix from object to world
        :return:
        '''
        transformation_matrix_o2w = transformation_matrix_o2w.detach().cpu().numpy()
        shape_dimensions = bounding_box.get_size().detach().cpu().numpy()

        # Since pyrender places the box with the center of the bottom side at (0, 0, 0) we may need to translate it
        # in order to reflect the possible displacements of the bounding box with respect to the bounding box coordinate system
        center_offset = bounding_box.get_center_offset().detach().cpu().numpy()
        offset_transformation = np.eye(4, dtype=np.float)
        offset_transformation[:3, 3] += center_offset

        # First translate in the object coordinate system to apply offset, then convert to world
        transformation_matrix_o2w = np.matmul(transformation_matrix_o2w, offset_transformation)

        # Inserts the object in the scene
        triemsh_mesh = trimesh.creation.box(shape_dimensions)
        # Makes it transparent
        triemsh_mesh.visual.face_colors[:, -1] = 60
        current_mesh = pyrender.Mesh.from_trimesh(triemsh_mesh, smooth=False)
        mesh_node = pyrender.Node(mesh=current_mesh, matrix=transformation_matrix_o2w)
        self.scene.add_node(mesh_node)

    def add_points(self, points: torch.Tensor, color=None, radius=0.05):
        '''
        Inserts the points into the scene

        :param points: (..., 3) tensor with points
        :param color: (3) tuple with color to give to all points
        :return:
        '''

        # Flattens points and sample color if necessary
        points = points.reshape((-1, 3))
        points_count = points.size(0)
        points = points.detach().cpu().numpy()
        if color is None:
            color = np.random.uniform(size=3)

        # Creates a single mesh for all points
        sphere = trimesh.creation.uv_sphere(radius=radius)
        sphere.visual.vertex_colors = color

        # Creates translation matrices for each point
        transformation_matrices_o2w = np.tile(np.eye(4), (points_count, 1, 1))
        transformation_matrices_o2w[:, :3, 3] = points

        # Creates the mesh and adds it to the scene
        points_mesh = pyrender.Mesh.from_trimesh(sphere, poses=transformation_matrices_o2w)

        mesh_node = pyrender.Node(mesh=points_mesh)
        self.scene.add_node(mesh_node)

    def add_rays(self, ray_origins: torch.Tensor, ray_directions: torch.Tensor, focal_normals: torch.Tensor,
                 ray_positions: torch.Tensor = None, color=None, radius=0.05, spatial=False):
        '''
        Inserts the camera rays into the scene

        :param ray_origins: (..., 3) tensor with ray origins
        :param ray_directions: (..., samples_per_image, 3) tensor with ray directions
        :param focal_normals: (..., 3) tensor with focal normals
        :param ray_positions: (..., samples_per_image, positions_count, 3) tensor with ray directions
        :param color: (3) tuple with color to give to all points
        :param radius: the radius to use for each point
        :param spatial: False if rays have been sampled, True if they still contain the spatial dimension. Instead of
                        the samples_per_image dimension they contain dimensions (height, width)
        :return:
        '''

        if spatial:
            height = ray_directions.size(-3)
            width = ray_directions.size(-2)

            # Samples only the corners in the camera viewing space
            ray_directions = torch.stack([
                ray_directions[..., 0, 0, :],
                ray_directions[..., height - 1, 0, :],
                ray_directions[..., 0, width - 1, :],
                ray_directions[..., height - 1, width - 1, :]
            ], dim=-2)

            # Samples only the ocrners
            ray_positions = torch.stack([
                ray_positions[..., 0, 0, :, :],
                ray_positions[..., height - 1, 0, :, :],
                ray_positions[..., 0, width - 1, :, :],
                ray_positions[..., height - 1, width - 1, :, :]
            ], dim=-2)

        ray_origins = ray_origins[..., 0, :]
        ray_directions = ray_directions[..., 0, :, :]
        focal_normals = focal_normals[..., 0, :]
        ray_positions = ray_positions[..., 0, :, :, :]

        self.add_points(ray_origins, color=[1.0, 1.0, 1.0], radius=4 * radius)
        self.add_points(ray_origins.unsqueeze(-2) + ray_directions, color=color)
        self.add_points(ray_origins + focal_normals, color=[1.0, 1.0, 1.0], radius=4 * radius)
        self.add_points(ray_positions, color=color)

    def add_axes(self):
        length = 10
        size = 0.01

        for dim_idx in range(3):
            current_color = np.zeros((4,), np.float)
            current_color[dim_idx] = 1.0
            current_color[3] = 1.0
            offset_transformation = np.eye(4, dtype=np.float)
            offset_transformation[dim_idx, 3] += length / 2
            dimensions = np.ones((3,)) * size
            dimensions[dim_idx] = length

            # Inserts the object in the scene
            triemsh_mesh = trimesh.creation.box(dimensions)
            triemsh_mesh.visual.face_colors = triemsh_mesh.visual.face_colors * 0 + current_color
            current_mesh = pyrender.Mesh.from_trimesh(triemsh_mesh, smooth=False)
            mesh_node = pyrender.Node(mesh=current_mesh, matrix=offset_transformation)
            self.scene.add_node(mesh_node)

    def render_views(self):
        '''
        Views the scene interactively

        :return:
        '''

        r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)

        camera_poses = [PoseParameters(rotation=[0.0, 0.0, 0.0], translation=[0.0, 2.0, 15])]
        #camera_poses = PoseParameters.generate_camera_poses_on_sphere(np.pi / 4, 20, 4)
        #camera_poses += PoseParameters.generate_camera_poses_on_sphere(0, 20, 4)
        #camera_poses += PoseParameters.generate_camera_poses_on_sphere(-np.pi / 2, 20, 3)
        all_camera_nodes = []
        for current_pose in camera_poses:
            current_matrix = current_pose.as_homogeneous_matrix_numpy()
            pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.333)
            nc = pyrender.Node(camera=pc, matrix=current_matrix)
            all_camera_nodes.append(nc)
            self.scene.add_node(nc)
            self.scene.main_camera_node = nc

            image, depth_image = r.render(self.scene)
            Image.fromarray(image).show()


        pass