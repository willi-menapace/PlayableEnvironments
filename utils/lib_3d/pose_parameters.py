from typing import Sequence, Tuple

import numpy as np
import torch

from utils.lib_3d.transformations_3d import Transformations3D


class PoseParameters:
    '''
    Represents fundamental pose parameters such as position and orientation in the space
    If inputs are torch tensors, preserves differentiability
    '''

    def __init__(self, rotation: Sequence, translation: Sequence):
        '''

        :param rotation: (..., (rotation_x (tilt), rotation_y (pan))) array
        :param translation: (..., (x, y, z)) translation
        '''

        if torch.is_tensor(rotation):
            self.rotation = rotation
        else:
            self.rotation = torch.as_tensor(rotation, dtype=torch.float32)

        if torch.is_tensor(translation):
            self.translation = translation
        else:
            self.translation = torch.as_tensor(translation, dtype=torch.float32)

        if self.rotation.size(-1) != 3:
            raise Exception(f"Received a rotation of length ({self.rotation.size(-1)}) elements, while (3) were expected")
        if self.translation.size(-1) != 3:
            raise Exception(f"Received a translation of length ({self.translation.size(-1)}) elements, while (3) were expected")

    def detach(self):
        '''
        Creates an equivalent pose parameters object but with detached attributes
        :return: PoseParameters with detached attributes
        '''

        return PoseParameters(self.rotation.detach(), self.translation.detach())

    def as_homogeneous_matrix_numpy(self) -> np.ndarray:
        '''
        Transforms the pose parameters into a 4x4 homogeneous transformation matrix
        :return: (..., 4, 4) numpy array
        '''

        return self.as_homogeneous_matrix_torch().detach().cpu().numpy()

    def as_homogeneous_matrix_torch(self) -> torch.Tensor:
        '''
        Transforms the pose parameters into a 4x4 homogeneous transformation matrix
        :return: (..., 4, 4) torch tensor
        '''

        return Transformations3D.homogeneous_rotation_translation(self.rotation, self.translation)

    def get_rotation_translation(self) -> Tuple[torch.Tensor]:
        '''
        Get the transformation rotation and translation parameters
        :return: (..., (x rotation, y rotation, z_rotation)) rotation tensor, (..., (x, y, z)) translation tensor
        '''

        return self.rotation, self.translation

    def get_inverse_homogeneous_matrix(self) -> torch.Tensor:
        '''
        Computes the inverse transformation and returns them as a 4x4 homogeneous transformation matrix

        :return: (..., 4, 4) torch tensor
        '''

        result = self.as_homogeneous_matrix_torch().inverse()
        return result

    @staticmethod
    def generate_camera_poses_on_sphere(elevation: float, distance: float, num_cameras: int, offset: float = 0.0):
        '''
        Generates camera poses in a sphere with cameras looking towards the center
        :param elevation: elevation in radians of the cameras
        :param distance: the distance from the origin
        :param num_cameras: number of cameras
        :param offset: offset to add to the angles on the xz plane where to place cameras
        :return:
        '''

        # Angles measured in the xz plane starting from the -z axis (where the default camera points)
        angles = np.linspace(0, 2 * np.pi, num=num_cameras, endpoint=False) + offset
        camera_poses = []

        for current_angle in angles:
            rotation_x = -elevation
            rotation_y = current_angle + np.pi
            rotation_z = 0.0

            cos_elevation = np.cos(elevation)
            translation_x = distance * cos_elevation * np.cos(current_angle + np.pi / 2)
            translation_y = distance * np.sin(elevation)
            translation_z = distance * cos_elevation * -np.sin(current_angle + np.pi / 2)

            camera_poses.append(PoseParameters([rotation_x, rotation_y, rotation_z], [translation_x, translation_y, translation_z]))

        return camera_poses


class PoseParametersNumpy:
    '''
    Represents fundamental pose parameters such as position and orientation in the space
    If inputs are torch tensors, preserves differentiability
    '''

    def __init__(self, rotation: Sequence, translation: Sequence):
        '''

        :param rotation: (..., (rotation_x (tilt), rotation_y (pan))) array
        :param translation: (..., (x, y, z)) translation
        '''

        if torch.is_tensor(rotation):
            self.rotation = rotation.cpu().numpy()
        else:
            self.rotation = np.asarray(rotation, dtype=np.float32)

        if torch.is_tensor(translation):
            self.translation = translation.cpu().numpy()
        else:
            self.translation = np.asarray(translation, dtype=np.float32)

        if self.rotation.shape[-1] != 3:
            raise Exception(f"Received a rotation of length ({self.rotation.shape[-1]}) elements, while (3) were expected")
        if self.translation.shape[-1] != 3:
            raise Exception(f"Received a translation of length ({self.translation.shape[-1]}) elements, while (3) were expected")

    def to_torch(self) -> PoseParameters:
        return PoseParameters(self.rotation.tolist(), self.translation.tolist())


if __name__ == "__main__":

    rotations = torch.nn.Parameter(torch.as_tensor([np.pi, np.pi, 0]).unsqueeze(0))
    translations = torch.nn.Parameter(torch.as_tensor([1.0, 1.0, 1.0]).unsqueeze(0))

    pose = PoseParameters(rotations, translations)
    pose_matrix = pose.as_homogeneous_matrix_torch()

    pose_matrix[0, 0, 2].backward()

    pass





















