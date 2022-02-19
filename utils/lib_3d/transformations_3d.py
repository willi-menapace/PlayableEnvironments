import math

import numpy as np
import torch


class Transformations3D:

    @staticmethod
    def rotation_matrix_x(radians: torch.Tensor) -> torch.Tensor:
        '''
        Obtains a 3x3 rotation matrix on the x axis (vertical rotation)
        :param radians: (...) tensor with degrees of which rotate on the x axis. Positive rotation is clockwise
        :return:
        '''
        cos = torch.cos(radians)
        sin = torch.sin(radians)

        dimensions_to_add = list(radians.size())
        elements = torch.zeros(dimensions_to_add + [3, 3], dtype=torch.float32, device=radians.device)
        elements[..., 0, 0] += 1.0
        elements[..., 1, 1] += cos
        elements[..., 1, 2] += -sin
        elements[..., 2, 1] += sin
        elements[..., 2, 2] += cos

        return elements

    @staticmethod
    def rotation_matrix_y(radians: torch.Tensor) -> torch.Tensor:
        '''
        Obtains a 3x3 rotation matrix on the y axis (horizontal rotation)
        :param radians: degrees of which rotate on the y axis. Positive rotation is clockwise
        :return:
        '''
        cos = torch.cos(radians)
        sin = torch.sin(radians)

        dimensions_to_add = list(radians.size())
        elements = torch.zeros(dimensions_to_add + [3, 3], dtype=torch.float32, device=radians.device)
        elements[..., 1, 1] += 1.0
        elements[..., 0, 0] += cos
        elements[..., 2, 0] += -sin
        elements[..., 0, 2] += sin
        elements[..., 2, 2] += cos

        return elements

    @staticmethod
    def rotation_matrix_z(radians: torch.Tensor) -> torch.Tensor:
        '''
        Obtains a 3x3 rotation matrix on the z axis (tilt rotation)
        :param radians: degrees of which rotate on the z axis. Positive rotation is clockwise
        :return:
        '''
        cos = torch.cos(radians)
        sin = torch.sin(radians)

        dimensions_to_add = list(radians.size())
        elements = torch.zeros(dimensions_to_add + [3, 3], dtype=torch.float32, device=radians.device)
        elements[..., 2, 2] += 1.0
        elements[..., 0, 0] += cos
        elements[..., 0, 1] += -sin
        elements[..., 1, 0] += sin
        elements[..., 1, 1] += cos

        return elements

    @staticmethod
    def homogeneous_rotation_translation(rotations: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        '''
        Obtains a 4x4 rotation and translation matrix
        :param rotations: (..., (x_degrees, y_degrees, z_degrees)) rotation angles in radians
        :param translation: (..., (x, y, z)) translation vector
        :return: (..., 4, 4) tensor with the corresponding homogeneous matrix
        '''

        if rotations.size(-1) != 3:
            raise Exception(f"Expected rotations to have size 3, but size {rotations.size(-1)} was provided")
        if translation.size(-1) != 3:
            raise Exception(f"Expected translations to have size 3, but size {translation.size(-1)} was provided")

        x_rotation = Transformations3D.rotation_matrix_x(rotations[..., 0])
        y_rotation = Transformations3D.rotation_matrix_y(rotations[..., 1])
        z_rotation = Transformations3D.rotation_matrix_z(rotations[..., 2])

        dimensions_to_add = list(rotations.size()[:-1])
        # Applies zxy rotation orders
        rotation = torch.matmul(x_rotation, z_rotation)
        rotation = torch.matmul(y_rotation, rotation)
        matrix = torch.zeros(dimensions_to_add + [4, 4], dtype=torch.float32, device=rotation.device)
        matrix[..., :3, :3] = rotation
        matrix[..., :3, 3] = translation
        matrix[..., 3, 3] = 1.0

        return matrix

    @staticmethod
    def recover_rotation_translation(homogeneous_matrix: torch.Tensor):
        '''
        Recovers rotations and translations from the homogeneous matrix.
        Assumes the homogeneous matrix is built by combining the rotation in z->x->y order

        :param homogeneous_matrix: (..., 4, 4) tensor with homogeneous transformation matrix
        :return (..., (x_degrees, y_degrees, z_degrees)) tensor rotation angles in radians
                (..., (x, y, z)) tensor with translation vector
        '''

        eps = 1e-6
        # We do not consider the solution pi - x_angle because our camera x rotation lies in -pi/2, pi/2
        x_angle = -torch.asin(homogeneous_matrix[..., 1, 2])
        # Case with x_cos == 0 should be treated with special case. In our data however we assume x_angle is never
        # vertical. To avoid exceptions we use a small
        x_cos = torch.cos(x_angle)
        y_sin = homogeneous_matrix[..., 0, 2] / (x_cos + eps)
        y_cos = homogeneous_matrix[..., 2, 2] / (x_cos + eps)

        z_sin = homogeneous_matrix[..., 1, 0] / (x_cos + eps)
        z_cos = homogeneous_matrix[..., 1, 1] / (x_cos + eps)

        # Recovers the angles from sin and cos
        y_angle = torch.atan2(y_sin, y_cos)
        z_angle = torch.atan2(z_sin, z_cos)

        rotations = torch.stack([x_angle, y_angle, z_angle], dim=-1)
        translations = homogeneous_matrix[..., :3, 3]

        return rotations, translations

    @staticmethod
    def look_at(point: torch.Tensor, camera_position: torch.Tensor):
        '''
        Obtains the rotation parameters for the camera at a given position to look at the given point

        :param point: (3) tensor with point to look at
        :param camera_position: (3) tensor with camera position
        :return: (3) tensor with camera x, y, z rotation
        '''

        difference = point - camera_position
        distance = torch.sqrt((difference ** 2).sum(dim=-1))

        # x = psi
        # y = theta
        # z = phi

        sen_psi = difference[1] / distance
        psi = torch.arcsin(sen_psi)  # Keep the solution in [-pi/2, + pi/2]

        cos_psi = torch.sqrt(1 - (sen_psi ** 2))
        sen_theta = -(difference[0]) / (distance * cos_psi + 1e-6)
        cos_theta = -(difference[2]) / (distance * cos_psi + 1e-6)

        theta = torch.arcsin(sen_theta)
        if cos_theta < 0.0:
            theta = math.pi - theta  # If the respective cosine is not positive then the solution cannot be the one in [-pi/2, + pi/2]
                                     # and we use the other

        tan_phi = sen_theta / (cos_theta * sen_psi + 1e-6)
        phi = torch.arctan(tan_phi)
        sen_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)

        # The y axis when rotated must have the z direction positive, otherwise camera is upside down
        if sen_theta * sen_phi + cos_theta * sen_psi * cos_phi < 0.0:
            phi = phi + math.pi
            sen_phi = sen_phi * -1
            cos_phi = cos_phi * -1

        rotations = torch.cat([psi.unsqueeze(0), theta.unsqueeze(0), phi.unsqueeze(0)])

        return rotations


if __name__ == "__main__":

    rotations = torch.tensor([np.pi/2, 1.0, 1.0])
    translations = torch.tensor([1.0, 2.0, 3.0])

    homogeneous_matrix = Transformations3D.homogeneous_rotation_translation(rotations, translations)

    recovered_rotations, recovered_translations = Transformations3D.recover_rotation_translation(homogeneous_matrix)

    pass