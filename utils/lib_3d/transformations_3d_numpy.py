import numpy as np

from utils.lib_3d.transformations_3d import Transformations3D


class Transformations3DNumpy:

    @staticmethod
    def rotation_matrix_x(radians) -> np.ndarray:
        '''
        Obtains a 3x3 rotation matrix on the x axis (vertical rotation)
        :param radians: degrees of which rotate on the x axis. Positive rotation is clockwise
        :return:
        '''
        cos = np.cos(radians)
        sin = np.sin(radians)

        elements = [
            [1.0, 0.0, 0.0],
            [0.0, cos, -sin],
            [0.0, sin, cos],
        ]

        return np.asarray(elements, dtype=np.float)

    @staticmethod
    def rotation_matrix_y(radians) -> np.ndarray:
        '''
        Obtains a 3x3 rotation matrix on the y axis (horizontal rotation)
        :param radians: degrees of which rotate on the x axis. Positive rotation is clockwise
        :return:
        '''
        cos = np.cos(radians)
        sin = np.sin(radians)

        elements = [
            [cos, 0.0, sin],
            [0.0, 1.0, 0.0],
            [-sin, 0.0, cos],
        ]

        return np.asarray(elements, dtype=np.float)

    @staticmethod
    def homogeneous_rotation_translation(rotations: np.ndarray, translation: np.ndarray) -> np.ndarray:
        '''
        Obtains a 4x4 rotation and translation matrix
        :param rotations: (x_degrees, y_degrees) rotation angles in radians
        :param translation: (x, y, z) translation vector
        :return:
        '''

        x_rotation = Transformations3D.rotation_matrix_x(rotations[0])
        y_rotation = Transformations3D.rotation_matrix_y(rotations[1])

        rotation = np.matmul(y_rotation, x_rotation)
        matrix = np.zeros((4, 4), dtype=np.float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        matrix[3, 3] = 1.0

        return matrix
