from typing import Sequence

import numpy as np
import torch
import cv2 as cv

from utils.lib_3d.transformations_3d import Transformations3D


class FieldCalibrator:
    '''
    Class with utilities for camera calibration from planar fields
    '''

    @staticmethod
    def calibrate_camera(world_points: np.ndarray, image_points: np.ndarray, image_size: Sequence[int]):
        '''
        Performs calibration of the camera based on correspondences from a known planar field.
        Camera is assumed to adhere to OpenGL standard with camera facing in the negative z direction.

        The world system has origin in the origin of the planar field
        Y goes up in the field
        X goes right in the field
        Z goes upwards from the field (right handed coordinate system)

        :param world_points: (points_count, 3) array with real world points. Must have z=0
        :param image_points: (points_count, 2) array with correspondences on the image plane of the real world points.
                                               Points are expressed in coordinates from the top left camera angle
        :param image_size: (width, height) of the image
        :return: (3) array with camera to world rotations
                 (3) array with camera to world translations
                 float with camera focal length in pixels
        '''

        # Puts points in the format expected by OpenCV
        world_points = world_points[None, ...].astype(np.float32)
        image_points = image_points[None, ...].astype(np.float32)

        center_point_x = image_size[0] // 2
        center_point_y = image_size[1] // 2

        # The initial camera guess has the central point in the center of the image and same focal length for x and y axes
        camera_matrix_guess = np.asarray([[1.0, 0.0, center_point_x], [0.0, 1.0, center_point_y], [0.0, 0.0, 1.0]])

        fitting_error, camera_matrix, distortion_coefficients, w2c_rotation_vectors, w2c_translation_vectors, intrinsic_std, extrinsic_std, errors = cv.calibrateCameraExtended(world_points, image_points, image_size, camera_matrix_guess, None, flags=cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_PRINCIPAL_POINT | cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6)

        # Focal length must be the same for x and y
        assert(camera_matrix[0, 0] == camera_matrix[1, 1])

        # Computes the rotation matrix
        w2c_rotation_matrix = cv.Rodrigues(w2c_rotation_vectors[0])[0]
        w2c_rotation_matrix = torch.from_numpy(w2c_rotation_matrix)

        # Creates the translation w2c matrix
        w2c_matrix = torch.zeros((4, 4), dtype=torch.float32)
        w2c_matrix[:3, :3] = w2c_rotation_matrix
        w2c_matrix[3, 3] = 1
        w2c_matrix[:3, 3] = torch.from_numpy(w2c_translation_vectors[0][:, 0])

        c2w_matrix = w2c_matrix.inverse()

        # Matrix for conversion to the OpenGL camera system from the OpenCV camera system.
        # OpenCV camera faces the positive Z, OpenGL the negative Z, so a 180 degree rotation on the x axis is necessary
        # This also inverts the Y coordinates
        opengl_conversion_matrix = Transformations3D.homogeneous_rotation_translation(torch.as_tensor([np.pi, 0.0, 0.0]), torch.as_tensor([0.0, 0.0, 0.0]))
        # Covnerts the matrix to OpenGL conversion
        c2w_opengl_matrix = torch.matmul(c2w_matrix, opengl_conversion_matrix)

        c2w_rotations, c2w_translations = Transformations3D.recover_rotation_translation(c2w_opengl_matrix)

        return c2w_rotations.numpy(), c2w_translations.numpy(), float(camera_matrix[0, 0])

