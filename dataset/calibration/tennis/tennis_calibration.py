import numpy as np

from dataset.calibration.tennis.field_calibrator import FieldCalibrator

x_singles = 4.115
x_doubles = 5.485

y_small = 6.40
y_large = 11.885

# World coordinates for tennis court keypoints
world_points = [
    (-x_doubles, y_large, 0),  # Bottom line
    (-x_singles, y_large, 0),
    (x_singles, y_large, 0),
    (x_doubles, y_large, 0),
    (-x_singles, y_small, 0),  # bottom mid
    (0.0, y_small, 0),
    (x_singles, y_small, 0),
    (-x_doubles, 0.0, 0),  # central line
    (-x_singles, 0.0, 0),
    (0.0, 0.0, 0),
    (x_singles, 0.0, 0),
    (x_doubles, 0.0, 0),
    (-x_singles, -y_small, 0),  # front mid
    (0.0, -y_small, 0),
    (x_singles, -y_small, 0),
    (-x_doubles, -y_large, 0),  # Front line
    (-x_singles, -y_large, 0),
    (x_singles, -y_large, 0),
    (x_doubles, -y_large, 0),
]

world_points = np.array(world_points, dtype=np.float32)

frame_points = [
    (766.5, 486.6),
    (919.1, 486.5),
    (1828.3, 486.5),
    (1983.2, 486.3),
    (869.1, 596.2),
    (1373.3, 597.6),
    (1879.8, 597.5),
    (598.5, 758.0),
    (793.1, 760.7),
    (1374.8, 765.8),
    (1959.0, 765.7),
    (2157.5, 763.0),
    (689.4, 987.0),
    (1376.8, 987.0),
    (2066.3, 992.3),
    (283.3, 1262.6),
    (563.9, 1259.1),
    (2200.6, 1269),
    (2491.3, 1275.6),
]

frame_points = np.array(frame_points, dtype=np.float32)

# Guessed range for the focal
focal_low = 604
focal_mid = 1028
focal_high = 2687

image_size = (2560, 1440)


if __name__ == "__main__":

    c2w_rotations, c2w_translations, focal_length = FieldCalibrator.calibrate_camera(world_points, frame_points, image_size)

    """noises = [0] #, 1, 2, 4, 8, 16, 32, 64]
    for current_noise_std in noises:
        print(f"= Noise {current_noise_std}")

        current_world_points = world_points[None, ...]
        current_frame_points = frame_points[None, ...]

        current_noise = np.random.normal(size=current_frame_points.shape) * current_noise_std
        current_frame_points = current_frame_points + current_noise
        current_frame_points = current_frame_points.astype(np.float32)

        camera_matrix_guess = np.asarray([[1.0, 0.0, 1280], [0.0, 1.0, 720], [0.0, 0.0, 1.0]])
        ret, cameraMatrix, distortion_coeffs, rvecs, tvecs, intrinsic_std, extrinsic_std, errors = cv.calibrateCameraExtended(current_world_points, current_frame_points, image_size, camera_matrix_guess, None, flags=cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_PRINCIPAL_POINT
                                                                                                                              | cv.CALIB_FIX_K1 | cv.CALIB_FIX_K2 | cv.CALIB_FIX_K3 | cv.CALIB_FIX_K4 | cv.CALIB_FIX_K5 | cv.CALIB_FIX_K6)
        rotation_matrix = cv.Rodrigues(rvecs[0])[0]
        w2c_matrix = torch.zeros((4, 4), dtype=torch.float32)
        rotation_matrix = torch.from_numpy(rotation_matrix)
        w2c_matrix[:3, :3] = rotation_matrix
        w2c_matrix[3, 3] = 1
        w2c_matrix[:3, 3] = torch.from_numpy(tvecs[0][:, 0])

        c2w_matrix = w2c_matrix.inverse()

        c2w_recovered_angles, c2w_recovered_translations = Transformations3D.recover_rotation_translation(c2w_matrix)
        w2c_recovered_angles, w2c_recovered_translations = Transformations3D.recover_rotation_translation(w2c_matrix)

        opengl_conversion_matrix = Transformations3D.homogeneous_rotation_translation(torch.as_tensor([np.pi, 0.0, 0.0]), torch.as_tensor([0.0, 0.0, 0.0]))
        c2w_opengl_matrix = torch.matmul(c2w_matrix, opengl_conversion_matrix)

        c2w_recovered_angles *= 180 / np.pi
        c2w_opengl_recovered_angles, c2w_opengl_recovered_translations = Transformations3D.recover_rotation_translation(c2w_opengl_matrix)

        c2w_opengl_recovered_angles *= 180 / np.pi
        print(f"- Angles: {c2w_opengl_recovered_angles}")
        print(f"- Translations: {c2w_opengl_recovered_translations}")

        #cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv.decomposeProjectionMatrix(projection_matrix)
"""
    pass