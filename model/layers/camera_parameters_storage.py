from typing import Tuple

import torch
import torch.nn as nn

from model.layers.indexed_storage import IndexedStorage


class CameraParametersStorage(nn.Module):
    '''
    A module for storing optimizable camera parameters for each frame
    '''

    def __init__(self, storage_size: int, cameras_count: int):
        '''

        :param storage_size: Maximum number of frames to memorize
        :param cameras_count: Number of cameras
        '''
        super(CameraParametersStorage, self).__init__()

        self.cameras_count = cameras_count
        self.storage_size = storage_size
        self.camera_adjusted_storage_size = self.storage_size * cameras_count
        self.features_count = 7  # 3 rotations, 3 translations, 1 focal length

        # The storage stores the parameters for all the frames of the first camera and then all parameters for the frames
        # of each successive camera
        self.storage = IndexedStorage(self.camera_adjusted_storage_size, self.features_count, init_mode="zero")

    def forward(self, frame_indexes: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Retrieves camera parameter offsets for the given frame indexes.
        Parameters are available for training only.
        Returns 0 if the model is not in training mode

        :param frame_indexes: (...) tensor of integers with frame indexes of the parameters to retrieve
        :return: (..., cameras_count, 3) tensor with rotation offset parameters
                 (..., cameras_count, 3) tensor with translation offset parameters
                 (..., cameras_count) tensor with focal length offsets
        '''

        # Translates the frame indexes to the storage index associated to the frame of each camera
        camera_adjusted_frame_indexes = []
        for camera_idx in range(self.cameras_count):
            # The first camera has the original frame indexes, the ones for the second are offsetted by storage_size
            camera_adjusted_frame_indexes.append(frame_indexes + camera_idx * self.storage_size)
        camera_adjusted_frame_indexes = torch.stack(camera_adjusted_frame_indexes, dim=-1)

        # If training retrieves the camera parameter offsets
        if self.training:
            outputs = self.storage(camera_adjusted_frame_indexes)
            rotation_offsets = outputs[..., :3]
            translation_offsets = outputs[..., 3:6]
            focal_offsets = outputs[..., 6:].squeeze(-1)
        # Otherwise return 0
        else:
            initial_dimensions = list(camera_adjusted_frame_indexes.size())
            rotation_offsets = torch.zeros(initial_dimensions + [3], dtype=torch.float32, device=frame_indexes.device)
            translation_offsets = torch.zeros(initial_dimensions + [3], dtype=torch.float32, device=frame_indexes.device)
            focal_offsets = torch.zeros(initial_dimensions, dtype=torch.float32, device=frame_indexes.device)

        # Scales outputs for ease of optimization since rotations are typically in e-3 scale, translations in e-2 scale and
        # focals in e+0 scale
        translation_offsets = translation_offsets * 10
        focal_offsets = focal_offsets * 1000

        return rotation_offsets, translation_offsets, focal_offsets