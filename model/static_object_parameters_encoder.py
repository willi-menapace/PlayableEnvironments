from typing import Tuple

import torch
import torch.nn as nn


class StaticObjectParametersEncoder(nn.Module):
    '''
    Computes the geometry of the scene
    '''

    def __init__(self, config, model_config):
        '''
        Initializes the model representing the geometry of the scene

        :param config: the configuration file
        '''
        super(StaticObjectParametersEncoder, self).__init__()

        self.config = config
        self.model_config = model_config
        self.objects_count = model_config["objects_count"]

        # (objects_count, 3, 2)
        # Creates tensors for the ranges of rotation and translation
        self.register_buffer("translation_range", torch.tensor(model_config["translation_range"], dtype=torch.float32))
        # (objects_count, 3, 2)
        self.register_buffer("rotation_range", torch.tensor(model_config["rotation_range"], dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor]:
        '''
        Obtains the translations o2w of each scene object represented in the observations

        :param observations: (..., cameras_count, 3, height, width) observations

        :return: (..., 3, objects_count) tensor with rotation parameters from object to world for each object
                 (..., 3, objects_count) tensor with translation parameters from object to world for each object
        '''

        initial_dimensions = list(observations.size())[:-4]
        # All static objects are centered in their position
        all_object_parameters = torch.zeros(initial_dimensions + [6, self.objects_count], device=observations.device)

        # Separates rotations from translations
        rotation_parameters = all_object_parameters[..., :3, :]
        translation_parameters = all_object_parameters[..., 3:, :]

        for object_idx in range(self.objects_count):
            # Brings the values from (-1, +1) to the expected ranges
            rotation_min = self.rotation_range[object_idx, :, 0]
            rotation_max = self.rotation_range[object_idx, :, 1]
            translation_min = self.translation_range[object_idx, :, 0]
            translation_max = self.translation_range[object_idx, :, 1]
            rotation_parameters[..., object_idx] = (rotation_parameters[..., object_idx] + 1) / 2 * (rotation_max - rotation_min) + rotation_min
            translation_parameters[..., object_idx] = (translation_parameters[..., object_idx] + 1) / 2 * (translation_max - translation_min) + translation_min

        return rotation_parameters, translation_parameters


def model(config, model_config):
    '''
    Instantiates a parameters encoder

    :param config:
    :param model_config:
    :return:
    '''

    return StaticObjectParametersEncoder(config, model_config)


