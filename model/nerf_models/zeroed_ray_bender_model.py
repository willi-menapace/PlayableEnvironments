from typing import Dict, Tuple

import torch
import torch.nn as nn


class ZeroedRayBender(nn.Module):
    '''
    A model for bending rays that never bends them
    '''

    def __init__(self, config: Dict, model_config: Dict):
        super(ZeroedRayBender, self).__init__()

        self.config = config
        self.model_config = model_config

    def set_step(self, current_step: int):
        '''
        Sets the current step to the specified value
        :param current_step:
        :return:
        '''

        # No need to do anything
        pass

    def forward(self, ray_positions: torch.Tensor, deformation: torch.Tensor, video_indexes: torch.Tensor=None) -> Tuple[torch.Tensor]:
        '''

        :param ray_positions: (..., 3) tensor with ray positions
        :param deformation: (..., deformation_features_count) tensor with deformation encodings for each position
        :param video_indexes: (...) tensor of integers representing indexes of each video in the dataset. Indexes are inferred if None
        :return: (..., 3) tensor with position displacements
        '''

        return ray_positions * 0.0


def model(config, model_config):
    '''
    Instantiates a nerf model with the given parameters

    :param config:
    :param model_config:
    :return:
    '''
    return ZeroedRayBender(config, model_config)

