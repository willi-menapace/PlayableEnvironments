import torch
import torch.nn as nn

from evaluation.metrics.motion_mask import MotionMaskCalculator


class MotionMaskedMSE(nn.Module):

    def __init__(self):
        super(MotionMaskedMSE, self).__init__()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the mean squared error between the reference and the generated observations

        :param reference_observations: (bs, observations_count, cameras_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, cameras_count, channels, height, width) tensor with generated observations
        :return: (bs, observations_count) tensor with MSE for each observation
        '''

        cameras_count = reference_observations.size(2)
        if cameras_count != 1:
            raise Exception(f"Expected 1 cameras, but the observations have {cameras_count} cameras")
        reference_observations = reference_observations[:, :, 0]
        generated_observations = generated_observations[:, :, 0]

        motion_mask = MotionMaskCalculator.compute_frame_difference_motion_mask(reference_observations)
        differences = (reference_observations - generated_observations).pow(2) * motion_mask

        return torch.mean(differences, dim=[2, 3, 4])
