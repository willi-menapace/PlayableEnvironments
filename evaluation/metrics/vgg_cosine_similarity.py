import torch
import torch.nn as nn

from model.layers.vgg import Vgg19
from utils.tensor_folder import TensorFolder


class VGGCosineSimilarity(nn.Module):

    def __init__(self):
        super(VGGCosineSimilarity, self).__init__()

        self.vgg = Vgg19()
        self.vgg = self.vgg.cuda()

    def forward(self, reference_observations: torch.Tensor, generated_observations: torch.Tensor, range=1.0) -> torch.Tensor:
        '''
        Computes the VGG Cosine Similarity between the reference and the generated observations

        :param reference_observations: (bs, observations_count, cameras_count, channels, height, width) tensor with reference observations
        :param generated_observations: (bs, observations_count, cameras_count, channels, height, width) tensor with generated observations
        :param range: The maximum value used to represent each pixel
        :return: (bs, observations_count) tensor with ssim for each observation
        '''

        cameras_count = reference_observations.size(2)
        if cameras_count != 1:
            raise Exception(f"Expected 1 cameras, but the observations have {cameras_count} cameras")
        reference_observations = reference_observations[:, :, 0]
        generated_observations = generated_observations[:, :, 0]

        # Normalizes the observations for VGG
        reference_observations = reference_observations / range
        generated_observations = generated_observations / range
        normalization_mean = 0.5
        normaliation_std = 0.5
        normalization_eps = 1e-6
        reference_observations = (reference_observations - normalization_mean) / (normaliation_std + normalization_eps)
        generated_observations = (generated_observations - normalization_mean) / (normaliation_std + normalization_eps)

        bs = reference_observations.size(0)
        observations_count = reference_observations.size(1)

        flattened_reference_observations, initial_dimensions = TensorFolder.flatten(reference_observations)
        flattened_generated_observations, _ = TensorFolder.flatten(generated_observations)

        # Computes vgg features
        flattened_reference_features = self.vgg(flattened_reference_observations)
        flattened_generated_observations = self.vgg(flattened_generated_observations)

        features_count = len(flattened_reference_features)

        similarities = torch.zeros((bs, observations_count), dtype=torch.float).cuda() # Accumulator for the similarities
        cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
        for current_reference_feature, current_generated_feature in zip(flattened_reference_features, flattened_generated_observations):
            # Flattens the features
            current_reference_feature = current_reference_feature.reshape((bs * observations_count, -1))
            current_generated_feature = current_generated_feature.reshape((bs * observations_count, -1))

            similarities += TensorFolder.fold(cosine_similarity(current_reference_feature, current_generated_feature), initial_dimensions)
        similarities /= features_count  # Computes the mean similarity

        return similarities


