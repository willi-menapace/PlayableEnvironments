from typing import Dict, Callable, List, Tuple

import torch
import torch.nn as nn

from model.autoencoder_models.decoder_v5 import DecoderV5
from model.autoencoder_models.encoder_v3 import EncoderV3
from utils.tensor_folder import TensorFolder
from utils.tensor_splitter import TensorSplitter


class AutoencoderV7(nn.Module):
    '''
    A variational autoencoder with support for multiple resolutions and skip connections
    '''

    def __init__(self, model_config: Dict):
        super(AutoencoderV7, self).__init__()

        # Signals that the current autoencoder is variational
        model_config["variational"] = True
        self.model_config = model_config

        self.encoder = EncoderV3(model_config)
        self.decoder = DecoderV5(model_config)

    def get_features_count_by_layer(self) -> List[int]:
        '''
        Computes the number of features that are output at each resolution
        :return: List with the number of features corresponding to the encoded at each resolution
        '''
        return self.encoder.features_count_by_layer

    def sample_list(self, distributions: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
        '''
        Samples tensors from the given distributions.
        :param distributions: List of (mean, log_variance) tensors with matching shape. See 'sample' for details
        :return: A list of outputs from the 'sample' function, one for each input distribution.
        '''

        results = [self.sample(*current_distribution) for current_distribution in distributions]
        return results

    def sample(self, mean: torch.Tensor, log_variance: torch.Tensor):
        '''
        Samples from the posterior distribution with given mean and variance

        :param mean: (..., features_count, height, width) tensor with posterior mean
        :param log_variance: (..., features_count, height, width) tensor with posterior variance
        :return: (..., features_count, height, width) tensor with points sampled from the posterior
        '''

        noise = torch.randn(mean.size(), dtype=mean.dtype, device=mean.device)
        sampled_points = noise * torch.sqrt(torch.exp(log_variance)) + mean

        return sampled_points

    def forward(self, *args, mode="encoder+decoder", **kwargs):
        '''

        :param args: Positional arguments to pass to the target forward mode
        :param mode: Mode to use for the forward
                     "encoder+decoder" reconstructs observations starting from observations
                     "encoder" obtains encoded image features starting from the observations
                     "decoder" reconstructs observations starting from encoded image features

        :param kwargs: Keyword arguments to pass to the target forward mode
        :return: Output of the chosen forward mode
        '''

        return_value = None
        if mode == "encoder+decoder":
            return_value = self.forward_complete(*args, **kwargs)
        elif mode == "encoder":
            return_value = self.forward_encoder(*args, **kwargs)
        elif mode == "decoder":
            return_value = self.forward_decoder(*args, **kwargs)
        else:
            raise Exception(f"Unknown forward mode '{mode}'")

        return return_value

    def forward_complete(self, observations: torch.Tensor, bottleneck_transform: Callable=None):
        '''

        :param observations: (..., in_features, height, width) tensor with input observations
        :param bottleneck_transform: Transformation to apply to the bottleneck features. Not supported.
        :return: dictionary with fields for
                 (..., in_features, height, width) tensor with reconstructed observations
                 [(..., bottleneck_features_i * 2, bottleneck_height_i, bottleneck_width_i) for i in resolutions_count]
                                                                                     tensor with bottleneck features at the current bottleneck resolution.
                                                                                     Output feature represent a distribution with the first half of the features representing mean
                                                                                     and the second half log variance of the distribution
        '''

        flat_observations, initial_dimensions = TensorFolder.flatten(observations, -3)

        encoded_observations = self.encoder(flat_observations)

        plain_encoded_observations = encoded_observations

        splitted_encoded_observations = TensorSplitter.split(encoded_observations, dim=-3, factor=2)
        sampled_encoded_observations = self.sample_list(splitted_encoded_observations)

        # Transforms the bottleneck features if required
        if bottleneck_transform is not None:
            print("Warning: Bottleneck transformation not supported by multiresolution autoencoders and will be discarded")

        flat_reconsturcted_observatiosn = self.decoder(sampled_encoded_observations)
        folded_reconstructed_observations = TensorFolder.fold(flat_reconsturcted_observatiosn, initial_dimensions)
        folded_plain_encoded_observations = TensorFolder.fold_list(plain_encoded_observations, initial_dimensions)

        results = {
            "reconstructed_observations": folded_reconstructed_observations,
            "encoded_observations": folded_plain_encoded_observations
        }

        return results

    def forward_encoder(self, observations: torch.Tensor):
        '''

        :param observations: (..., in_features, height, width) tensor with input observations
        :return: [(..., bottleneck_features_i * 2, bottleneck_height_i, bottleneck_width_i) for i in resolutions_count]
                                                                                     tensor with bottleneck features at the current bottleneck resolution.
                                                                                     Output feature represent a distribution with the first half of the features representing mean
                                                                                     and the second half log variance of the distribution
        '''

        flat_observations, initial_dimensions = TensorFolder.flatten(observations, -3)

        encoded_observations = self.encoder(flat_observations)

        folded_encoded_observations = TensorFolder.fold_list(encoded_observations, initial_dimensions)

        return folded_encoded_observations

    def forward_decoder(self, encoded_observations: torch.Tensor):
        '''

        :param encoded_observations: [(..., bottleneck_features_i, bottleneck_height_i, bottleneck_width_i) for i in resolutions_count]
                                                                                     tensor with bottleneck features at the current bottleneck resolution.
                                                                                     Features are assumed to be sampled already

        :return: (..., in_features, height, width) tensor with reconstructed observations
        '''

        flat_encoded_observations, initial_dimensions = TensorFolder.flatten_list(encoded_observations, -3)

        reconstructed_observations = self.decoder(flat_encoded_observations)

        folded_reconstructed_observations = TensorFolder.fold(reconstructed_observations, initial_dimensions)

        return folded_reconstructed_observations


def model(model_config):
    '''
    Istantiates a decoder with the given parameters
    :param model_config:
    :return:
    '''
    return AutoencoderV7(model_config)


