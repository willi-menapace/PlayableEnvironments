from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from model.layers.masked_avg_pool import MaskedAvgPool1d
from model.layers.masked_sequential import MaskedSequential


class DiscriminatorV6(nn.Module):
    '''
    Discriminator for 1d vector sequences based on convolutions
    A DiscriminatorV6 with more capacity
    '''

    def __init__(self, config, model_config: Dict):
        super(DiscriminatorV6, self).__init__()

        # Gets the number of input features
        self.in_features = model_config["input_features"]
        self.layers_count = model_config["layers_count"]
        self.layers_width = model_config["layers_width"]

        all_layers = []
        current_features_count = self.in_features
        for layer_idx in range(self.layers_count):
            all_layers.append(spectral_norm(nn.Conv1d(current_features_count, self.layers_width, kernel_size=5, padding=2, bias=True)))
            all_layers.append(nn.ReLU())
            current_features_count = self.layers_width

        # Averages all predictions
        self.pool = MaskedAvgPool1d(1)
        # Adds a final fc layer. Outputs a single logit for real or fake prediction
        self.linear = spectral_norm(nn.Linear(self.layers_width, 1))

        self.model = MaskedSequential(*all_layers)

    def forward(self, tensor: torch.Tensor, sequence_validity: torch.Tensor):
        '''

        :param tensor: (batch_size, observations_count, input_features) tensor with input observations
                       (batch_size, observations_count) tensor with sequence validity
        :return: (batch_size) tensor with real/fake prediction for each input sequence
        '''

        # Puts the tensor in (batch_size, input_features, observations_count) format required by the convolutions
        tensor = tensor.permute(0, 2, 1)

        # Computes predictions
        predictions = self.model(tensor, sequence_validity)
        pooled_predictions = self.pool(predictions, sequence_validity).squeeze(-1)
        linear_predictions = self.linear(pooled_predictions)
        predictions = linear_predictions.squeeze(-1)

        return predictions


def model(config, model_config):
    '''
    Instantiates a discriminator with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return DiscriminatorV6(config, model_config)



