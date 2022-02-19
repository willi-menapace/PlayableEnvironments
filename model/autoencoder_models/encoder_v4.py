import functools
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.autoencoder_models.layers.cyclegan_resnet_block import CycleGanResnetBlock
from utils.tensor_splitter import TensorSplitter


class EncoderV4(nn.Module):
    '''
    CycleGAN encoder with downsampling layers implemented as average pooling and bottlenect
    '''

    def __init__(self, model_config: Dict):
        super(EncoderV4, self).__init__()

        # Gets the number of input features
        self.in_features = model_config["input_features"]
        self.bottleneck_features = model_config["bottleneck_features"]
        self.bottleneck_blocks = model_config["bottleneck_blocks"]
        self.downsampling_layers_count = model_config["downsampling_layers_count"]

        # The features output by each downsampling layer
        self.features_count_by_layer = []

        # Checks whether the encoder is to be used in a variational pipeline
        self.variational = False
        if "variational" in model_config:
            self.variational = model_config["variational"]
        if not self.variational:
            raise Exception("Only the variaitonal model is supported")

        norm_layer = nn.BatchNorm2d
        use_dropout = False
        padding_type = 'reflect'

        initial_features_count = self.bottleneck_features // (2 ** sum(self.downsampling_layers_count))

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Creates the initial convolution
        self.initial_convolution = nn.Sequential(nn.ReflectionPad2d(3),
                                             nn.Conv2d(self.in_features, initial_features_count, kernel_size=7, padding=0, bias=use_bias),
                                             norm_layer(initial_features_count),
                                             nn.ReLU(True))

        downsampling_layers_module_lists = []
        cumulative_downsampling_layer = 0
        for downsampling_layer_set_idx, current_downsampling_layers_count in enumerate(self.downsampling_layers_count):
            # Creates the downsampling layers
            layer_list = []
            for i in range(current_downsampling_layers_count):  # add downsampling layers

                mult = 2 ** cumulative_downsampling_layer
                layer_list += [nn.Conv2d(initial_features_count * mult, initial_features_count * mult * 2, kernel_size=3, padding=1, padding_mode="reflect", bias=use_bias),
                               norm_layer(initial_features_count * mult * 2),
                               nn.ReLU(True),
                               nn.AvgPool2d(kernel_size=2)]

                cumulative_downsampling_layer += 1

            # Adds the bottleneck layers after each downsampling layer
            for bottleneck_idx in range(self.bottleneck_blocks):  # add ResNet blocks

                out_dim = initial_features_count * mult * 2
                # if the encoder is variational, double the number of outputs to include log variance
                if bottleneck_idx == self.bottleneck_blocks - 1 and self.variational:
                    out_dim *= 2

                layer_list.append(
                                CycleGanResnetBlock(initial_features_count * mult * 2, padding_type=padding_type,
                                norm_layer=norm_layer,
                                use_dropout=use_dropout,
                                use_bias=use_bias, out_dim=out_dim))


            # Saves the number of features output by the current layer
            self.features_count_by_layer.append(initial_features_count * mult * 2)
            downsampling_layers_module_lists.append(layer_list)

        self.downsampling_layers = nn.ModuleList([nn.Sequential(*current_layers) for current_layers in downsampling_layers_module_lists])

    def forward(self, observations: torch.Tensor):
        '''

        :param observations: (batch_size, in_features, height, width) tensor with input observations
        :return: (batch_size, bottleneck_features, bottleneck_height, bottleneck_width) tensor with bottleneck features at bottleneck resolution.
                 if the encoder is variational, then the output features count is double to output mean and log variance of each feature
        '''

        current_features = self.initial_convolution(observations)
        encoded_features = []
        for current_downsampling_layer in self.downsampling_layers:
            current_features = current_downsampling_layer(current_features)
            encoded_features.append(current_features)
            current_mean, _ = TensorSplitter.split(current_features, dim=-3, factor=2)

            # The output features are unactivated, so we compute the activation.
            # Cannot do inplace since we also return intermediate, unactivated features
            current_features = F.relu(current_mean, inplace=False)

        return encoded_features



