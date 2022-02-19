import functools
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.autoencoder_models.layers.cyclegan_resnet_block import CycleGanResnetBlock


class DecoderV5(nn.Module):
    '''
    CycleGAN decoder with bilinear upsampling layers and variable bottleneck layer
    '''

    def __init__(self, model_config: Dict):
        super(DecoderV5, self).__init__()

        # Gets the number of input features
        self.in_features = model_config["input_features"]
        self.bottleneck_features = model_config["bottleneck_features"]
        self.bottleneck_blocks = model_config["bottleneck_blocks"]
        self.downsampling_layers_count = model_config["downsampling_layers_count"]

        norm_layer = nn.BatchNorm2d
        use_dropout = False
        padding_type = 'reflect'

        initial_features_count = self.bottleneck_features // (2 ** sum(self.downsampling_layers_count))

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # Computes the current multiplier for the number of features at the current level
        mult = 2 ** sum(self.downsampling_layers_count)

        # Creates the bottleneck layers
        bottleneck_blocks_list = []
        # Adds the bottleneck blocks
        for _ in range(self.bottleneck_blocks):
            # Adds a single residual block before upsampling
            bottleneck_blocks_list += [CycleGanResnetBlock(initial_features_count * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.bottleneck_blocks = nn.Sequential(*bottleneck_blocks_list)

        self.upsample_blocks = nn.ModuleList()
        for downsampling_layer_set_idx, current_downsampling_layers_count in enumerate(reversed(self.downsampling_layers_count)):
            current_upsampling_layers = []
            # Creates the upsampling layers using bilinear interpolation
            for i in range(current_downsampling_layers_count):

                # If skip connections are expected, double the number of input features
                input_features_multiplier = 1
                if i == 0 and downsampling_layer_set_idx > 0:
                    input_features_multiplier = 2

                current_upsampling_layers += [nn.UpsamplingBilinear2d(scale_factor=2),
                                              nn.Conv2d(initial_features_count * mult * input_features_multiplier, int(initial_features_count * mult / 2), kernel_size=3, padding=1, padding_mode="reflect", bias=False),
                                              norm_layer(int(initial_features_count * mult / 2)),
                                              nn.ReLU(True)]

                # Halves the number of features for the next layer
                mult = mult // 2

            self.upsample_blocks.append(nn.Sequential(*current_upsampling_layers))

        # Creates the final convolution layer
        self.final_convolutions = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(initial_features_count, self.in_features, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

    def forward(self, encoded_observations: torch.Tensor):
        '''

        :param encoded_observations: (batch_size, bottleneck_features, bottleneck_height, bottleneck_width) tensor with bottleneck features at bottleneck resolution
        :return: (batch_size, in_features, height, width) tensor with reconstructed observations

        '''

        # Starts from the lower resolution features
        current_features = encoded_observations[-1]
        current_features = self.bottleneck_blocks(current_features)
        for upsample_block_idx, current_upsample_block in enumerate(self.upsample_blocks):
            current_features = current_upsample_block(current_features)

            # If this was not the last upsample block
            if upsample_block_idx != len(self.upsample_blocks) - 1:
                # Gets skip connections. Skip connections are not activated, so apply activation
                current_skip_features = encoded_observations[-upsample_block_idx - 2]
                # TODO Activating skip connections after the sampling operation is different from what done in the usual VAE
                # where sampled features are not activated. Remove this activation
                current_skip_features = F.relu(current_skip_features, inplace=False)

                current_features = torch.cat([current_features, current_skip_features], dim=-3)

        reconstructed_observations = self.final_convolutions(current_features)
        return reconstructed_observations



