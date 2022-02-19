import torch
import torch.nn as nn


class PositionalStyleEncoder(nn.Module):
    '''
    Adaptive instance normalization module with style affine transformation
    '''

    def __init__(self, positional_features: int, style_features: int, layers_width: int, layers_count: int):
        '''

        :param positional_features: Number of features for the position
        :param style_features: Number of features for the style
        :param layers_width: Width of each layer in the backbone
        :param layers_count: Number of layers in the backbone
        '''
        super(PositionalStyleEncoder, self).__init__()

        # Instantiates the backbone layers
        backbone_layers = []
        current_input_size = style_features + positional_features
        for layer_idx in range(layers_count):

            backbone_layers.append(nn.Linear(current_input_size, layers_width))
            current_input_size = layers_width

        self.backbone_layers = nn.Sequential(*backbone_layers)

    def forward(self, ray_positions: torch.Tensor, style: torch.Tensor):
        '''

        :param ray_positions: (batch_size, positional_features_count) tensor representing input vectors
        :param style: (batch_size, style_features_count) tensor representing styles to apply to the input
        :return:
        '''

        input = torch.cat([ray_positions, style], dim=-1)

        output = self.backbone_layers(input)
        return output

