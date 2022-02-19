import torch
import torch.nn as nn


class StyleLinearLayerV2(nn.Module):
    '''
    Linear layer applying style similarly to StyleGANv2 style blocks
    '''

    def __init__(self, in_features: int, out_features: int, style_features: int):
        super(StyleLinearLayerV2, self).__init__()

        self.style_features = style_features

        self.linear_layer = nn.Linear(in_features, out_features)

        # Computes scale and bias from the positional style
        self.affine_transform = nn.Linear(style_features, in_features)
        # Makes scale parameters biased 1
        self.affine_transform.bias.data[:] = 1

    def forward(self, input: torch.Tensor, style: torch.Tensor):
        '''

        :param input: (batch_size, in_features) tensor representing input vectors
        :param style: (batch_size, style_features_count) tensor representing styles to apply to the input
        :return:
        '''

        # Applies affine transformation to produce the style
        scale = self.affine_transform(style)  # (batch_size, in_features)
        # Scales the input
        scaled_input = input * scale

        # Produces the output using the linear layer
        output = self.linear_layer(scaled_input)

        return output

