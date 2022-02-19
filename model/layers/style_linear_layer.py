import math

import torch
import torch.nn as nn


class StyleLinearLayer(nn.Module):
    '''
    Linear layer applying style similarly to StyleGANv2 style blocks
    '''

    def __init__(self, in_features: int, out_features: int, style_features: int):
        super(StyleLinearLayer, self).__init__()

        self.style_features = style_features

        self.weight = nn.Parameter(torch.zeros((out_features, in_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((out_features,), dtype=torch.float32))

        self.reset_parameters()

        # Computes scale and bias from the positional style
        self.affine_transform = nn.Linear(style_features, out_features * 2)
        # Makes scale parameters biased 1 and bias parameters biased 0
        self.affine_transform.bias.data[:out_features] = 1
        self.affine_transform.bias.data[out_features:] = 0

    def reset_parameters(self) -> None:
        '''
        Initializes parameters as in torch.nn.Linear
        :return:
        '''
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def compute_weights(self, scale: torch.Tensor) -> torch.Tensor:
        '''
        Computes the weights for the linear transformation to apply to the input

        :param scale: (batch_size, out_features) tensors with scales to apply to the input
        :return: (out_features, in_features) tensor with scaled weights
        '''

        scale = scale.unsqueeze(-1)  # (batch_size, out_features, 1)

        w1 = self.weight * scale  # (batch_size, out_features, in_features)
        normalization_factor = torch.rsqrt(w1.pow(2).sum(dim=-1, keepdim=True))
        w2 = w1 / normalization_factor  # (batch_size, out_features, in_feautures)

        return w2

    def forward(self, input: torch.Tensor, style: torch.Tensor):
        '''

        :param input: (batch_size, in_features) tensor representing input vectors
        :param style: (batch_size, style_features_count) tensor representing styles to apply to the input
        :return:
        '''

        # Applies affine transformation to produce the style
        encoded_style = self.affine_transform(style)
        # Separates scale from bias
        scale, bias = encoded_style.chunk(2, 1)

        scaled_weights = self.compute_weights(scale)  # (batch_size, out_features, in_features)

        # Applies the linear layer. Each input is multiplied with its weights
        input = input.unsqueeze(-1)  # (batch_size, in_features, 1)
        product = torch.matmul(scaled_weights, input).squeeze(-1)  # (batch_size, out_features)
        output = product + self.bias.unsqueeze(0)

        return output

