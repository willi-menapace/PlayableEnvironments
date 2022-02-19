import torch
import torch.nn as nn

import numpy as np


class CycleGanResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, out_dim=None):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(CycleGanResnetBlock, self).__init__()

        if out_dim is None:
            out_dim = dim

        self.conv_block = self.build_conv_block(dim, out_dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, out_dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            out_dim (int)           -- the number of output channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, out_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(out_dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(out_dim)]

        self.residual_connection_convolution = None
        if out_dim != dim:
            self.residual_connection_convolution = nn.Sequential(nn.Conv2d(dim, out_dim, kernel_size=1, bias=use_bias), norm_layer(out_dim))

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""

        residual = x
        if self.residual_connection_convolution is not None:
            residual = self.residual_connection_convolution(x)

        out = residual + self.conv_block(x)  # add skip connections
        return out