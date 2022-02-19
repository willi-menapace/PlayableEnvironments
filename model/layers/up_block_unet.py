import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers.double_convolution import DoubleConvolutionBlock


class UpBlockUnet(nn.Module):
    """
    Upsampling block.
    """

    def __init__(self, in_features, out_features, upscaling_mode="bilinear", use_skip_connections=True, padding_mode="zeros"):
        '''

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param upscaling_mode: interpolation upscaling mode
        :param use_skip_connections: if True enables support for skip connections
        :param padding_mode: the padding mode to use in the convolutions
        '''

        super(UpBlockUnet, self).__init__()

        self.use_skip_connections = use_skip_connections

        self.upscaling_mode = upscaling_mode
        self.convolutions = DoubleConvolutionBlock(in_features, out_features, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor, skip_connections: torch.Tensor=None):
        '''

        :param x: (batch_size, input_features_count, height, width) tensor with input features.
                                                                    Inputs features must be halved if skip connections are being used
        :param skip_connections: (batch_size, input_features_count / 2, height, width) tensor with skipped input features.
        :return: (batch_size, output_features_count, 2 * height, 2 * width) tensor with input features
        '''

        # Merge skip connections with the input if required
        if self.use_skip_connections:
            if skip_connections is None:
                raise Exception("The upsampling block requires skip connections but they were not provided")

            x = torch.cat([x, skip_connections], dim=-3)

        # Perform upscaling
        out = F.interpolate(x, scale_factor=2, mode=self.upscaling_mode, align_corners=True)
        out = self.convolutions(out)

        return out