import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, padding_mode="zeros"):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DoubleConvolutionBlock(nn.Module):
    '''
    Block with two convolutions
    '''

    def __init__(self, in_planes, out_planes, last_affine=True, drop_final_activation=False, padding_mode="zeros"):
        '''

        :param in_features: Input features to the module
        :param out_features: Output feature
        :param downsample_factor: Reduction factor in feature dimension
        :param drop_final_activation: if True does not pass the final output through the activation function
        :param padding_mode: the padding mode to use for the convolutions
        '''

        super(DoubleConvolutionBlock, self).__init__()

        norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(in_planes, out_planes, stride=1, padding_mode=padding_mode)
        self.bn1 = norm_layer(out_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, padding_mode=padding_mode)
        self.bn2 = norm_layer(out_planes, affine=last_affine) # Enable the possibility to force alignment to normal gaussian
        self.relu2 = nn.ReLU(inplace=True)
        self.drop_final_activation = drop_final_activation

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.drop_final_activation:
            out = self.relu2(out)

        return out
