import torch
import torch.nn as nn

from model.layers.adain import AffineTransformAdaIn
from model.layers.adain_unnormalized import AffineTransformAdaInUnnormalized
from model.layers.style_linear_layer import StyleLinearLayer
from model.layers.style_linear_layer_v2 import StyleLinearLayerV2


class AdaInSequential(nn.Sequential):
    '''
    A AdaIn aware sequential layer that allows feeding style information through a series of mixed layers
    '''
    def forward(self, x: torch.Tensor, style: torch.Tensor):
        '''
        Forwards the output of each module as input to the successive. Each AdaIn layer is also fed with the style
        :param x: (batch_size, in_features) Initial tensor that is fed to the first sequence element
        :param style: (batch_size, style_features_count) Style to feed to every AdaIn layer
        :return: the output of the last layer
        '''

        for module in self._modules.values():
            if isinstance(module, AffineTransformAdaIn) or isinstance(module, StyleLinearLayer) or isinstance(module, StyleLinearLayerV2) or isinstance(module, AffineTransformAdaInUnnormalized):
                x = module(x, style)
            else:
                x = module(x)

        return x
