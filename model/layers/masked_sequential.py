import torch
import torch.nn as nn

from model.layers.masked_avg_pool import MaskedAvgPool1d
from model.layers.masked_batch_norm import MaskedBatchNorm1d


class MaskedSequential(nn.Sequential):
    '''
    A Mask aware sequential layer that allows feeding style information through a series of mixed layers
    '''
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        '''
        Forwards the output of each module as input to the successive. Each Masked layer is also fed with the mask
        :param x: (batch_size, features, observations_count) Initial tensor that is fed to the first sequence element
        :param mask: (batch_size, observations_count) Mask to feed to masked layers
        :return: the output of the last layer
        '''

        for module in self._modules.values():
            if isinstance(module, MaskedBatchNorm1d) or isinstance(module, MaskedAvgPool1d):
                x = module(x, mask)
            else:
                x = module(x)

        return x
