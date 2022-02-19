from typing import List

import torch
from torchvision import models


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss
    """
    def __init__(self, requires_grad=False, features_count: int = 5):
        '''

        :param requires_grad:
        :param features_count: number of features to produce
        '''
        super(Vgg19, self).__init__()

        self.features_count = features_count

        # Loads the model
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        # Divides the vgg modules in slices
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        if features_count > 1:
            for x in range(2, 7):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
        if features_count > 2:
            for x in range(7, 12):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if features_count > 3:
            for x in range(12, 21):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if features_count > 4:
            for x in range(21, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # VGG parameters need to remain fixed
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        '''

        :param x: (bs, 3, height, width) tensor representing the input image
        :return: List of (bs, features_i, height_i, width_i) tensors representing vgg features at different levels
        '''

        h_relu1 = self.slice1(x)
        out = [h_relu1]
        if self.features_count > 1:
            h_relu2 = self.slice2(h_relu1)
            out.append(h_relu2)
        if self.features_count > 2:
            h_relu3 = self.slice3(h_relu2)
            out.append(h_relu3)
        if self.features_count > 3:
            h_relu4 = self.slice4(h_relu3)
            out.append(h_relu4)
        if self.features_count > 4:
            h_relu5 = self.slice5(h_relu4)
            out.append(h_relu5)

        return out
