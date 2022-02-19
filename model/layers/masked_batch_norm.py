import torch

'''
Applies Batch Normalization over a 1D input (or 2D tensor)

Shape:
  Input: (N, C)
  Output: (N, C)

Input Parameters:
  in_features: number of features of the input activations
  track_running_stats: whether to keep track of running mean and std. (default: True)
  affine: whether to scale and shift the normalized activations. (default: True)
  momentum: the momentum value for the moving average. (default: 0.9)

Usage:
  >>> # with learable parameters
  >>> bn = BatchNorm1d(4)
  >>> # without learable parameters
  >>> bn = BatchNorm1d(4, affine=False)
  >>> input = torch.rand(10, 4)
  >>> out = bn(input)
'''


class MaskedBatchNorm1d(torch.nn.Module):
    def __init__(self, in_features, track_running_stats=True, affine=True, momentum=0.9, eps=1e-5):
        super(MaskedBatchNorm1d, self).__init__()

        self.in_features = in_features
        self.track_running_stats = track_running_stats
        self.affine = affine
        self.momentum = momentum
        self.eps = eps

        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(self.in_features))
            self.beta = torch.nn.Parameter(torch.zeros(self.in_features))

        if self.track_running_stats:
            # register_buffer registers a tensor as a buffer that will be saved as part of the model
            # but which does not require to be trained, differently from nn.Parameter
            self.register_buffer('running_mean', torch.zeros(self.in_features))
            self.register_buffer('running_std', torch.ones(self.in_features))

    def forward(self, x, mask: torch.Tensor):
        '''

        :param x: (batch_size, features_count, observations_count) tensor with mask. observations_counnt dimension is optional
        :param mask: (batch_size, observations_count) tensor with mask of values to use for computation of bn statistics. observations_counnt dimension is optional
        :return:
        '''

        x_orig = x

        # transpose (batch_size, features_count, observations_count) to (features_count, batch_size, observations_count)
        x = x.transpose(0, 1).contiguous().view(x.shape[1], -1)
        mask = mask.contiguous().view((-1, ))

        # Masks the input
        x = x[:, mask]

        # calculate batch mean
        mean = x.mean(dim=1)

        # calculate batch std
        std = x.std(dim=1)

        # during training keep running statistics (moving average of mean and std)
        if self.training and self.track_running_stats:
            # No computational graph is necessary to be built for this computation
            with torch.no_grad():
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_std = self.momentum * self.running_std + (1 - self.momentum) * std

        # during inference time
        if not self.training and self.track_running_stats:
            mean = self.running_mean
            std = std = self.running_std

        # normalize the input activations
        if len(x_orig.size()) == 3:
            mean = mean.unsqueeze(-1)
            std = std.unsqueeze(-1)
        x_orig = (x_orig - mean) / (std + self.eps)

        # scale and shift the normalized activations
        if self.affine:
            if len(x_orig.size()) == 3:
                gamma = self.gamma.unsqueeze(-1)
                beta = self.beta.unsqueeze(-1)
            else:
                gamma = self.gamma
                beta = self.beta
            x_orig = x_orig * gamma + beta

        return x_orig


if __name__ == "__main__":

    input = torch.randn(((16, 4, 16)))
    mask = torch.ones_like(input, dtype=torch.bool)
    mask = mask[:, 0]
    mask[0] = False
    batch_norm = MaskedBatchNorm1d(4)

    normalized_input = batch_norm(input, mask)

    print(normalized_input)
