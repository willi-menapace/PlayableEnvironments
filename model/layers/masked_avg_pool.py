import torch


class MaskedAvgPool1d(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(MaskedAvgPool1d, self).__init__()

        self.eps = eps

    def forward(self, x, mask: torch.Tensor):
        '''

        :param x: (batch_size, features_count, observations_count) tensor with mask. observations_counnt dimension is optional
        :param mask: (batch_size, observations_count) tensor with mask of values to use for computation of the average. observations_count dimension is optional
        :return: (batch_size, 1) last 1 retained for compatibility with AdaptiveAvgPool1d
        '''

        features_count = x.size(1)
        mask = mask.unsqueeze(1)
        mask = mask.repeat((1, features_count, 1))

        # Zeros features to mask out invalid regions
        x[torch.logical_not(mask)] = 0.0

        # (batch_size, features_count)
        sums = x.sum(dim=-1)

        # (batch_size, features_count)
        trues_count = mask.sum(dim=-1, dtype=x.dtype) + self.eps

        means = sums / trues_count
        # (batch_size, -1)
        means = means.unsqueeze((-1))

        return means


if __name__ == "__main__":

    input = torch.ones(((16, 4, 16)))
    mask = torch.ones_like(input, dtype=torch.bool)
    mask = mask[:, 0]
    mask[0, 5:] = False
    avg_pooling = MaskedAvgPool1d()

    means = avg_pooling(input, mask)

    print(means)
