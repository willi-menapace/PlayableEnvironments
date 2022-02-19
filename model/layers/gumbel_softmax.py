import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.tensor_folder import TensorFolder


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Taken from: https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e

    Samples from the `Gumbel-Softmax distribution`_ and optionally discretizes.
    You can use this function to replace "F.gumbel_softmax".

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    .. _Gumbel-Softmax distribution:
        https://arxiv.org/abs/1611.00712
        https://arxiv.org/abs/1611.01144
    """

    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()


        if torch.isnan(gumbels).any():
            raise Exception(f"Nans at {__file__}:{1}, {gumbels}")
        if torch.isinf(gumbels).any():
            raise Exception(f"Infs at {__file__}:{1}, {gumbels}")


        return gumbels


    if torch.isnan(logits).any():
        raise Exception(f"Nans at {__file__}:{2}, {logits}")
    if torch.isinf(logits).any():
        raise Exception(f"Infs at {__file__}:{2}, {logits}")


    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)



    if torch.isnan(gumbels).any():
        raise Exception(f"Nans at {__file__}:{3}, {gumbels}")
    if torch.isinf(gumbels).any():
        raise Exception(f"Infs at {__file__}:{3}, {gumbels}")



    y_soft = gumbels.softmax(dim)



    if torch.isnan(y_soft).any():
        raise Exception(f"Nans at {__file__}:{4}, {y_soft}")
    if torch.isinf(y_soft).any():
        raise Exception(f"Infs at {__file__}:{4}, {y_soft}")


    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GumbelSoftmax(nn.Module):
    '''
    Module for gumbel sampling. Handles nans
    '''

    def __init__(self, initial_temperature, hard=False):
        '''
        Initializes the samples to operate at the given temperature
        :param initial_temperature: initial temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :param hard: if true uses the hard straight through gumbel implementation
        '''
        super(GumbelSoftmax, self).__init__()

        self.current_temperature = initial_temperature
        self.hard = hard

    def forward(self, log_probs, temperature=None):
        '''

        :param log_probs: (..., classes_count) tensor representing log of probabilities
        :param temperature: new temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :return: (..., classes_count) tensor with sampled actions
        '''

        if temperature is not None:
            self.current_temperature = temperature

        return gumbel_softmax(log_probs, self.current_temperature, self.hard)


class LegacyGumbelSoftmax(nn.Module):
    '''
    Module for gumbel sampling
    '''

    def __init__(self, initial_temperature, hard=False):
        '''
        Initializes the samples to operate at the given temperature
        :param initial_temperature: initial temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :param hard: if true uses the hard straight through gumbel implementation
        '''
        super(LegacyGumbelSoftmax, self).__init__()

        self.current_temperature = initial_temperature
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-20):
        '''
        Samples gumbel variable with given shape
        :param shape: shape of the variable to output
        :param eps: constant for numeric stability
        :return: (*shape) tensor with gumbel samples
        '''

        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_soft_sample(self, input):
        '''
        Computes soft gumbel samples

        :param input: (bs, classes_count) tensor representing log of probabilities
        :return: (bs, classes_count) soft samples
        '''

        y = input + self.sample_gumbel(input.size())
        return F.softmax(y / self.current_temperature, dim=-1)

    def forward(self, input, temperature=None):
        '''

        :param input: (..., classes_count) tensor representing log of probabilities
        :param temperature: new temperature at which to make the sampler operate.
                            temperatures close to 0 produce one hot samples, high temperatures approach uniform sampling
        :return: (..., classes_count) tensor with sampled actions
        '''

        if temperature is not None:
            self.current_temperature = temperature

        flat_input, initial_dimenstions = TensorFolder.flatten(input, -1)

        # Computes soft samples
        soft_samples = self.gumbel_soft_sample(flat_input)

        samples_to_return = None
        if self.hard:
            # Performs hard sampling
            shape = soft_samples.size()
            _, ind = soft_samples.max(dim=-1)
            y_hard = torch.zeros_like(soft_samples).view(-1, shape[-1])
            y_hard.scatter_(1, ind.view(-1, 1), 1)
            y_hard = y_hard.view(*shape)
            hard_samples = (y_hard - soft_samples).detach() + soft_samples  # Uses y_hard as output but with this detach trick, we use the gradients from y only, the non hard samples
            samples_to_return = hard_samples
        else:
            samples_to_return = soft_samples

        # Folds the results
        folded_samples_to_return = TensorFolder.fold(samples_to_return, initial_dimenstions)
        return folded_samples_to_return

def main():
    import math
    tens = Variable(torch.cuda.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 100000))

    zeros = torch.zeros((1, 7))
    zeros = zeros - 100
    zeros[0, 0] = 100.0
    log_probs = torch.log_softmax(zeros, dim=1)

    gumbel_softmax = GumbelSoftmax(1.0)
    samples_sum = gumbel_softmax(log_probs).sum(dim=0)
    print(samples_sum)

if __name__ == '__main__':
    main()