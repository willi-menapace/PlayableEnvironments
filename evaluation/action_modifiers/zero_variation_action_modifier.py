import torch


class ZeroVariationActionModifier:

    def __init__(self):
        self.name = "zero_variation"

    def apply(self, actions: torch.Tensor, action_variations: torch.Tensor):
        '''
        Applies the modification to actions and action variations
        Returns action variations that are 0. Actions are unaltered

        :param actions: (..., actions_count) tensor with action probabilities
        :param action_variations: (..., action_space_dimension) tensor with action variations
        :return:
        '''

        action_variations = action_variations * 0.0

        return actions, action_variations
