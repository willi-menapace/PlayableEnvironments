from typing import Iterator

from torch.nn import Parameter

from model.environment_model import EnvironmentModel
from model.playable_environment_model import PlayableEnvironmentModel


class PlayableEnvironmentModelV2(PlayableEnvironmentModel):
    '''
    A playable environment model without the non_environment_parameters bug returning also environment_parameters
    '''

    def __init__(self, config, environment_model: EnvironmentModel):
        '''
        Initializes the environment model

        :param config: the configuration file
        :param environment_model: the model for the environment. Must be in evaluation mode
        '''
        super(PlayableEnvironmentModelV2, self).__init__(config, environment_model)

    def non_environment_parameters(self, recurse: bool = True, additional_excluded_names = None) -> Iterator[Parameter]:
        '''
        Returns only non environment model parameters
        :param recurse:
        :return:
        '''

        # Gathers the names of parameters for the environment model
        excluded_names = set()
        for name, _ in self.environment_model.named_parameters():
            excluded_names.add("environment_model." + name)

        if additional_excluded_names is not None:
            additional_excluded_names = set(additional_excluded_names)
            excluded_names = excluded_names.union(additional_excluded_names)

        # Yields each parameter that is not in the environment model
        for name, parameter in self.named_parameters(recurse=recurse):
            if name not in excluded_names:
                yield parameter


def model(config, environment_model: EnvironmentModel):
    return PlayableEnvironmentModelV2(config, environment_model)
