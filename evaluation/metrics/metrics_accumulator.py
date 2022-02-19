from typing import List, Union

import numpy as np
import torch

class MetricsAccumulator:
    '''
    Utility class for metrics accumulation and averaging
    '''

    def __init__(self):
        self.data = {}

    def reset(self):
        self.data = {}

    def to_numpy(self, item):

        # Convert torch tensors to numpy
        if torch.is_tensor(item):
            return item.detach().cpu().numpy()

        # Converts recursively
        if isinstance(item, list) or isinstance(item, tuple):
            new_item = []
            for current_item in item:
                new_item.append(self.to_numpy(current_item))
            new_item = np.stack(new_item, axis=0)
            return new_item

        return item

    def add(self, key: str, value: Union[np.ndarray, torch.Tensor, List]):
        '''
        Adds value to the set of values represented by key

        :param key: The key to associate to the current values
        :param value: (size, ...) tensor or array with the values to store.
                      Alternatively arbitrarily nested list of the above
        :return:
        '''

        value = self.to_numpy(value)

        if not key in self.data:
            self.data[key] = []

        self.data[key].append(value)

    def pop(self, key: str, dim=0):
        '''
        Obtains a tensor with all the concatenated values corresponding to a key
        Eliminates the key

        :param key: The key for which to retrieve the values
        :return: tensor with all the values concatenated along dimension dim
        '''

        if key not in self.data:
            raise Exception(f"Key '{key}' is not presetn")

        result = np.concatenate(self.data[key], axis=dim)
        del self.data[key]
        return result