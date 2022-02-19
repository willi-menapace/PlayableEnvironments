from typing import Union, List

import torch
import torch.nn as nn

from utils.tensor_folder import TensorFolder


class IndexedStorage(nn.Module):
    '''
    Module for storing and retrieving tensors by index
    '''

    def __init__(self, storage_size: int, features_size: int, init_mode="random"):
        '''
        Initializes the model

        :param storage_size: Number of entries to store with indexes int [0, storage_size - 1]
        :param features_size: Number of features for each entry
        '''
        super(IndexedStorage, self).__init__()

        self.storage_size = storage_size
        self.features_size = features_size

        # Storage containing all tensors
        # Each tensor is memorized separately so that the optimizer won't update optimization
        # parameters if tensors have not received gradients
        self.storage = nn.ParameterList()

        for idx in range(storage_size):
            # Initializes the storage
            if init_mode == "random":
                storage_tensor = torch.randn((features_size), dtype=torch.float32)
            elif init_mode == "zero":
                storage_tensor = torch.zeros((features_size), dtype=torch.float32)
            else:
                raise Exception(f"Unknown init mode {init_mode}")

            self.storage.append(nn.Parameter(storage_tensor, requires_grad=True))

    def forward(self, indexes: Union[List[int], torch.Tensor]):
        '''

        :param indexes: (...) Long tensor with the indexes to retrieve. List of integers is also accepted
        :return: (..., features_size) tensor with the retrieved features
        '''

        # If indexes are integers, convert them to tensors
        if not torch.is_tensor(indexes):
            indexes = torch.as_tensor(indexes, dtype=torch.long, device=self.storage.device)

        flat_indexes, index_dimensions = TensorFolder.flatten(indexes, 0)

        # Extracts each tensor
        selected_entries = []
        for current_idx in flat_indexes:
            current_parameter = self.storage[current_idx.item()]
            selected_entries.append(current_parameter)

        # Transforms the entries in a tensor
        selected_entries = torch.stack(selected_entries, dim=0)
        selected_entries = torch.reshape(selected_entries, index_dimensions + [self.features_size])

        return selected_entries
