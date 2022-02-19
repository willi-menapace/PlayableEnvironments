from typing import Tuple

import torch
import numpy
import pickle


class TensorStorage:

    def __init__(self):

        self.storage = {}

    def add_tensor(self, idx: int, tensor: torch.Tensor):
        '''
        Adds a trajectory to the trajectory set
        :param idx: ID of the trajectory
        :param tensor: (..., 3) tensor to store
        :return:
        '''

        tensor = tensor.cpu().numpy()

        self.storage[idx] = tensor

    def get_tensor(self, idx: int, device=None, add_batch_dimension=False) -> torch.Tensor:
        '''
        Gets a trajctory from the set of trajectories
        :param idx: Id of the trajectory to retrieve
        :param device: device where to place the tensors
        :param add_batch_dimension: if True adds a 1 batch dimension as the first tensor dimension
        :return: tensor
        '''

        tensor = self.storage[idx]

        tensor = torch.from_numpy(tensor)

        if device is not None:
            tensor = tensor.to(device)

        if add_batch_dimension:
            tensor = tensor.unsqueeze(0)

        return tensor

    def save(self, path: str):
        '''
        Saves the set of current trajectories
        :param path: path to the file where to store the trajectories
        :return:
        '''

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str):
        '''
        Load a set of trajectories
        :param path: path to the file where the trajectories are stored
        :return:
        '''

        with open(path, "rb") as file:
            storage = pickle.load(file)

        return storage
