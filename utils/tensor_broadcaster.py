import torch


class TensorBroadcaster:

    @staticmethod
    def add_dimension(tensor: torch.Tensor, size: int, dim: int) -> torch.Tensor:
        '''

        :param tensor: the tensor on which to add the dimension
        :param size: size of the new dimension to add
        :param dim: dimension for the new dimension
        :return: The input tensor with the new dimension added
        '''

        # Adds 1 dimension
        tensor = tensor.unsqueeze(dim)

        # Computes the number of times each dimension should be repeated
        dimensions_count = len(tensor.size())
        repeat_counts = [1] * dimensions_count
        repeat_counts[dim] = size

        tensor = tensor.repeat(repeat_counts)

        return tensor


