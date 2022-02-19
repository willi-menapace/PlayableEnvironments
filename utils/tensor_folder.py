from typing import List, Tuple, Sequence

import torch


class TensorFolder:

    @staticmethod
    def prod(input: Sequence):
        '''
        Computes the product of the elements in a sequence
        :param input: sequence of elements
        :return: Product of the elements in the seuqence
        '''
        result = 1
        for i in input:
            result *= i
        return result

    @staticmethod
    def flatten(tensor: torch.Tensor, dimensions: int = 2) -> Tuple[torch.Tensor, List]:
        '''
        Flattens the first dimensions of the tensor

        :param tensor: (...) tensor
        :param dimensions: number of dimensions to include in the flattened dimension. If negative it represents the
                           number of dimensions at the end that must not be included in the flattening
        :return: (dim1 * dim2 *... * dimdimensions, ...) tensor
        '''

        tensor_size = list(tensor.size())
        if dimensions <= 0:
            dimensions = len(tensor_size) + dimensions
        flattened_tensor = torch.reshape(tensor, tuple([TensorFolder.prod(tensor_size[:dimensions])] + tensor_size[dimensions:]))
        removed_dimensions = tensor_size[:dimensions]

        return flattened_tensor, removed_dimensions

    @staticmethod
    def flatten_list(tensors: List[torch.Tensor], dimensions: int = 2) -> Tuple[torch.Tensor, List]:
        '''
        Flattens the first dimensions of the tensor

        :param tensor: (...) list of tensors
        :param dimensions: number of dimensions to include in the flattened dimension. If negative it represents the
                           number of dimensions at the end that must not be included in the flattening.
                           The flattened dimensions must be the same size for all elements in the list
        :return: [(dim1 * dim2 *... * dimdimensions, ...)] list of flattened tensors
        '''

        first_flattened_tensor, initial_dimensions = TensorFolder.flatten(tensors[0], dimensions)
        all_flattened_tensors = [first_flattened_tensor] + [TensorFolder.flatten(current_tensor, dimensions)[0] for current_tensor in tensors[1:]]

        return all_flattened_tensors, initial_dimensions

    @staticmethod
    def fold(tensor: torch.Tensor, dimensions: List[int]) -> torch.Tensor:
        '''
        Separates the first tensor dimensions into separate dimensions of the specified size. If the specified dimensions
        do not multiply to the size of the initial dimension, an initial dimension is automatically computed and added
        if possible

        :param tensor: (dim1 * dim2 * ... * dimdimensions, ...) tensor
        :param dimensions: the initial dimensions of the output tensor. dim1 is inferred if not specified
        :return: (dim1, dim2, ..., dimdimensions, ...) tensor
        '''

        tensor_size = list(tensor.size())
        first_dimension_size = tensor_size[0]

        dimensions_product = TensorFolder.prod(dimensions)

        # Checks sizes
        if (dimensions_product != 0) and (first_dimension_size % dimensions_product != 0):
            raise Exception(f"First dimension {first_dimension_size} is not the product of the specified dimensions, nor dim1 can be inferred {dimensions}")

        if first_dimension_size != dimensions_product:
            dimensions = [first_dimension_size // dimensions_product] + dimensions

        tensor = torch.reshape(tensor, (list(dimensions) + tensor_size[1:]))
        return tensor

    @staticmethod
    def fold_list(tensors: List[torch.Tensor], dimensions: List[int]) -> torch.Tensor:
        '''
        Separates the first tensor dimensions into separate dimensions of the specified size. If the specified dimensions
        do not multiply to the size of the initial dimension, an initial dimension is automatically computed and added
        if possible

        :param tensor: (dim1 * dim2 * ... * dimdimensions, ...) list of tensors to fold
        :param dimensions: the initial dimensions of the output tensor. dim1 is inferred if not specified
        :return: [(dim1, dim2, ..., dimdimensions, ...)] list of tensors
        '''

        all_tensors = [TensorFolder.fold(current_tensor, dimensions) for current_tensor in tensors]

        return all_tensors

