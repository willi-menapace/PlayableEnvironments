import math

import torch

from model.positional_encoder import PositionalEncoder


class AnnealablePositionalEncoder(PositionalEncoder):
    '''
    Helper class that encodes the input
    '''

    def __init__(self, input_dimensions: int, octaves_count: int, append_original: bool, num_steps: int):
        '''
        Builds the positional encoder
        :param input_dimensions: number of input dimensions to encode
        :param octaves_count: number of octaves to use for encoding. 2 * octaves_count encodings are created for each input dimension
        :param append_original: if True also appends to the encoded input the original input
        :param num_steps: Number of steps to use for the annealing
        '''

        super(AnnealablePositionalEncoder, self).__init__(input_dimensions, octaves_count, append_original)

        self.num_steps = num_steps

        # Saves the current step as a parameter
        current_step_tensor = torch.zeros((), dtype=torch.int)
        self.register_buffer("current_step", current_step_tensor)

        # The indexes for each octave
        octave_indexes = []
        for idx in range(self.octaves_count):
            octave_indexes.append(idx)
        octave_indexes_tensor = torch.as_tensor(octave_indexes, dtype=torch.float32)
        self.register_buffer("octave_indexes", octave_indexes_tensor, persistent=False)

    def set_step(self, current_step: int):
        '''
        Sets the current step to the specified value
        :param current_step:
        :return:
        '''

        self.current_step = self.current_step * 0 + current_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Encodes the given input
        :param input: (..., input_dimensions) tensor with input to encode
        :return: (..., ([input_dimensions +] 2 * octaves_count) * input_dimensions) tensor with encoded input. the initial input_dimensions
                                                                                    are added only if append_original is specified
        '''

        # Computes the weights for the annealing
        alpha = self.current_step * self.octaves_count / self.num_steps
        clamped_annealing_values = math.pi * torch.clamp(alpha - self.octave_indexes, min=0.0, max=1.0)
        # (octaves_count) tensor with annealing weights
        annealing_weights = (1 - torch.cos(clamped_annealing_values)) / 2

        input_dimension = input.size(-1)
        if input_dimension != self.input_dimensions:
            raise Exception(f"Input dimension ({input_dimension}) differs from expected input dimension ({self.input_dimension})")

        # Gathers the encoded parts
        all_encodings = []
        if self.append_original:
            all_encodings.append(input)

        # Encodes all the input
        for octave_idx, current_octave in enumerate(self.octaves):
            for current_function in self.encoding_functions:
                current_encoding = current_function(current_octave * input) * annealing_weights[octave_idx]
                all_encodings.append(current_encoding)

        results = torch.cat(all_encodings, dim=-1)
        return results
