import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    '''
    Helper class that encodes the input
    '''

    def __init__(self, input_dimensions: int, octaves_count: int, append_original: bool):
        '''
        Builds the positional encoder
        :param input_dimensions: number of input dimensions to encode
        :param octaves_count: number of octaves to use for encoding. 2 * octaves_count encodings are created for each input dimension
        :param append_original: if True also appends to the encoded input the original input
        '''

        super(PositionalEncoder, self).__init__()

        self.input_dimensions = input_dimensions
        self.octaves_count = octaves_count
        self.append_original = append_original

        # Precomputes octaves and embedding functions
        self.encoding_functions = [torch.sin, torch.cos]

        octaves_tensor = 2.0 ** torch.linspace(0.0, self.octaves_count - 1, self.octaves_count)
        self.register_buffer("octaves", octaves_tensor, persistent=False)  # Do not save in state_dict

    def get_encoding_size(self) -> int:
        '''
        Returns the size of the encoded input
        :return:
        '''

        encoding_size = 2 * self.octaves_count * self.input_dimensions
        if self.append_original:
            encoding_size += self.input_dimensions

        return encoding_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Encodes the given input
        :param input: (..., input_dimensions) tensor with input to encode
        :return: (..., ([input_dimensions +] 2 * octaves_count) * input_dimensions) tensor with encoded input. the initial input_dimensions
                                                                                    are added only if append_original is specified
        '''

        input_dimension = input.size(-1)
        if input_dimension != self.input_dimensions:
            raise Exception(f"Input dimension ({input_dimension}) differs from expected input dimension ({self.input_dimension})")

        # Gathers the encoded parts
        all_encodings = []
        if self.append_original:
            all_encodings.append(input)

        # Encodes all the input
        for current_octave in self.octaves:
            for current_function in self.encoding_functions:
                current_encoding = current_function(current_octave * input)
                all_encodings.append(current_encoding)

        results = torch.cat(all_encodings, dim=-1)
        return results
