import torch

class RotationEncoder:
    '''
    Class for rotation encoding and decoding
    '''

    @staticmethod
    def encode(rotations: torch.Tensor, dim: int) -> torch.Tensor:
        '''
        Encodes the given rotations
        :param rotations: (..., 3, ...) tensor with rotations
        :param dim: the dimension with the rotations, must be negative
        :return: (..., 6, ...) tensor with encoded rotations where each angle is encoded in (sin, cos) format
        '''

        sin_tensor = torch.sin(rotations)
        cos_tensor = torch.cos(rotations)

        # Interleaves sin and cos in the shape (sinx, cosx, siny, cosy, sinz, cosz)
        stacked_tensor = torch.stack([sin_tensor, cos_tensor], dim=dim)
        interleaved_tensor = torch.flatten(stacked_tensor, start_dim=dim-1, end_dim=dim)

        return interleaved_tensor

    @staticmethod
    def decode(encoded_rotations: torch.Tensor, dim: int) -> torch.Tensor:
        '''
        Decodes the given rotations
        :param encoded_rotations: (..., 6, ...) tensor with encoded rotations where each angle is encoded in (sin, cos) format (sinx, cosx, siny, cosy, sinz, cosz)
        :param dim: the dimension with the rotations, must be negative
        :return: (..., 3, ...) tensor with decoded rotations
        '''

        sin_index = torch.as_tensor([0, 2, 4], dtype=torch.long, device=encoded_rotations.device)
        cos_index = torch.as_tensor([1, 3, 5], dtype=torch.long, device=encoded_rotations.device)
        sin_tensor = torch.index_select(encoded_rotations, dim, sin_index)
        cos_tensor = torch.index_select(encoded_rotations, dim, cos_index)

        # sinxyz, cosxyz
        # outputs (..., 3)
        decoded_rotations = torch.atan2(sin_tensor, cos_tensor)

        return decoded_rotations


if __name__ == "__main__":

    tensor = torch.tensor([[0.0, 3.14, -0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    encoded_tensor = RotationEncoder.encode(tensor, dim=-2)
    decoded_tensor = RotationEncoder.decode(encoded_tensor, dim=-2)

    print(tensor)
    print(encoded_tensor)
    print(decoded_tensor)