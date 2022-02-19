import torch

from dataset.batching import Batch


def compute_ground_truth_object_translations(batch: Batch, objects_count: int) -> torch.Tensor:
    '''
    Computes ground truth object translations starting from the batch
    :param batch: the batch from which to extract the translations
    :param objects_count: the number of objects in the model
    :return: (batch_size, observations_count, 3, objects_count) tensor with ground truth object positions
    '''

    metadata = batch.metadata

    batch_size = batch.observations.size(0)
    observations_count = batch.observations.size(1)

    positions = torch.zeros((batch_size, observations_count, 3, 1))
    for batch_idx in range(batch_size):
        for observation_idx in range(observations_count):
            positions[batch_idx, observation_idx, :, 0] = torch.from_numpy(
                metadata[batch_idx][observation_idx][0]["current_position"])

    # Object y is 0
    positions[..., 1, :] = 0
    positions = positions.cuda()

    # Each object apart from the last is considered background, so we create tensors with 0 translation
    # for each background object
    all_positions = []
    for background_object_idx in range(objects_count - 1):
        all_positions.append(torch.zeros_like(positions))

    all_positions.append(positions)
    return torch.cat(all_positions, dim=-1)