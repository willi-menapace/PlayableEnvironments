import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d

from evaluation.metrics.fid import FID
from pytorch_fid.inception import InceptionV3
from utils.tensor_folder import TensorFolder


class IncrementalFID(nn.Module):

    def __init__(self):
        super(IncrementalFID, self).__init__()

        # Uses the default block size
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx])

        self.clear()

    def clear(self):
        '''
        Clears the state of the object
        :return:
        '''

        self.all_reference_preds = []
        self.all_generated_preds = []

    def calculate_activation_statistics(self, activations):

        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def get_current_activations(self, observations: torch.Tensor):
        '''
        Extracts the activations for the current observations

        :param observations: (..., 3, height, width) tensor with observations
        :return: numpy array with resulting activations
        '''

        observations, _ = TensorFolder.flatten(observations, -3)

        with torch.no_grad():
            pred = self.model(observations)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        return pred

    def compute_fid(self) -> float:
        '''
        Computes the FID for the activations accumulated up to the current moment
        :return:
        '''

        all_reference_activations = np.concatenate(self.all_reference_preds, axis=0)
        all_generated_activations = np.concatenate(self.all_generated_preds, axis=0)

        m1, s1 = self.calculate_activation_statistics(all_reference_activations)
        m2, s2 = self.calculate_activation_statistics(all_generated_activations)

        fid_value = FID.calculate_frechet_distance(m1, s1, m2, s2)

        return float(fid_value)

    def forward(self, reference_observations, generated_observations):
        '''
        Accumulates activation statistics about the current reference and generated observations

        :param reference_observations: (..., 3, height, width) tensor with reference observations
        :param generated_dataloader: (..., 3, height, width) tensor with generated observations
        :return:
        '''

        self.eval()

        # Computes the current activations
        current_reference_activations = self.get_current_activations(reference_observations)
        current_generated_activations = self.get_current_activations(generated_observations)

        # Stores the current activations
        self.all_reference_preds.append(current_reference_activations)
        self.all_generated_preds.append(current_generated_activations)

