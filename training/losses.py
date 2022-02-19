import math
import sys
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tensor_folder import TensorFolder
from model.layers.vgg import Vgg19


class ReconstructionLoss:

    def __init__(self):
        pass

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param observations: (..., 3) ground truth observations with values in (-1, 1)
        :param reconstructed_observations: (..., 3) tensor with reconstructed frames and values in (0, 1)
        :return:
        '''

        # Normalizes the observations
        observations = (observations + 1) / 2

        loss = (observations - reconstructed_observations).pow(2).mean()
        #print("Warning: using L1 rec loss")
        #loss = torch.abs(observations - reconstructed_observations).mean()
        return loss


class AutoencoderReconstructionLoss:

    def __init__(self, type: str, normalize: bool):
        '''

        :param type: The type of loss to use
        :param normalize: Whether to normalize the loss by the norm of the features
        '''

        if type not in ["l2", "l1"]:
            raise NotImplementedError(f"Loss type {type} is not implemented")

        self.type = type
        self.normalize = normalize

    def __call__(self, features: torch.Tensor, reconstructed_features: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param features: (..., features_count) ground truth features
        :param reconstructed_features: (..., features_count) tensor with reconstructed features
        :return:
        '''

        if self.type == "l2":
            features_norm = features.pow(2).sum(-1)
            loss = (features - reconstructed_features).pow(2).sum(-1)
        elif self.type == "l1":
            features_norm = torch.abs(features).sum(-1)
            loss = torch.abs(features - reconstructed_features).sum(-1)

        if self.normalize:
            loss = loss / features_norm
        loss = loss.mean()

        return loss


class ImageReconstructionLoss:

    def __init__(self, use_radial_weights=False):
        '''

        :param use_radial_weights: If true, weights the loss giving more importance to features at the center
        '''

        self.use_radial_weights = use_radial_weights

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param observations: (..., 3, height, width) ground truth observations with values in (-1, 1)
        :param reconstructed_observations: (..., 3, height, width) tensor with reconstructed frames and values in (0, 1)
        :return:
        '''

        # Normalizes the observations
        observations = (observations + 1) / 2

        loss = (observations - reconstructed_observations).pow(2)

        # Create the weight mask if needed
        if self.use_radial_weights:
            height = observations.size(-2)
            width = observations.size(-1)
            initial_dimensions = list(observations.size())[:-3]

            # (1, height, width)
            weight_mask = WeightMaskBuilder.build_radial_weight_mask(height, width, observations.device).unsqueeze(0)
            # Makes the weight mask the same size as the observations
            for _ in range(len(initial_dimensions)):
                weight_mask = weight_mask.unsqueeze(0)
            weight_mask = weight_mask.repeat(initial_dimensions + [1, 1, 1])

            # Computes the weightes sum of the loss across the spatial dimensions
            weight_mask_sum = weight_mask.sum(dim=[-1, -2])
            loss = (loss * weight_mask).sum(dim=[-1, -2]) / weight_mask_sum

        loss = loss.mean()
        #print("Warning: using L1 rec loss")
        #loss = torch.abs(observations - reconstructed_observations).mean()
        return loss


class RayObjectDistanceLoss:
    '''
    Loss for distance between detected objects and rendered rays with high error
    '''

    def __init__(self):
        pass

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, ray_object_distance: torch.Tensor):
        '''
        Computes an error-weighted distance loss between rays and objects

        :param observations: (..., 3) ground truth observations with values in (-1, 1)
        :param reconstructed_observations: (..., 3) tensor with reconstructed frames and values in (0, 1)
        :param ray_object_distance: (..., objects_count) tensor with distance to each object
        :return:
        '''

        observations = (observations + 1) / 2
        #print("Warning: using rec loss in modified background version")
        reconstruction_error = (observations - reconstructed_observations).pow(2).sum(-1)
        reconstruction_error = reconstruction_error.unsqueeze(-1)  # Creates a dimension for objects_count

        # Computes the error-weighted distance
        ray_object_loss = (reconstruction_error * ray_object_distance).mean()
        return ray_object_loss


class BoundingBoxDistanceLoss:
    '''
    Loss for distance between detected and ground truth bounding boxes
    '''

    def __init__(self):
        pass

    def __call__(self, bounding_boxes: torch.Tensor, reconstructed_bounding_boxes: torch.Tensor, bounding_boxes_validity: torch.Tensor):
        '''
        Computes an error-weighted distance loss between rays and objects

        :param bounding_boxes: (..., 4, objects_count) ground truth bounding boxes for each object
        :param reconstructed_bounding_boxes: (..., 4, objects_count) ground truth bounding boxes for each object
        :param bounding_boxes_validity: (..., objects_count) boolean tensor indicating whether a valid detection is present for the object
        :return: scalar with average distance, (objects_count) tensor with distance for each object
        '''

        bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -2)
        reconstructed_bounding_boxes, _ = TensorFolder.flatten(reconstructed_bounding_boxes, -2)
        bounding_boxes_validity, _ = TensorFolder.flatten(bounding_boxes_validity, -1)
        objects_count = bounding_boxes.size(-1)
        reconstructed_objects_count = reconstructed_bounding_boxes.size(-1)

        if objects_count != reconstructed_objects_count:
            print(f"Warning: the number of bounding boxes ({objects_count}) and reconstructed bounding boxes ({reconstructed_objects_count}) differs."
                  f"This should happen only if the wrong number of objects was intentionally specified in the configuration files."
                  f"If this is not what you meant correct the configuration file by ensuring the number of dynamic objects matches the number of dynamic objects in the dataset?")

            # Returns zero distances
            distance = torch.tensor(0.0, device=bounding_boxes.device)
            per_object_distances = torch.zeros((objects_count,), dtype=bounding_boxes.dtype, device=bounding_boxes.device)
            return distance, per_object_distances

        per_object_distances = []
        for object_idx in range(objects_count):
            current_object_validity = bounding_boxes_validity[..., object_idx]

            current_bounding_boxes = bounding_boxes[current_object_validity][..., object_idx]
            current_reconstructed_bounding_boxes = reconstructed_bounding_boxes[current_object_validity][..., object_idx]

            current_object_distance = (current_reconstructed_bounding_boxes - current_bounding_boxes).pow(2).sum(-1).mean(0)
            per_object_distances.append(current_object_distance)

        distance = torch.stack(per_object_distances).mean()

        return distance, per_object_distances


class OpacityLoss:

    def __init__(self):
        pass

    def __call__(self, opacity: torch.Tensor, bounding_box_validity: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss for opacities

        :param opacity: (..., samples_per_image) tensor with opacities
        :param bounding_box_validity: (...) boolean tensor with values that indicate whether the current
                                                           object is present in the scene
        :return:
        '''

        # Considers only observations where the object is present in the camera view
        opacity = opacity[bounding_box_validity]

        # Computes the L1 magnitude of the opacities and returns it
        # abs should not be necessary
        return torch.abs(opacity).mean()


class AttentionLoss:

    def __init__(self):
        pass

    def __call__(self, attention: torch.Tensor, bounding_box_validity: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param attention: (..., 1, features_height, features_width) attention values
        :param bounding_box_validity: (..., cameras_count) boolean tensor with values that indicate whether the current
                                                           object is present in the scene
        :return:
        '''
        # Only the first camera is used to extract attention values
        bounding_box_validity = bounding_box_validity[..., 0]
        # Only considers values that refer to observations where the object is in the scene
        attention = attention[bounding_box_validity]
        # Computes the average L1 magnitude of attention
        return attention.mean()


class SharpnessLoss:

    def __init__(self, mean: float = 0.5, std: float = 0.15):
        '''
        :param mean: The mean for the gaussian density
        :param std: The standard deviation for the gaussian density
        '''

        self.std = std
        self.mean = mean

    def __call__(self, opacity: torch.Tensor, bounding_box_validity: torch.Tensor) -> torch.Tensor:
        '''
        Computes the sharpness loss

        :param opacity: (..., samples_per_image) tensor with opacities
        :param bounding_box_validity: (...) boolean tensor with values that indicate whether the current
                                                           object is present in the scene
        :return:
        '''

        # Considers only observations where the object is present in the camera view
        opacity = opacity[bounding_box_validity]

        var = self.std ** 2

        # Computes the gaussian density
        density = torch.exp((-((opacity - self.mean) ** 2) / (2 * var)))
        density = density / (math.sqrt(2 * math.pi * var))

        return density.mean()


class FixedMatrixEstimator(nn.Module):

    def __init__(self, rows, columns, initial_alpha=0.2, initial_value=None):
        '''
        Initializes the joint probability estimator for a (rows, columns) matrix with the given fixed alpha factor
        :param rows, columns: Dimension of the probability matrix to estimate
        :param initial_alpha: Value to use assign for alpha
        '''
        super(FixedMatrixEstimator, self).__init__()

        self.alpha = initial_alpha

        # Initializes the joint matrix as a uniform, independent distribution. Does not allow backpropagation to this parameter
        if initial_value is None:
            initial_value = torch.tensor([[1.0 / (rows * columns)] * columns] * rows, dtype=torch.float32)
        self.estimated_matrix = nn.Parameter(initial_value, requires_grad=False)

    def forward(self, latest_probability_matrix):
        return_matrix = self.estimated_matrix * (1 - self.alpha) + latest_probability_matrix * self.alpha

        # The estimated matrix must be detached from the backpropagation graph to avoid exhaustion of GPU memory
        self.estimated_matrix.data = return_matrix.detach()

        return return_matrix


class MutualInformationLoss(nn.Module):

    def __init__(self):
        super(MutualInformationLoss, self).__init__()

    def compute_joint_probability_matrix(self, distribution_1: torch.Tensor,
                                         distribution_2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the joint probability matrix

        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :return: (dim, dim) tensor with joint probability matrix
        '''

        # Flattens the distributions
        dim = distribution_1.size(-1)
        assert (distribution_2.size(-1) == dim)
        distribution_1 = distribution_1.view(-1, dim)
        distribution_2 = distribution_2.view(-1, dim)

        batch_size = distribution_1.size(0)
        assert (distribution_2.size(0) == batch_size)

        p_i_j = distribution_1.unsqueeze(2) * distribution_2.unsqueeze(1)  # (batch_size, dim, dim)
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def __call__(self, distribution_1: torch.Tensor, distribution_2: torch.Tensor,  lamb=1.0,
                 eps=sys.float_info.epsilon) -> torch.Tensor:
        '''
        Computes the mutual information loss for a joint probability matrix
        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :param lamb: lambda parameter to change the importance of entropy in the loss
        :param eps: small constant for numerical stability
        :return: mutual information loss for the given joint probability matrix
        '''

        # Computes the joint probability matrix
        joint_probability_matrix = self.compute_joint_probability_matrix(distribution_1, distribution_2)
        rows, columns = joint_probability_matrix.size()

        # Computes the marginals
        marginal_rows = joint_probability_matrix.sum(dim=1).view(rows, 1).expand(rows, columns)
        marginal_columns = joint_probability_matrix.sum(dim=0).view(1, columns).expand(rows,
                                                                                       columns)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        joint_probability_matrix[(joint_probability_matrix < eps).data] = eps
        marginal_rows = marginal_rows.clone()
        marginal_columns = marginal_columns.clone()
        marginal_rows[(marginal_rows < eps).data] = eps
        marginal_columns[(marginal_columns < eps).data] = eps

        mutual_information = joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                                         - lamb * torch.log(marginal_rows) \
                                                         - lamb * torch.log(marginal_columns))

        mutual_information = mutual_information.sum()

        return -1 * mutual_information


class KLGaussianDivergenceLoss:

    def __init__(self):
        pass

    def __call__(self, distribution_parameters: torch.Tensor):
        '''
        Computes the KL divergence between the given distribution and the N(0, 1) distribution

        :param distribution_parameters: (..., 2, space_dimension) tensor with distribution mean and log variances
        :return: KL distance between the given distribution and the N(0, 1) distribution
        '''

        space_dimension = distribution_parameters.size(-1)
        distribution_parameters = distribution_parameters.view(-1, 2, space_dimension)
        mean = distribution_parameters[:, 0]
        log_variance = distribution_parameters[:, 1]
        variance = torch.exp(log_variance)

        kl = 1 + log_variance - mean.pow(2) - variance  # (bs, space_dimension)
        kl = kl.sum(dim=-1)  # Sums across the space dimension
        kl = -0.5 * kl.mean()  # Averages across the batch dimension

        return kl


class KLGeneralGaussianDivergenceLoss:

    def __init__(self):
        pass

    def __call__(self, distribution_parameters: torch.Tensor, reference_distribution_parameters: torch.Tensor, eps=0.05):
        '''
        Computes the KL divergence between two given distributions

        :param distribution_parameters: (..., 2, space_dimension) tensor with distribution mean and variances
        :param reference_distribution_parameters: (..., 2, space_dimension) tensor with distribution mean and log variances
        :return: KL distance between distribution and the reference distribution
        '''

        space_dimension = distribution_parameters.size(-1)
        distribution_parameters = distribution_parameters.view(-1, 2, space_dimension)
        reference_distribution_parameters = reference_distribution_parameters.view(-1, 2, space_dimension)

        mean = distribution_parameters[:, 0]
        log_variance = distribution_parameters[:, 1].detach()  # Do not backpropagate through variance
        variance = torch.exp(log_variance)

        reference_mean = reference_distribution_parameters[:, 0]
        reference_variance = reference_distribution_parameters[:, 1].detach()  # Do not backpropagate through variance
        reference_log_variance = torch.log(reference_variance)

        variance = torch.clamp(variance, min=eps)
        reference_variance = torch.clamp(reference_variance, min=eps)

        variance_ratio = variance / reference_variance
        mus_term = (reference_mean - mean).pow(2) / reference_variance

        kl = reference_log_variance - log_variance - 1 + variance_ratio + mus_term  # (bs, space_dimension)

        kl = kl.sum(dim=-1)  # Sums across the space dimension
        kl = 0.5 * kl.mean()  # Averages across the batch dimension

        return kl


class SpatialKLGaussianDivergenceLoss:

    def __init__(self):
        pass

    def __call__(self, distribution_parameters: torch.Tensor):
        '''
        Computes the KL divergence between the given distribution and the N(0, 1) distribution

        :param distribution_parameters: (..., features_count * 2, height, width) tensor with distribution mean and log variances
                                                                                 The first half of the features represent mean, the
                                                                                 second represents log variance
        :return: KL distance between the given distribution and the N(0, 1) distribution
        '''

        mean, log_variance = torch.split(distribution_parameters, distribution_parameters.size(-3) // 2, dim=-3)

        """space_dimension = distribution_parameters.size(-1)
        distribution_parameters = distribution_parameters.view(-1, 2, space_dimension)
        mean = distribution_parameters[:, 0]
        log_variance = distribution_parameters[:, 1]"""
        variance = torch.exp(log_variance)

        kl = 1 + log_variance - mean.pow(2) - variance  # (..., features_count, height, width)
        kl = kl.sum(dim=-3)  # Sums across the features_count dimension
        kl = -0.5 * kl.mean()  # Averages across all the other dimensions

        return kl


class SmoothMutualInformationLoss(MutualInformationLoss):
    '''
    Mutual information loss with smooth joint probability matrix estimation
    '''

    def __init__(self, actions_count: int, alpha: float):
        '''
        Creates the loss according to the specified configuration
        :param config: The configuration
        '''

        super(SmoothMutualInformationLoss, self).__init__()

        self.actions_count = actions_count
        self.mi_estimation_alpha = alpha
        self.matrix_estimator = FixedMatrixEstimator(self.actions_count, self.actions_count, self.mi_estimation_alpha)

    def compute_joint_probability_matrix(self, distribution_1: torch.Tensor,
                                         distribution_2: torch.Tensor) -> torch.Tensor:
        '''
        Computes the joint probability matrix

        :param distribution_1: (..., dim) tensor of samples from the first distribution
        :param distribution_2: (..., dim) tensor of samples from the second distribution
        :return: (dim, dim) tensor with joint probability matrix
        '''

        # Compute the joint probability matrix as before
        current_joint_probability_matrix = super(SmoothMutualInformationLoss, self).compute_joint_probability_matrix(distribution_1, distribution_2)
        # Smooth the joint probability matrix with the estimator
        smoothed_joint_probability_matrix = self.matrix_estimator(current_joint_probability_matrix)
        return smoothed_joint_probability_matrix


class EntropyLogitLoss:

    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor):
        '''
        Computes the entropy of the passed logits
        :param logits: (..., classes_counts) tensor
        :return: entropy over the last dimension averaged on each sample
        '''

        classes_count = logits.size(-1)
        flat_logits = logits.reshape((-1, classes_count))
        samples_count = flat_logits.size(0)

        entropy = -1 * torch.sum(F.softmax(flat_logits, dim=1) * F.log_softmax(flat_logits, dim=1)) / samples_count
        return entropy


class EntropyProbabilityLoss:

    def __init__(self):
        pass

    def __call__(self, probabilities: torch.Tensor):
        '''
        Computes the entropy of the passed probabilities
        :param probabilities: (..., classes_counts) tensor with probabilitiy distribution over the classes
        :return: entropy over the last dimension averaged on each sample
        '''

        classes_count = probabilities.size(-1)
        flat_probabilities = probabilities.reshape((-1, classes_count))
        samples_count = flat_probabilities.size(0)

        entropy = -1 * torch.sum(flat_probabilities * torch.log(flat_probabilities)) / samples_count
        return entropy


class HeadSelectionLoss:

    def __init__(self):
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(self, head_selection_logits: torch.Tensor, ground_truth_indexes: torch.Tensor):
        '''
        Computes the entropy of the passed probabilities
        :param head_selection_logits: (batch_size, ..., heads_count) tensor with logits for head selection
        :param ground_truth_indexes: (batch_size) long tensor with head ground truth index for each frame
        :return: entropy over the last dimension averaged on each sample
        '''

        # Brings ground_truth_indexes to (batch_size, observations_count, ...)
        head_selection_dimensions = list(head_selection_logits.size())
        for i in range(len(head_selection_dimensions) - 2):
            ground_truth_indexes = ground_truth_indexes.unsqueeze(-1)
        _, ground_truth_indexes = torch.broadcast_tensors(head_selection_logits[..., 0], ground_truth_indexes)

        # Flattens the tensors
        flat_head_selection_logits, _ = TensorFolder.flatten(head_selection_logits, dimensions=-1)
        flat_ground_truth_indexes, _ = TensorFolder.flatten(ground_truth_indexes, dimensions=0)

        # Filters out outputs that are 0.0 because they correspond to samples outside of the bounding box
        mask = flat_head_selection_logits[:, 0] != 0.0
        flat_head_selection_logits = flat_head_selection_logits[mask, :]
        flat_ground_truth_indexes = flat_ground_truth_indexes[mask]

        cross_entropy_loss = self.cross_entropy_loss(flat_head_selection_logits, flat_ground_truth_indexes)

        return cross_entropy_loss


class WeightMaskBuilder:
    '''
    Helper class for building weight masks
    '''

    @staticmethod
    def build_radial_weight_mask(height: int, width: int, device) -> torch.Tensor:
        '''

        :param height: height of the mask to build
        :param width: width of the mask to build
        :param device: the device on which to place the tensors
        :return: (height, width) tensor with weights starting from 0 at the border and gradually increasing to 1 in the center
        '''

        center_height = (height - 1) / 2
        center_width = (width - 1) / 2

        mesh_rows, mesh_columns = torch.meshgrid([torch.arange(0, height, device=device), torch.arange(0, width, device=device)])
        # Computes the distances in the column and in the rows dimensions for each cell
        row_distances = torch.abs(mesh_rows - center_height)
        height_distances = torch.abs(mesh_columns - center_width)
        # Gets the maximum distance between the one in the rows and the one in the columns
        all_distances = torch.stack([row_distances, height_distances], dim=-1)
        all_distances = torch.max(all_distances, dim=-1)[0]

        # Translates the distances from the interval [max, min] to [0, 1] to create the weights
        max_distance = torch.max(all_distances).item()
        min_distance = torch.min(all_distances).item()
        weights = (all_distances - min_distance) / (max_distance - min_distance)
        # Makes 1 -> 0 and 0 -> 1
        weights = weights * -1 + 1
        return weights


class ParallelPerceptualLoss:

    def __init__(self, features_count: int = 5, use_radial_weights=False):
        '''

        :param features_count: see UnmeanedPerceptualLoss
        :param use_radial_weights: see UnmeanedPerceptualLoss
        '''

        self.perceptual_loss = UnmeanedPerceptualLoss(features_count=features_count, use_radial_weights=use_radial_weights)
        self.perceptual_loss = nn.DataParallel(self.perceptual_loss).cuda()

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None):
        total_loss, individual_losses = self.perceptual_loss(observations, reconstructed_observations, weight_mask)

        meaned_individual_losses = [current_loss.mean() for current_loss in individual_losses]
        return total_loss.mean(), meaned_individual_losses


class UnmeanedPerceptualLoss(nn.Module):

    def __init__(self, features_count: int = 5, use_radial_weights=False):
        '''

        :param features_count:
        :param use_radial_weights: If true, weights the loss giving more importance to features at the center
        '''

        super(UnmeanedPerceptualLoss, self).__init__()

        self.features_count = features_count
        self.vgg = Vgg19(features_count=features_count)
        self.vgg = self.vgg

        self.use_radial_weights = use_radial_weights

    def forward(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor, weight_mask=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Computes the perceptual loss between the sets of observations

        :param observations: (..., 3, h, w) ground truth observations. Resolution is rescaled if needed
        :param reconstructed_observations: (..., 3, height, width) tensor with reconstructed observations
        :param weight_mask: (..., 1, h, w) tensor weights to assign to each spatial position for loss computation. Rescaled if needed

        :return: total_loss, individual_losses Perceptual loss between ground truth and reconstructed observations. Both the total loss and the loss
                 for each vgg feature level are returned. Losses have a batch size dimension
        '''

        original_observation_height = observations.size(-2)
        original_observation_width = observations.size(-1)
        height = reconstructed_observations.size(-2)
        width = reconstructed_observations.size(-1)

        flattened_ground_truth_observations, _ = TensorFolder.flatten(observations, -3)
        flattened_reconstructed_observations, _ = TensorFolder.flatten(reconstructed_observations, -3)

        if weight_mask is not None:
            flattened_weight_mask, _ = TensorFolder.flatten(weight_mask, -3)
        if self.use_radial_weights:
            if weight_mask is not None:
                raise Exception("The loss function should use radial weight masks, but a mask was manually specified")

            elements_count = flattened_ground_truth_observations.size(0)
            # (1, 1, height, width)
            flattened_weight_mask = WeightMaskBuilder.build_radial_weight_mask(height, width, observations.device).unsqueeze(0).unsqueeze(0)
            # Expands the mask to create a dimension for the batch size
            # (elements_count, 1, height, width)
            flattened_weight_mask = flattened_weight_mask.repeat((elements_count, 1, 1, 1))

        # Resizes to the resolution of the reconstructed observations if needed
        if original_observation_width != width or original_observation_height != height:
            flattened_ground_truth_observations = F.interpolate(flattened_ground_truth_observations, (height, width), mode='bilinear')

        # Computes vgg features. Do not build the computational graph for the ground truth observations
        with torch.no_grad():
            ground_truth_vgg_features = self.vgg(flattened_ground_truth_observations)
        reconstructed_vgg_features = self.vgg(flattened_reconstructed_observations)

        total_loss = None
        single_losses = []
        # Computes the perceptual loss
        for current_ground_truth_feature, current_reconstructed_feature in zip(ground_truth_vgg_features, reconstructed_vgg_features):

            # Compute unweighted loss
            if weight_mask is None and not self.use_radial_weights:
                current_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature).mean(dim=[1, 2, 3]) # Detach signals to not backpropagate through the ground truth branch
            # Compute loss scaled by weights
            else:
                current_feature_channels = current_ground_truth_feature.size(1)
                current_feature_height = current_ground_truth_feature.size(2)
                current_feature_width = current_ground_truth_feature.size(3)

                # Resize the weight mask
                scaled_weight_masks = F.interpolate(flattened_weight_mask,
                                                    size=(current_feature_height, current_feature_width),
                                                    mode='bilinear', align_corners=False)

                unreduced_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature)
                # Computes the weighted sum of the loss using the weight mask as weights
                # Computes the loss such that each image has the same relative importance
                # Only the relative importance between positions in the same frame is modified
                unreduced_loss = unreduced_loss * scaled_weight_masks
                current_loss = unreduced_loss.sum(dim=(1, 2, 3))
                current_loss = current_loss / (scaled_weight_masks.sum(dim=(1, 2, 3)) * current_feature_channels)  # Since weight mask is broadcasted in the channel
                                                                                                                   # directions we need to multiply per the number of channels

            if total_loss is None:
                total_loss = torch.clone(current_loss)
            else:
                total_loss += current_loss
            single_losses.append(current_loss)

        return total_loss, single_losses


class PerceptualLoss:

    def __init__(self, features_count: int = 5):

        self.features_count = features_count
        self.vgg = Vgg19(features_count=features_count)
        self.vgg = nn.DataParallel(self.vgg)
        self.vgg = self.vgg.cuda()

    def __call__(self, observations: torch.Tensor, reconstructed_observations: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Computes the perceptual loss between the sets of observations

        :param observations: (..., 3, h, w) ground truth observations. Resolution is rescaled if needed
        :param reconstructed_observations: (..., 3, height, width) tensor with reconstructed observations

        :return: total_loss, individual_losses Perceptual loss between ground truth and reconstructed observations. Both the total loss and the loss
                 for each vgg feature level are returned
        '''

        original_observation_height = observations.size(-2)
        original_observation_width = observations.size(-1)
        height = reconstructed_observations.size(-2)
        width = reconstructed_observations.size(-1)

        flattened_ground_truth_observations, _ = TensorFolder.flatten(observations, -3)
        flattened_reconstructed_observations, _ = TensorFolder.flatten(reconstructed_observations, -3)

        # Resizes to the resolution of the reconstructed observations if needed
        if original_observation_width != width or original_observation_height != height:
            flattened_ground_truth_observations = F.interpolate(flattened_ground_truth_observations, (height, width), mode='bilinear')

        # Computes vgg features. Do not build the computational graph for the ground truth observations
        with torch.no_grad():
            ground_truth_vgg_features = self.vgg(flattened_ground_truth_observations.detach())

        reconstructed_vgg_features = self.vgg(flattened_reconstructed_observations)

        total_loss = torch.tensor([0.0], device=observations.device)
        single_losses = []
        # Computes the perceptual loss
        for current_ground_truth_feature, current_reconstructed_feature in zip(ground_truth_vgg_features, reconstructed_vgg_features):

            current_loss = torch.abs(current_ground_truth_feature.detach() - current_reconstructed_feature).mean()  # Detach signals to not backpropagate through the ground truth branch

            total_loss += current_loss
            single_losses.append(current_loss)

        return total_loss, single_losses


class PoseConsistencyLoss:

    def __init__(self):
        pass

    def __call__(self, previous_expected_positions: torch.Tensor, next_expected_positions: torch.Tensor, bounding_boxes_validity: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param previous_expected_positions: (... observations_count - 1, cameras_count, samples_per_image, 3) tensor with expected positions for the preceding frame
        :param next_expected_positions: (... observations_count - 1, cameras_count, samples_per_image, 3) tensor with expected positions for the successive frame
        :param bounding_boxes_validity: (... observations_count, cameras_count)
        :return:
        '''

        # Keeps only the positions corresponding to valid bounding boxes.
        # Valid bounding boxes are the ones that are present both in the preceding and in the successive frame
        previous_bounding_boxes_validity = bounding_boxes_validity[..., :-1, :]
        next_bounding_boxes_validity = bounding_boxes_validity[..., 1:, :]
        both_bounding_boxes_validity = torch.logical_and(previous_bounding_boxes_validity, next_bounding_boxes_validity)

        # Filters the positions corresponding to valid bounding boxes
        filtered_previous_expected_positions = previous_expected_positions[both_bounding_boxes_validity]
        filtered_next_expected_positions = next_expected_positions[both_bounding_boxes_validity]

        # Computes MSE
        loss = (filtered_previous_expected_positions - filtered_next_expected_positions).pow(2).mean()

        return loss


class KeypointConsistencyLoss:

    def __init__(self, confidence_threshold: float):
        '''

        :param confidence_threshold: Value under which not to consider the values associated an expected position
        '''
        self.confidence_threshold = confidence_threshold

    def __call__(self, expected_positions: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param expected_positions: (... observations_count, cameras_count, samples_per_image, 3) tensor with expected positions
        :param confidence: (... observations_count, cameras_count, samples_per_image) tensor with confidence scores
        :return:
        '''

        observations_count = expected_positions.size(-4)
        flat_expected_positions, _ = TensorFolder.flatten(expected_positions, -4)
        flat_confidence, _ = TensorFolder.flatten(confidence, -3)

        # (..., observations_count, observations_count, cameras_count, samples_per_image, 3)
        squared_errors = (flat_expected_positions.unsqueeze(-4) - flat_expected_positions.unsqueeze(-5)).pow(2)

        # Excludes squared error positions where the confidence of one of the samples is below the threshold
        # (..., observations_count, observations_count, cameras_count, samples_per_image, 3)
        exclusion_map = torch.logical_or(
            flat_confidence.unsqueeze(-3).unsqueeze(-1).repeat(1, 1, observations_count, 1, 1, 3) < self.confidence_threshold,
            flat_confidence.unsqueeze(-4).unsqueeze(-1).repeat(1, observations_count, 1, 1, 1, 3) < self.confidence_threshold
        )
        valid_positions_count = exclusion_map.sum()

        # Removes errors in the exclusion positions
        squared_errors = squared_errors * (1 - exclusion_map.long())

        # Computes the mean on the valid positions only
        loss = squared_errors.sum() / (valid_positions_count + 1e-6)

        return loss


class KeypointOpacityLoss:

    def __init__(self, confidence_threshold: float):
        '''

        :param confidence_threshold: Value under which not to consider the values associated an opacity
        '''
        self.confidence_threshold = confidence_threshold

    def __call__(self, opacity: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        '''
        Computes the loss between observations and reconstructed observations

        :param opacity: (..., samples_per_image) tensor with opacities
        :param confidence: (..., samples_per_image) tensor with confidence scores
        :return:
        '''

        # Considers the values with confidence above threshold
        mask = confidence > self.confidence_threshold
        opacity = opacity[mask]

        loss = (1 - opacity).pow(2).mean()

        return loss


class SquaredL2NormLoss:

    def __init__(self):
        '''

        '''
        pass

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        '''
        Computes the squared L2 norm of the given features

        :param features: (..., features_count, height, width) tensor with expected positions
        :return:
        '''

        loss = features.pow(2).sum(-3).mean()

        return loss


class GANLoss(nn.Module):
    """
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label against which to compute the loss must be the one for real images or for fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss


class ACMV:
    '''
    Action Conditioned Movement Variance
    '''

    def __init__(self):
        '''

        '''
        pass

    def __call__(self, movements: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        '''
        Computes the ACMV for the given movements and actions

        :param movements: (..., dim_count) tensor with movements
        :param actions: (..., actions_count) tensor with probability distribution over actions
        :return:
        '''

        movements, _ = TensorFolder.flatten(movements, -1)
        actions, _ = TensorFolder.flatten(actions, -1)

        elements_count = movements.size(0)

        eps = 1e-4

        global_action_distribution = actions.sum(0)
        # (actions_count, 1)
        global_action_distribution = global_action_distribution.unsqueeze(-1)

        # (elements, actions_count, dim_count)
        expected_action_movements = movements.unsqueeze(1) * actions.unsqueeze(-1)
        # (actions_count, dim_count)
        expected_action_movements = expected_action_movements.sum(0)

        # Mean vector of movement for each action
        # (actions_count, dim_count)
        action_means = expected_action_movements / (global_action_distribution + eps)

        # (elements_count, actions_count, dim_count)
        squared_differences = (movements.unsqueeze(1) - action_means.unsqueeze(0)).pow(2)

        # (elements_count, actions_count, dim_count)
        numerator = squared_differences * actions.unsqueeze(-1)
        # scalar
        numerator = numerator.sum(dim=[0, 1]).mean() / elements_count

        # (dim_count)
        denominator = movements.var(dim=0, unbiased=False)
        denominator = denominator.mean()

        return numerator / (denominator + eps)

if __name__ == "__main__":

    movements = torch.tensor([
        [0.0, 1.0],
        [0.0, 1.2],
        [0.0, 0.9],
        [0.0, 2.0],
        [0.0, 2.1],
        [0.0, 2.2],
        [1.0, 1.0],
        [1.2, 1.0],
        [1.0, 1.2],
        [2.2, 2.0],
        [2.1, 2.0],
        [2.0, 2.2],
    ])

    actions = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.9, 0.1],
        [0.0, 0.0, 0.1, 0.9],
        [0.0, 0.0, 0.1, 0.9],
        [0.0, 0.0, 0.1, 0.9],
    ])

    '''movements = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    actions = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])'''

    acmv = ACMV()
    loss = acmv(movements, actions)

    print(loss)