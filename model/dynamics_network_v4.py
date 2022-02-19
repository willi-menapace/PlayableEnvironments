from typing import Tuple, Dict

import torch
import torch.nn as nn

from model.layers.rotation_encoder import RotationEncoder
from utils.lib_3d.transformations_3d import Transformations3D


class DynamicsNetworkV4(nn.Module):
    '''
    Model for the dynamics of the environment
    '''

    def __init__(self, config: Dict, model_config: Dict):
        '''
        Initializes the model representing the dynamics of the environment

        :param config: the configuration file
        :param model_config: the configuration for the specific object
        '''
        super(DynamicsNetworkV4, self).__init__()

        self.config = config

        # Computes the number of input features
        style_features = model_config["style_features"]
        deformation_features = model_config["deformation_features"]
        actions_count = model_config["actions_count"]
        action_space_dimension = model_config["action_space_dimension"]
        self.features_count = [6, 3, style_features, deformation_features, actions_count, action_space_dimension]

        # Whether to force output elements to specific values
        self.rotation_axis = model_config["rotation_axis"]
        # None if should not be forced, otherwise the value to which should be forced
        self.force_rotation_axis_translations = model_config["force_rotation_axis_translations"]
        # Whether all output rotations should be 0
        self.force_rotations_zero = model_config["force_rotations_zero"]

        self.input_features = sum(self.features_count)
        self.output_features = model_config["output_features"]
        self.cells_count = model_config["cells_count"]  # Number of recurrent cells to use

        self.all_cells = nn.ModuleList()
        # Initializes the LSTM cells
        current_features = self.input_features
        for cell_idx in range(self.cells_count):
            current_cell = nn.LSTMCell(current_features, self.output_features)
            self.all_cells.append(current_cell)
            current_features = self.output_features

        # Learnable initial cell states
        self.all_initial_hidden_states = nn.ParameterList()
        self.all_initial_hidden_cell_states = nn.ParameterList()
        for cell_idx in range(self.cells_count):
            self.all_initial_hidden_states.append(nn.Parameter(torch.zeros(self.output_features)))
            self.all_initial_hidden_cell_states.append(nn.Parameter(torch.zeros(self.output_features)))

        # The current states
        self.all_current_hidden_states = []
        self.all_current_hidden_cell_states = []

        self.mlp_backbone = nn.Sequential(
            nn.Linear(self.output_features, self.output_features),
            nn.ReLU()
        )

        # Builds a head for each output parameter with the exception of the conditioning actions
        heads = [nn.Linear(self.output_features, current_features_count) for current_features_count in self.features_count[:-2]]
        self.mlp_heads = nn.ModuleList(heads)

    def reinit_memory(self):
        '''
        Initializes the state of the recurrent model
        :return:
        '''

        # Removes the stored state from the cell of present so that the next forward will reinitialize it
        # Warning: state is not created here directly, otherwise if the model is employed inside DataParallel,
        # the state would be created only in the original object rather than in the replicas on each GPU
        if len(self.all_current_hidden_states) > 0:
            self.all_current_hidden_states = []
            self.all_current_hidden_cell_states = []

    def get_memory_state(self):
        '''
        Obtains the current state of the memory

        :return: tuple of
                 (batch_size, features_count) tensor with hidden state
                 (batch_size, features_count) tensor with hidden cell state
                 None is state has not been initialized
        '''

        memory_state = (None, None)
        if len(self.all_current_hidden_states) > 0:
            memory_state = (self.all_current_hidden_states, self.all_current_hidden_cell_states)

        return memory_state

    def set_memory_state(self, memory_state: Tuple[torch.Tensor]):
        '''
        Sets the state of the memory with the given values. The memory state representation format is the one described in get_memory_state
        :return:
        '''

        hidden_state, hidden_cell_state = memory_state
        # If present restore the previous state
        if hidden_state is not None:
            self.all_current_hidden_states = hidden_state
            self.all_current_hidden_cell_states = hidden_cell_state
        # If there was no previous state, the cell must have been freshly initialized
        else:
            self.reinit_memory()

    def get_rotation_matrix(self, rotations: torch.Tensor) -> torch.Tensor:
        '''

        :param rotations: (bs, 3) tensor with rotations
        :return: (bs, 3, 3) tensor of rotation matrices around the rotation_axis
        '''

        if self.rotation_axis == 0:
            rotation_function = Transformations3D.rotation_matrix_x
        elif self.rotation_axis == 1:
            rotation_function = Transformations3D.rotation_matrix_y
        elif self.rotation_axis == 2:
            rotation_function = Transformations3D.rotation_matrix_z
        else:
            raise Exception(f"Invalid rotation axis {self.rotation_axis}")

        # Transforms in a matrix the rotation around the rotation axis
        rotation_matrices = rotation_function(rotations[..., self.rotation_axis])
        return rotation_matrices

    def forward(self, current_object_rotations: torch.Tensor, current_object_translations: torch.Tensor,
                current_object_style: torch.Tensor, current_object_deformation: torch.Tensor,
                current_action: torch.Tensor, current_action_variation: torch.Tensor) -> torch.Tensor:
        '''
        Generates the next state of the environment starting from the current state and the actions for the current step

        :param current_object_rotations: (bs, 3) tensor with rotations of the object
        :param current_object_translations: (bs, 3) tensor with translations of the object
        :param current_object_style: (bs, style_features_count) tensor with style of the object
        :param current_object_deformation: (bs, deformation_features_count) tensor with deformations of the object
        :param current_action: (bs, actions_count) tensor with current actions
        :param current_action_variation: (bs, action_space_dimension) tensor with variation for the current action

        :return: (bs, 3) tensor with next rotations of the object
                 (bs, 3) tensor with next translations of the object
                 (bs, style_features_count) tensor with next style of the object
                 (bs, deformation_features_count) tensor with next deformations of the object
        '''

        batch_size = current_object_rotations.size(0)

        # Checks if state must be initialized
        # Initializes memory by repeating for each batch element the learned initial values
        if len(self.all_current_hidden_states) == 0:
            # Unsure whether DataParallel may reference the same list in different processes, so reinitialize it defensively
            self.all_current_hidden_states = []
            self.all_current_hidden_cell_states = []
            for cell_idx in range(self.cells_count):
                self.all_current_hidden_states.append(self.all_initial_hidden_states[cell_idx].repeat((batch_size, 1)))
                self.all_current_hidden_cell_states.append(self.all_initial_hidden_cell_states[cell_idx].repeat((batch_size, 1)))

        current_encoded_rotations = RotationEncoder.encode(current_object_rotations, dim=-1)
        current_cell_input = torch.cat([current_encoded_rotations, current_object_translations, current_object_style, current_object_deformation,
                                   current_action, current_action_variation], dim=-1)

        # Forwards through the layers
        for idx, (current_cell, current_hidden_state, current_hidden_cell_state) in enumerate(zip(self.all_cells, self.all_current_hidden_states, self.all_current_hidden_cell_states)):
            next_hidden_state, next_hidden_cell_state = current_cell(current_cell_input, (current_hidden_state, current_hidden_cell_state))

            current_cell_input = next_hidden_state
            self.all_current_hidden_states[idx] = next_hidden_state
            self.all_current_hidden_cell_states[idx] = next_hidden_cell_state

        # Passes the lstm outputs through the mlp for decoding
        current_output = self.mlp_backbone(next_hidden_state)
        all_outputs = [current_head(current_output) for current_head in self.mlp_heads]

        delta_encoded_rotations, delta_translations, next_style, next_deformation = all_outputs
        delta_rotations = RotationEncoder.decode(delta_encoded_rotations, dim=-1)

        # If rotations are to be zeroed, we zero them all
        if self.force_rotations_zero:
            delta_rotations = delta_rotations * 0.0
        # Otherwise just zero the ones that do not refer to a rotation axis
        else:
            non_rotation_axes = set(range(3))
            non_rotation_axes.remove(self.rotation_axis)
            for current_non_rotation_axis in non_rotation_axes:
                delta_rotations[..., current_non_rotation_axis] *= 0

        next_rotations = current_object_rotations + delta_rotations

        # (bs, 3, 3) Rotates the delta translations to the world coordinate system so that they can be expressed with respect to the rotated object coordinate system
        current_object_rotation_matrices = self.get_rotation_matrix(current_object_rotations)
        # (bs, 3, 1) Translations expressed in the object coordinate system
        delta_translations = delta_translations.unsqueeze(-1)
        # (bs, 3). Translations expresesd in the world coordinate system
        rotated_delta_translations = torch.matmul(current_object_rotation_matrices, delta_translations).squeeze(-1)

        # Computes the next translations. Force them if specified
        next_translations = current_object_translations + rotated_delta_translations
        if self.force_rotation_axis_translations is not None:
            next_translations[..., self.rotation_axis] = self.force_rotation_axis_translations

        return next_rotations, next_translations, next_style, next_deformation


def model(config, model_config):
    '''
    Instantiates a dynamics network with the given parameters
    :param config:
    :param model_config:
    :return:
    '''
    return DynamicsNetworkV4(config, model_config)
