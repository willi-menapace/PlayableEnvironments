import copy
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video import Video
from dataset.video_dataset import MulticameraVideoDataset
from evaluation.action_modifiers.zero_variation_action_modifier import ZeroVariationActionModifier
from model.utils.object_ids_helper import ObjectIDsHelper
from utils.average_meter import AverageMeter
from utils.drawing.image_helper import ImageHelper


class ReconstructedPlayabilityDatasetCreator:
    '''
    Helper class for the creation of reconstruction quality evaluation datasets
    '''

    def __init__(self, config, dataset: MulticameraVideoDataset, logger, name="test"):
        '''

        :param config: Configuration file
        :param dataset: The dataset to reconstruct
        :param logger:
        :param name: prefix for the logger and name of the directory in the results folder where to save the reconstruction results
        '''

        self.config = config
        self.logger = logger
        self.name = name
        self.dataset = dataset

        self.dataloader = DataLoader(dataset, batch_size=self.config["playable_model_evaluation"]["reconstructed_dataset_batching"]["batch_size"], shuffle=False, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["playable_model_evaluation"]["reconstructed_dataset_batching"]["num_workers"], pin_memory=True)

        # Directory which will serve as the root for the output
        self.output_root = os.path.join(self.config["logging"]["reconstructed_playability_dataset_directory"], name)
        Path(self.output_root).mkdir(exist_ok=True)

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        # Helper for loggin the images
        self.image_helper = ImageHelper(self.config, logger, name)

    def output_paths_from_observations_paths(self, observations_paths: np.ndarray):
        '''
        Gets the paths where to save reconstructed images

        :param observations_paths: (batch_size, observations_count, cameras_count) string array with location on disk of the observations. None if the observation is not stored on disk
        :return:
        '''

        # Copies to avoid modifying the original array
        observations_paths = observations_paths.copy()

        # Iterates over all paths with a writable iterator
        with np.nditer(observations_paths, flags=["refs_ok"], op_flags=['readwrite']) as iterator:
            for current_path in iterator:

                # Splits the path .../sequence_name/camera_name/file_name
                path = os.path.normpath(current_path.item()).split(os.sep)[-3:]
                # Concatenates the path with the new output root
                # output_root/sequence_name/camera_name/file_name
                translated_path = os.path.join(self.output_root, os.path.join(*path))

                current_path[...] = translated_path

        return observations_paths

    def metadata_output_paths_from_observations_paths(self, metadata_paths: np.ndarray):
        '''
        Gets the paths where to save metadata

        :param metadata_paths: (batch_size, cameras_count) string array with location on disk of the metadata. None if the observation is not stored on disk
        :return:
        '''

        # Copies to avoid modifying the original array
        metadata_paths = metadata_paths.copy()

        # Iterates over all paths with a writable iterator
        with np.nditer(metadata_paths, flags=["refs_ok"], op_flags=['readwrite']) as iterator:
            for current_path in iterator:

                # Splits the path .../sequence_name/camera_name/file_name
                path = os.path.normpath(current_path.item()).split(os.sep)[-3:-1]
                # Concatenates the path with the new output root
                # output_root/sequence_name/camera_name
                translated_path = os.path.join(self.output_root, os.path.join(*path), Video.metadata_filename)

                current_path[...] = translated_path

        return metadata_paths

    def make_folders_for_output(self, paths: np.ndarray):
        '''
        Creates a folder corresponding to each of the paths

        :param paths: (...) string array with paths to files
        :return:
        '''

        # Iterates over all directories
        with np.nditer(paths, flags=["refs_ok"]) as iterator:
            for current_path in iterator:

                directory_path = current_path.item()

                # Strips the file name from the path if it is not a directory path
                if not os.path.isdir(directory_path):
                    directory_path = os.path.normpath(current_path.item()).split(os.sep)[:-1]
                    directory_path = os.path.join(*directory_path)

                # Creates the directory
                Path(directory_path).mkdir(exist_ok=True, parents=True)

    def compute_dynamic_object_parameters(self, object_results: Dict) -> Tuple:
        '''
        Computes dynamic object rotations, translations, style and deformation starting from the results output by the
        playable environment model
        :param object_results: object results output of the playable environment model
        :return: (bs, observations_count, 3, dynamic_objects_count) tensor with object rotations
                 (bs, observations_count, 3, dynamic_objects_count) tensor with object translations
                 (bs, observations_count, style_features_count, dynamic_objects_count) tensor with object style features
                 (bs, observations_count, deformation_features_count, dynamic_objects_count) tensor with object deformation features
        '''

        dynamic_objects_count = self.object_id_helper.dynamic_objects_count

        all_rotations = []
        all_translations = []
        all_style_features = []
        all_deformation_features = []
        # Extracts the reconstructed values for each object
        for dynamic_object_idx in range(dynamic_objects_count):
            current_results = object_results[dynamic_object_idx]

            all_rotations.append(current_results["reconstructed_object_rotations"])
            all_translations.append(current_results["reconstructed_object_translations"])
            all_style_features.append(current_results["reconstructed_object_style"])
            all_deformation_features.append(current_results["reconstructed_object_deformation"])

        # Stacks the values along the object dimension
        all_rotations = torch.stack(all_rotations, dim=-1)
        all_translations = torch.stack(all_translations, dim=-1)
        all_style_features = torch.stack(all_style_features, dim=-1)
        all_deformation_features = torch.stack(all_deformation_features, dim=-1)

        return all_rotations, all_translations, all_style_features, all_deformation_features

    def use_first_observation(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Creates a tensor where the first observation is used in all observation steps

        :param object_style: (bs, observations_count, ...)
        :return: (bs, observations_count, ...) tensor with the first observation repeated in place of all observations
        '''

        observations_count = tensor.size(1)
        tensor = tensor[:, :1] # Selects the first observation

        # Repeats the first observation
        repeats = [1] * len(tensor.size())
        repeats[1] = observations_count
        tensor = tensor.repeat(repeats)

        return tensor

    def reconstruct_dataset(self, model):
        '''
        Evaluates the performances of the given model

        :param model: The model to use for dataset reconstruction
        :return:
        '''

        loss_averager = AverageMeter()
        self.logger.print(f"- Reconstructing dataset '{self.name}'")

        batches_count = len(self.dataloader)

        # Saves sample images
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):

                self.logger.print(f"-- [{batch_idx:04d}/{batches_count:04d}] [{datetime.now()}] Reconstructing batch")

                # Performs inference
                batch_tuple = batch.to_tuple()
                observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple
                observations_paths = batch.observations_paths  # Paths where the current observations are stored on disk
                metadata = copy.deepcopy(batch.metadata)

                batch_size = observations.size(0)
                observations_count = observations.size(1)
                cameras_count = observations.size(2)

                # Forwards the batch through the model
                scene_encoding, object_results = model(observations, camera_rotations,
                                                        camera_translations,
                                                        focals,
                                                        bounding_boxes, bounding_boxes_validity,
                                                        global_frame_indexes, video_frame_indexes,
                                                        video_indexes,
                                                        ground_truth_observations=1,
                                                        shuffle_style=False,
                                                        action_modifier=ZeroVariationActionModifier())

                # Extracts the scene encodings
                camera_rotations = scene_encoding["camera_rotations"]
                camera_translations = scene_encoding["camera_translations"]
                focals = scene_encoding["focals"]
                object_rotations = scene_encoding["object_rotation_parameters"]
                object_translations = scene_encoding["object_translation_parameters"]
                object_style = scene_encoding["object_style"]
                object_deformation = scene_encoding["object_deformation"]

                # Computes the number of objects
                static_objects_count = self.object_id_helper.static_objects_count
                dynamic_objects_count = self.object_id_helper.dynamic_objects_count
                objects_count = self.object_id_helper.objects_count

                # Substitutes the values for camera and static object parameters with the ones of the first observation
                #camera_rotations = self.broadcast_first_observation(camera_rotations)
                #camera_translations = self.broadcast_first_observation(camera_translations)
                #focals = self.broadcast_first_observation(focals)
                static_object_rotations = object_rotations[..., :static_objects_count]
                static_object_translations = object_translations[..., :static_objects_count]
                static_object_style = object_style[..., :static_objects_count]
                static_object_deformation = object_deformation[..., :static_objects_count]

                # Computes the values for the reconstructed dynamic objectss
                dynamic_object_rotations, dynamic_object_translations, dynamic_object_style, dynamic_object_deformation = \
                    self.compute_dynamic_object_parameters(object_results)

                # Concatenates static and dynamic object reconstructed values together
                object_rotations = torch.cat([static_object_rotations, dynamic_object_rotations], dim=-1)
                object_translations = torch.cat([static_object_translations, dynamic_object_translations], dim=-1)
                object_style = torch.cat([static_object_style, dynamic_object_style], dim=-1)
                object_style = self.use_first_observation(object_style)
                object_deformation = torch.cat([static_object_deformation, dynamic_object_deformation], dim=-1)

                # Object is set to always be in the scene
                object_in_scene = torch.ones((object_rotations.size(0), object_rotations.size(1), object_rotations.size(-1)),
                                                dtype=torch.bool, device=object_rotations.device)

                image_size = (observations.size(-2), observations.size(-1))  # (height, width)

                # Renders the results
                render_results = model.module.environment_model.render_full_frame_from_scene_encoding(camera_rotations,
                                                                                         camera_translations,
                                                                                         focals, image_size,
                                                                                         object_rotations,
                                                                                         object_translations,
                                                                                         object_style,
                                                                                         object_deformation,
                                                                                         object_in_scene,
                                                                                         perturb=False,
                                                                                         samples_per_image_batching=60,
                                                                                         upsample_factor=1.0)

                # Gets output filenames and creates output directories for the current batch
                output_paths = self.output_paths_from_observations_paths(observations_paths)
                # Gets output filenames for metadata in the current batch
                metadata_output_paths = self.metadata_output_paths_from_observations_paths(observations_paths)
                self.make_folders_for_output(output_paths)

                # Saves the images
                self.image_helper.save_reconstructed_images_to_paths(render_results, output_paths)

                dynamic_objects_count = self.object_id_helper.dynamic_objects_count

                # Saves the metadata
                for batch_idx in range(batch_size):
                    for camera_idx in range(cameras_count):
                        # Makes copy to avoid writing in the same dictionary should dictionary metadata refer to the same object
                        current_metadata = [dict(**metadata[batch_idx][observation_idx][camera_idx]) for observation_idx in range(observations_count)]
                        current_metadata_output_path = metadata_output_paths[batch_idx, camera_idx].item()
                        for observation_idx, current_observation_metadata in enumerate(current_metadata[:-1]):  # For the last metadata element there are no actions

                            current_observation_metadata["model"] = "ours"
                            current_observation_metadata["inferred_action"] = []
                            current_observation_metadata["encoded_action"] = []

                            for dynamic_object_idx in range(dynamic_objects_count):
                                current_player_results = object_results[dynamic_object_idx]
                                inferred_actions = current_player_results["sampled_actions"]
                                sampled_action_directions = current_player_results["sampled_action_directions"]

                                # Gets inferred action and action direction
                                current_inferred_action = torch.argmax(inferred_actions[batch_idx, observation_idx]).item()
                                current_sampled_action_direction = sampled_action_directions[batch_idx, observation_idx].cpu().numpy()

                                current_observation_metadata["model"] = "ours"
                                current_observation_metadata["inferred_action"].append(current_inferred_action)
                                current_observation_metadata["encoded_action"].append(current_sampled_action_direction)
                        current_metadata[-1]["model"] = "ours"

                        # Writes the metadata file
                        with open(current_metadata_output_path, "wb") as file:
                            pickle.dump(current_metadata, file)

        self.logger.print(f"- Syncing dataset metadata '{self.name}'")

        # Uses rsync to sync the dataset pkl files, escluding metadata
        original_dataset_directory = os.path.join(self.config["data"]["data_root"], self.name) + "/"
        rsync_args = ["rsync", "-a", "--ignore-times", "--exclude", "*.png", "--exclude", f"*{Video.metadata_filename}", original_dataset_directory, self.output_root]  # Forces to overwrite files independently from time
        subprocess.run(rsync_args)

        self.logger.print(f"- Done reconstructing dataset '{self.name}'")

        return


def dataset_creator(config, dataset: MulticameraVideoDataset, logger, name="test"):
    return ReconstructedPlayabilityDatasetCreator(config, dataset, logger, name)
