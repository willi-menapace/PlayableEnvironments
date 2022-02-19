import argparse
import importlib
import os
import shutil
import sys
import time
from typing import List, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.multiprocessing
import torchvision
import numpy as np

import cv2 as cv
from PIL import Image

from dataset.dataset_splitter import DatasetSplitter
from dataset.transforms import TransformsGenerator, OpticalFlowTransformsGenerator
from dataset.video_dataset import MulticameraVideoDataset
from utils.configuration import Configuration
from utils.drawing.video_saver import VideoSaver
from utils.logger import Logger

torch.backends.cudnn.benchmark = True


def display_observation(observation: torch.Tensor, window_name: str, actions: List[int] = None, zoom_factor: int = 4):
    '''
    Displays the observation on the given window

    :param observation: (height, width, 3) tensor with values in [0, 1] with the observation to show
    :param window_name: Name of the window on which to display the frame
    :param actions: Optional list of actions to display on the frame
    :param zoom_factor: Number of times the output should be magnified
    :return:
    '''

    # Display the frame
    observation = (observation.cpu().numpy() * 255).astype(np.uint8)
    color_corrected_frame = np.copy(observation)
    color_corrected_frame[:, :, 0] = observation[:, :, 2]
    color_corrected_frame[:, :, 2] = observation[:, :, 0]
    display_frame = cv.resize(color_corrected_frame, (color_corrected_frame.shape[1] * zoom_factor, color_corrected_frame.shape[0] * zoom_factor))

    # Converts actions to strings and shows them on the frame
    if actions is not None:
        actions_text = ""
        for current_action in actions:
            if actions_text:
                actions_text += ", "
            actions_text += str(current_action + 1)
        display_frame = video_saver.draw_text_on_frame(display_frame, (40, 20), actions_text, pointsize=128)

    cv.imshow(window_name, display_frame)


def print_state(state: Dict[str, torch.Tensor]):
    '''
    Prints information on the current state
    :param state:
    :return:
    '''

    object_rotations = state["object_rotation_parameters"][0, 0]
    object_translations = state["object_translation_parameters"][0, 0]

    object_rotations = torch.transpose(object_rotations, 0, 1)
    object_translations = torch.transpose(object_translations, 0, 1)

    object_rotations = object_rotations / (3.1415 * 2) * 360

    print(f"Object rotations: {object_rotations.detach().cpu().numpy()}")
    print(f"Object translations: {object_translations.detach().cpu().numpy()}")


def read_action(current_actions_count: int):
    '''
    Reads the current action from the currently open window
    :param current_actions_count: The number of actions for the current object
    :return: The currently read action
    '''
    # Asks for input until a correct one is received
    success = False
    while not success:
        success = False
        try:
            print(f"\n- Insert current action in [1, {current_actions_count}] for object {dynamic_object_idx}, 0 to reset: ")
            # current_action = int(input_helper.read_character())
            current_action = int(cv.waitKey(0)) - ord('0')

            current_action -= 1  # Puts the action in the expected range for the model
            if current_action != -1 and (current_action < 0 or current_action >= current_actions_count):
                success = False
            else:
                success = True

        except Exception as e:
            time.sleep(0.1)
            success = False

    print(f"Action read: {current_action + 1}")
    return current_action

save_directory = f"results/play_results_{datetime.now()}"
image_extension = "png"
framerate = 5

sample_action_variations = False
automatic = False
#automatic_actions = [0] * 10
automatic_actions = list(range(0, 7))
frames_per_action = 15
image_width = 512

if __name__ == "__main__":

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    arguments = parser.parse_args()

    config_path = arguments.config

    configuration = Configuration(config_path)
    configuration.check_config()
    configuration.create_directory_structure()

    config = configuration.get_config()

    logger = Logger(config)
    search_name = config["model"]["architecture"]
    environment_model = getattr(importlib.import_module(search_name), 'model')(config)
    environment_model.cuda()

    datasets = {}

    dataset_splits = DatasetSplitter.generate_playable_model_splits(config)
    transformations = TransformsGenerator.get_final_transforms(config)
    optical_flow_transformations = OpticalFlowTransformsGenerator.get_final_transforms(config)

    for key in dataset_splits:
        path, batching_config = dataset_splits[key]
        transform = transformations[key]
        optical_flow_transform = optical_flow_transformations[key]

        datasets[key] = MulticameraVideoDataset(path, batching_config, transform, optical_flow_transform)

    # Creates trainer and evaluator
    environment_model_trainer = getattr(importlib.import_module(config["training"]["trainer"]), 'trainer')(config, environment_model, datasets["train"], logger)

    # Loads the environment model
    try:
        environment_model_trainer.load_checkpoint(environment_model)
    except Exception as e:
        logger.print(e)
        raise Exception("Could not load environment model. Playable model training aborted")

    # Creates the playable model
    environment_model.cuda()
    environment_model.eval()
    search_name = config["playable_model"]["architecture"]
    playable_environment_model = getattr(importlib.import_module(search_name), 'model')(config, environment_model)
    playable_environment_model.cuda()
    # Creates the trainer and evaluator for the playable model
    playable_environment_model_trainer = getattr(importlib.import_module(config["playable_model_training"]["trainer"]), 'trainer')(config, playable_environment_model, datasets["train"], logger)
    evaluator_inferred_actions = getattr(importlib.import_module(config["playable_model_evaluation"]["evaluator"]), 'evaluator')(config, playable_environment_model_trainer, datasets["validation"], logger, logger_prefix="playable_model_validation")

    # Loads the playable model
    try:
        playable_environment_model_trainer.load_checkpoint(playable_environment_model)
    except Exception as e:
        logger.print(e)
        raise Exception("Could not load playable environment model")

    playable_environment_model.cuda()

    logger.get_wandb().watch(playable_environment_model, log='all')

    # Uses the evaluation dataloader
    dataloader = evaluator_inferred_actions.dataloader
    # Retrieves the number of actions for each model
    actions_count = []
    for current_animation_model_idx, current_animation_model in enumerate(playable_environment_model.object_animation_models):
        objects_count = playable_environment_model.object_id_helper.objects_count_by_animation_model_idx(current_animation_model_idx)
        for _ in range(objects_count):
            actions_count.append(current_animation_model.actions_count)

    print(f"Detected {len(actions_count)} animated objects with {actions_count} actions")

    # Erases and creates the new directory
    print(f"- Erasing '{save_directory}'")
    if os.path.isdir(save_directory):
        shutil.rmtree(save_directory)
    os.mkdir(save_directory)
    current_sequence_idx = 0

    # Initializes window for rendering
    window_name = "rendered_frame"
    cv.namedWindow(window_name)

    video_saver = VideoSaver()

    # Infers one sequence
    while True:
        # Gets the first batch
        for batch in dataloader:
            pass

        # Creates the output directory
        current_output_directory = os.path.join(save_directory, str(current_sequence_idx))
        video_filename = os.path.join(current_output_directory, "video.mp4")
        gif_filename = os.path.join(current_output_directory, "video.gif")
        os.mkdir(current_output_directory)
        print(f"- Saving output to '{current_output_directory}'")

        # Performs inference
        batch_tuple = batch.to_tuple()
        observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple
        image_size = (observations.size(-2), observations.size(-1))

        with torch.no_grad():
            current_observation, current_state = playable_environment_model.initialize_interactive_generation(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, 0, 0)

        current_observation_idx = 0
        all_observations = []
        while True:
            # Prints debug information
            print_state(current_state)

            display_observation(current_observation, window_name, zoom_factor=1)
            int_observation = (current_observation.cpu().numpy() * 255).astype(np.uint8)
            all_observations.append(int_observation)

            # Saves the frame as an image
            pil_image = Image.fromarray(int_observation)
            pil_image.save(os.path.join(current_output_directory, f"{current_observation_idx}.{image_extension}"))

            # Collects action inputs for the current step
            action_inputs = []
            for dynamic_object_idx, current_actions_count in enumerate(actions_count):

                if automatic:
                    current_action = automatic_actions[current_sequence_idx]
                else:
                    current_action = read_action(current_actions_count)

                action_inputs.append(current_action)
                if current_action == -1:
                    break

            # Terminates the current sequence
            if current_action == -1 or (automatic and current_observation_idx == frames_per_action):

                all_observations = np.stack(all_observations, axis=0)
                video_saver.save_video(all_observations, video_filename, framerate)
                video_saver.video_to_gif(video_filename, gif_filename, framerate, image_width)

                break

            with torch.no_grad():
                current_observation, current_state = playable_environment_model.generate_next(action_inputs, current_state, image_size, sample_action_variations, use_initial_style=True)

            # Next observation
            current_observation_idx += 1

        # Next sequence
        current_sequence_idx += 1

        # If all actions have been generated terminate the program
        if automatic and current_sequence_idx == len(automatic_actions):
            break

    print("Program end")
