#!/bin/bash

# Transforms videos into our dataset format
python -m dataset.acquisition.minecraft.acquire_replays --replays_directory data/minecraft_dataset_raw --output_directory data/tmp_1_minecraft_v1
# Makes each sequence of the same length
python -m dataset.acquisition.minecraft.make_fixed_length --root_directory data/tmp_1_minecraft_v1 --output_directory data/tmp_2_minecraft_v1 --skip_frames 0 --sequence_length 400 --min_sequence_length 400
# Makes train and validation splits (use dataset/acquisition/minecraft/create_train_test_split_annotations.py to create the splits.csv file)
python -m dataset.acquisition.minecraft.train_val_test_split --splits_file dataset/acquisition/minecraft/minecraft_annotations/splits.csv --root_directory data/tmp_2_minecraft_v1 --output_directory data/minecraft_v1 --copy
# Adds nested folder representing camera directory
python -m dataset.acquisition.create_camera_folder --directory data/minecraft_v1