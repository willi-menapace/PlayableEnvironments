#!/bin/bash

# Downloads youtube videos
python -m dataset.acquisition.tennis.download_youtube_videos_0 --output_directory data/youtube_videos --video_list dataset/acquisition/tennis/tennis_annotations/tennis_youtube_videos.csv

# Extracts the interesting video portions from the downloaded youtube videos. Outputs videos
# NOTE: if videos become unavailable on youtube the script will fail. Please replace the missing videos with placeholder
# videos of length greater than the original video and remove the corresponding sequences from the final dataset.
python -m dataset.acquisition.tennis.extract_sequences_from_youtube_1 --video_directory data/youtube_videos --output_directory data/tennis_splitted --annotations_path dataset/acquisition/tennis/tennis_annotations/tennis_youtube_splits_annotations.yaml --splits_duration 240.00

# Extracts the frames from the videos extracted at the previous step.
python -m dataset.acquisition.tennis.extract_frames_4 --video_directory data/tennis_splitted --output_directory data/tennis_v7_frames

# Makes train, val, test splits
python -m dataset.acquisition.tennis.make_train_val_test_split_6 --dataset_directory data/tennis_v7_frames --output_directory data/tennis_v7

# Makes test and val sets fixed length
python -m dataset.acquisition.tennis.make_fixed_length_8 --root_directories data/tennis_v7/val data/tennis_v7/test --output_directories data/tennis_v7/val_fixed_length data/tennis_v7/test_fixed_length
# Renames the directories
mv data/tennis_v7/val data/tennis_v7/val_variable_length
mv data/tennis_v7/test data/tennis_v7/test_variable_length
mv data/tennis_v7/val_fixed_length data/tennis_v7/val
mv data/tennis_v7/test_fixed_length data/tennis_v7/test

# Extracts the tennis data annotations
tar -xf data/tennis_v7_annotation.tar.xz -C data
# Synchronizes the annotation with the tennis dataset. Forces overwrite of metadata
rsync -aI data/tennis_v7_annotation/ data/tennis_v7
# Cleans up the annotations
rm -r data/tennis_v7_annotation





