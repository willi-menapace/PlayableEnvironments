import argparse
import glob
import multiprocessing as mp
import os
import pickle
from datetime import datetime


def process_video(video_path):
    '''
    Patches the video annotation by adding missing annotations

    :param video_path: directory for the video to patch
    :return:
    '''

    print(f"-- [{datetime.now()}] Computing video {video_path}")
    video_path = os.path.join(video_path, "00000")

    # Gets the number of frames by looking at the number of focal lengths
    focals_filename = os.path.join(video_path, "focals.pkl")
    actions_filename = os.path.join(video_path, "actions.pkl")
    rewards_filename = os.path.join(video_path, "rewards.pkl")
    metadata_filename = os.path.join(video_path, "metadata.pkl")
    dones_filename = os.path.join(video_path, "dones.pkl")
    with open(focals_filename, "rb") as file:
        focals = pickle.load(file)

    frames_count = len(focals)

    # Creates default values
    actions = [0] * frames_count
    dones = [False] * frames_count
    rewards = [0.0] * frames_count
    metadata = [{}] * frames_count

    # Outputs the missing annotation
    with open(actions_filename, "wb") as file:
        pickle.dump(actions, file)
    with open(rewards_filename, "wb") as file:
        pickle.dump(rewards, file)
    with open(metadata_filename, "wb") as file:
        pickle.dump(metadata, file)
    with open(dones_filename, "wb") as file:
        pickle.dump(dones, file)

    print(f"-- [{datetime.now()}] Finished {video_path}")


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_directory", type=str, required=True, help="Directory with splitted youtube videos")
    parser.add_argument("--workers", type=int, default=16)

    arguments = parser.parse_args()

    dataset_directory = arguments.dataset_directory
    workers = arguments.workers

    root_directories = ["train", "test", "val"]
    root_directories = [os.path.join(dataset_directory, current_directory) for current_directory in root_directories]

    # Gets the paths to all video directories to patch
    all_video_paths = []
    for current_directory in root_directories:
        video_paths = list(glob.glob(os.path.join(current_directory, "*")))
        print(os.path.join(current_directory, "*"))
        video_paths = [current_path for current_path in video_paths if os.path.isdir(current_path)]
        all_video_paths.extend(video_paths)

    pool = mp.Pool(workers)

    print("- Start patching dataset frames")
    #for item in work_items:
    #    process_video(item)
    pool.map(process_video, all_video_paths)
    pool.close()
    print("- All Done")


if __name__ == "__main__":
    '''
    Adds missing annotation files such as actions, dones and metadata
    '''

    main()