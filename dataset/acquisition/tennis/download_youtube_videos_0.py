import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


def download_video(video_id: str, output_directory: str):
    '''
    Downloads a youtube video. The name of the saved video is its youtube id
    :param video_id: youtube video id to download
    :param output_directory: directory where to save output
    :return:
    '''

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_filepath = os.path.join(output_directory, f"{video_id}.%(ext)s")
    command_parameters = ["youtube-dl", "-f", "bestvideo[height>=1080,ext=mp4]+bestaudio/best[height>=1080,ext=m4a]", "--output", output_filepath,  url]
    print(command_parameters)
    subprocess.run(command_parameters)


if __name__ == "__main__":

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--video_list", type=str, required=True)

    arguments = parser.parse_args()

    output_directory = arguments.output_directory
    video_list = arguments.video_list
    Path(output_directory).mkdir(exist_ok=True, parents=True)

    video_dataframe = pd.read_csv(video_list)

    videos_count = video_dataframe.shape[0]
    for index, row in video_dataframe.iterrows():
        current_video_id = row["id"]

        print(f"- Downloading [{index + 1}/{videos_count}] '{current_video_id}'")

        download_video(current_video_id, output_directory)

    print("All done")

