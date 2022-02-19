import argparse
import multiprocessing as mp
import os
from datetime import datetime

from dataset.acquisition.utils.video_fragmenter import VideoFragmenter


def process_video(args):
    '''
    Extracts the frames for a certain video, outputting them in the output directory specified by the output index

    :param video_path: directory for the original videos
    :param output_root: root where the dataset is to be created
    :param output_index: index of the current video
    :param output_framerate: framerate for the output video
    :param output_size: (width, height) of the output video
    :param image_extension:
    :return:
    '''

    video_path, output_root, output_index, output_framerate, output_size, image_extension = args


    print(f"-- [{datetime.now()}] Computing video {video_path}")
    output_directory = os.path.join(output_root, f"{output_index:05d}", "00000")

    # Clean possible leftovers from previous executions
    VideoFragmenter.clean_frames(output_directory, image_extension)

    # If the frames have already been successfully extracted
    if os.path.isfile(os.path.join(output_directory, "frame_extraction_success.txt")):
        print(f"-- [{datetime.now()}] Video frames have already been extracted succesfully for {video_path}")

        return

    # If a failure happened during previous preprocessing, do not extract frames
    if os.path.isfile(os.path.join(output_directory, "failure.txt")) or os.path.isfile(os.path.join(output_directory, "boxes_failure.txt")):
        print(f"-- [{datetime.now()}] Video {video_path} failed in previous phases, skipping")

        return

    print(f"-- [{datetime.now()}] Extracting frames of {video_path}")
    # Extracts frames from the video and creates the output directory
    frames = VideoFragmenter.extract_frames(video_path, output_directory, output_framerate, output_size, image_extension)

    # Creates a file marking success in the extraction of frames
    success_filename = "frame_extraction_success.txt"
    success_path = os.path.join(output_directory, success_filename)
    open(success_path, 'a').close()


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_directory", type=str, required=True, help="Directory with splitted youtube videos")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory where to output the dataset")
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--image_extension", type=str, default="png")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--framerate", type=int, default=5, help="Dataset output framerate")
    parser.add_argument("--height", type=int, default=288, help="Dataset output height")
    parser.add_argument("--width", type=int, default=512, help="Dataset output width")

    arguments = parser.parse_args()

    video_directory = arguments.video_directory
    output_directory = arguments.output_directory
    video_extension = arguments.video_extension
    image_extension = arguments.image_extension
    workers = arguments.workers
    framerate = arguments.framerate
    height = arguments.height
    width = arguments.width

    output_size = (width, height)

    # Gets all input videos
    input_videos = VideoFragmenter.get_videos(video_directory, video_extension)

    # Creates work items for splitting the videos
    work_items = []
    for video_idx, current_video in enumerate(input_videos):
        current_work_item = current_video, output_directory, video_idx, framerate, output_size, image_extension
        work_items.append(current_work_item)

    pool = mp.Pool(workers)

    print("- Start extracting frames")
    #for item in work_items:
    #    process_video(item)
    pool.map(process_video, work_items)
    pool.close()
    print("- All Done")


if __name__ == "__main__":
    '''
    Extracts the subsequences of the youtube videos that contain relevant content
    '''

    main()