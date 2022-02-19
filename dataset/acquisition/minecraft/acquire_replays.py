import argparse
import glob
import multiprocessing as mp
import os

from dataset.acquisition.minecraft.splitted_recording import SplittedRecording


def get_file_info(directory: str, video_extension: str):
    '''
    Gets names of video files and annotation files
    :param directory: directory where to search the files
    :param video_extension: extension of the video files
    :return: list of (video_filename, annotation_filename, split_annotation_filename)
    '''

    all_videos = list(sorted(glob.glob(os.path.join(directory, f"*.{video_extension}"))))
    all_annotations = list(sorted(glob.glob(os.path.join(directory, f"*.json"))))
    all_split_annotations = list(sorted(glob.glob(os.path.join(directory, f"*.txt"))))

    if len(all_videos) != len(all_annotations) or len(all_videos) != len(all_split_annotations):
        raise Exception("Not all videos have an associated annotation or split annotation file")

    return list(zip(all_videos, all_annotations, all_split_annotations))


def split_video(splitted_recording: SplittedRecording, begin_index: int, output_directory: str):
    '''
    Splits a given video
    :param splitted_recording:
    :param begin_index:
    :param output_directory:
    :return:
    '''

    splitted_recording.output_video_frames(output_directory, begin_index)
    splitted_recording.output_annotations(output_directory, begin_index)


if __name__ == "__main__":
    '''
    Acquires ReplayMod replays putting them in the required dataset format
    '''

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--video_extension", type=str, default="mp4")
    parser.add_argument("--workers", type=int, default=4)

    arguments = parser.parse_args()

    workers_count = arguments.workers

    replays_directory = arguments.replays_directory
    output_directory = arguments.output_directory
    video_extension = arguments.video_extension

    all_split_recordings = []
    all_begin_indexes = []

    # Gets all the splits
    current_begin_index = 0
    files = get_file_info(replays_directory, video_extension)
    for current_video, current_annotation, current_split_annotation in files:
        current_splitted_recording = SplittedRecording(current_video, current_annotation, current_split_annotation)
        all_split_recordings.append(current_splitted_recording)
        all_begin_indexes.append(current_begin_index)

        # The next directory index is incremented by the number of splits that will be produced for the current video
        current_begin_index += current_splitted_recording.get_splits_count()

    all_output_directories = [output_directory] * len(all_split_recordings)

    with mp.Pool(processes=workers_count) as pool:
        results = pool.starmap(split_video, zip(all_split_recordings, all_begin_indexes, all_output_directories))

    print("Dataset produced")

