from typing import Tuple
import os

from PIL import Image


class LazyImage:
    '''
    Class representing an image lying on disk rather than in memory
    '''

    def __init__(self, path: str, crop_size: Tuple[int]=None, target_size: Tuple[int]=None):
        '''

        :param path: path where the image resides on disk
        :param crop_size: (left_index, upper_index, right_index, lower_index) size of the crop to take before resizing
        :param target_size: (width, height) desired size of the image after retrieval
        '''

        if not os.path.isfile(path):
            raise Exception(f'{path} does not exist')

        self.path = path
        self.crop_size = crop_size
        self.target_size = target_size

    def get_image(self) -> Image:
        '''
        Gets the image from disk
        :return:
        '''

        image = Image.open(self.path)

        # Crops the image
        if self.crop_size is not None:
            image = image.crop(self.crop_size)

        # Resizes the image if the target size differs from the current one
        original_width, original_height = image.size
        if self.target_size is not None and (original_width != self.target_size[0] or original_height != self.target_size[1]):
            image = image.resize(self.target_size, Image.BICUBIC)

        return image