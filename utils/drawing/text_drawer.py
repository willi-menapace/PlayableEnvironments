from typing import Union, List

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from utils.tensor_folder import TensorFolder


class TextDrawer:

    @staticmethod
    def cross_product(first: Union[List[int], List[List[int]]], second: List[int]):
        '''
        Computes the cross product between the two lists
        :param first:
        :param second:
        :return:
        '''

        results = []
        for current_first in first:
            for current_second in second:
                if isinstance(current_first, int):
                    current_first = [current_first]
                results.append(current_first + [current_second])
        return results

    @staticmethod
    def expand(dimensions: List):
        '''
        Computes a list of indexes that index each element in a tensor with the given dimensions
        :param dimensions:
        :return:
        '''

        if len(dimensions) == 0:
            return []
        if len(dimensions) == 1:
            return [[element] for element in range(dimensions[0])]

        current_expansion = list(range(dimensions[0]))
        for idx in range(1, len(dimensions)):
            current_expansion = TextDrawer.cross_product(current_expansion, list(range(dimensions[idx])))

        return current_expansion

    @staticmethod
    def draw_text_on_bidimensional_batch(images: torch.Tensor, text: str, font_size: int = 16, position=(10, 260), color=(255, 255, 255)):
        '''

        :param images: (batch_size1, batch_size2, ..., 3, height, width) tensor with input images
        :param text: list of length batch_size1 of lists of size batch_size2 with text to draw on each image
        :param font_size: see draw_text_on_PIL
        :param position: see draw_text_on_PIL
        :param color: see draw_text_on_PIL
        :return: (batch_size, batch_size2, ..., 3, height, width) tensor with input images with drawn text
        '''

        flat_images, dimensions = TensorFolder.flatten(images, 2)
        flat_text = []
        for current_text in text:
            flat_text.extend(current_text)

        flat_images = TextDrawer.draw_text_on_batch(flat_images, flat_text, font_size, position, color)

        images = TensorFolder.fold(flat_images, dimensions)
        return images

    @staticmethod
    def draw_text_on_batch(images: torch.Tensor, text: str, font_size: int = 16, position=(10, 260), color=(255, 255, 255)):
        '''

        :param images: (batch_size, ..., 3, height, width) tensor with input images
        :param text: list of length batch_size with text to draw on each image
        :param font_size: see draw_text_on_PIL
        :param position: see draw_text_on_PIL
        :param color: see draw_text_on_PIL
        :return: (batch_size, ..., 3, height, width) tensor with input images with drawn text
        '''

        images = images.clone()

        batch_size = images.size(0)
        if len(text) != batch_size:
            raise Exception(f"Batch size and length of text differ {batch_size}:{len(text)}")

        # Each batch may contain multiple images. Compute the indices of each
        broadcast_dimensions = list(images.size())[1:-3]
        broadcast_indices = TextDrawer.expand(broadcast_dimensions)

        for batch_element_idx in range(batch_size):
            current_batch = images[batch_element_idx]
            current_text = text[batch_element_idx]

            # Draws text on each image in the current batch
            for current_broadcasted_indices in broadcast_indices:
                current_image = current_batch

                # Indexes the image at the current position
                for current_idx in current_broadcasted_indices:
                    current_image = current_image[current_idx]

                # Draws text and obtains a tensor with text
                current_image_with_text = TextDrawer.draw_text_on_tensor(current_image, current_text, font_size, position, color)

                # Copies the content of the image with text into the original image
                current_image[:] = current_image_with_text[:]

        return images

    @staticmethod
    def draw_text_on_tensor(image: torch.Tensor, text: str, font_size: int = 16, position=(10, 260), color=(255, 255, 255)):
        '''

        :param image: (3, height, width) iamge tensor on which to draw the text.
        :param text: see draw_text_on_PIL
        :param font_size: see draw_text_on_PIL
        :param position: see draw_text_on_PIL
        :param color: see draw_text_on_PIL
        :return: (3, height, width) tensor with the added text
        '''

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.PILToTensor()

        pil_image = to_pil(image)
        TextDrawer.draw_text_on_PIL(pil_image, text, font_size, position, color)

        image = to_tensor(pil_image) / 255.0
        return image

    @staticmethod
    def draw_text_on_PIL(image: Image, text: str, font_size: int=16, position=(10, 260), color=(255, 255, 255)):
        '''

        :param image: PIL image on which to draw. Image is modified in place
        :param text: text to draw
        :param font_size:
        :param position: position in the image where to draw the text
        :return:
        '''

        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype("utils/drawing/fonts/Roboto-Regular.ttf", font_size)
        draw.text(position, text, font=font, fill=color)


def main():

    images = torch.zeros((2, 4, 5, 3, 100, 200))
    text = ["1", "2"]

    images_with_text = TextDrawer.draw_text_on_batch(images, text)
    pass

if __name__ == "__main__":
    main()