from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageDraw
from PIL.Image import Image

from utils.tensor_folder import TensorFolder


class BoundingBoxDrawer:

    @staticmethod
    def torch_to_pil(images: torch.Tensor) -> List[Image]:
        '''
        Converts a tensor to a list of PIL images
        :param images: (elements_count, 3, height, width) tensor with images
        :return: List of elements_count PIL images
        '''

        pil_images = []
        to_pil_transform = transforms.ToPILImage()

        elements_count = images.size(0)
        for idx in range(elements_count):
            current_image = images[idx]

            current_pil_image = to_pil_transform(current_image)
            pil_images.append(current_pil_image)

        return pil_images

    @staticmethod
    def pil_to_torch(images: List[Image]) -> torch.Tensor:
        '''
        Converts a list of PIL images to a tensor
        :param images: List of elements_count PIL images
        :return: (elements_count, 3, height, width) tensor with images
        '''

        tensors = []
        to_tensor_transform = transforms.ToTensor()

        for current_image in images:
            current_tensor = to_tensor_transform(current_image)
            tensors.append(current_tensor)

        tensors = torch.stack(tensors, dim=0)
        return tensors

    @staticmethod
    def draw_bounding_box(image: Image, bounding_box: np.ndarray, color=(255, 0, 0)):
        '''
        Draws a bounding box onto an image
        :param image: PIL image on where to draw the box
        :param bounding_box: (left, top, right, bottom) bounding box normalized in [0, 1]
        :param color: color for the bounding box
        :return:
        '''

        width, height = image.size

        draw = ImageDraw.Draw(image)

        # Denormalizes the boudning boxes
        bounding_box = bounding_box.copy() # Avoid overwriting the original
        bounding_box[0] *= width
        bounding_box[2] *= width
        bounding_box[1] *= height
        bounding_box[3] *= height

        draw.rectangle(bounding_box, fill=None, outline=color, width=2)

    @staticmethod
    def draw_3d_bounding_box(image: Image, bounding_box: np.ndarray, color=(255, 0, 0)):
        '''
        Draws a bounding box onto an image
        :param image: PIL image on where to draw the box
        :param bounding_box: (bounding_box_points, 2) bounding box normalized in [0, 1]
        :param color: color for the bounding box
        :return:
        '''

        width, height = image.size

        draw = ImageDraw.Draw(image)

        # Denormalizes the boudning boxes
        bounding_box = bounding_box.copy()  # Avoid overwriting the original
        bounding_box[:, 0] *= width
        bounding_box[:, 1] *= height
        bounding_box = [(pair[0] - 3, pair[1] - 3, pair[0] + 3, pair[1] + 3) for pair in bounding_box.tolist()]  # Makes each bounding box point a rectangle

        for current_rectangle in bounding_box:
            draw.rectangle(current_rectangle, fill=None, outline=color, width=1)

    @staticmethod
    def draw_axes_on_image(image: Image, axes_points: np.ndarray):
        '''
        Draws a bounding box onto an image
        :param image: PIL image on where to draw the box
        :param axes_points: (4, 2) axes projections normalized in [0, 1], in order origin, x_axis, y_axis, z_axis
        :return:
        '''

        width, height = image.size

        draw = ImageDraw.Draw(image)

        # Denormalizes the boudning boxes
        axes_points = axes_points.copy()  # Avoid overwriting the original
        axes_points[:, 0] *= width
        axes_points[:, 1] *= height

        for idx in range(1, 4):
            current_line = (tuple(axes_points[0].tolist()), tuple(axes_points[idx].tolist()))
            current_color = [0] * 3
            current_color[idx - 1] = 255
            draw.line(current_line, fill=tuple(current_color), width=2)

    @staticmethod
    def draw_bounding_boxes(images: torch.Tensor, bounding_boxes: torch.Tensor, color=(255, 0, 0)) -> torch.Tensor:
        '''
        Draws bouding boxes on the given image tensors

        :param images: (..., 3, height, width) tensor with images
        :param bounding_boxes: (..., 4) tensor with bouding boxes normalized in [0, 1]
        :param color: color for the boxes
        :return: (..., 3, height, width) tensor with images with bouding boxes on them
        '''

        flat_images, initial_dimensions = TensorFolder.flatten(images, -3)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -1)

        pil_images = BoundingBoxDrawer.torch_to_pil(flat_images)

        for idx in range(len(pil_images)):
            current_image = pil_images[idx]
            current_bounding_box = flat_bounding_boxes[idx].detach().cpu().numpy()

            BoundingBoxDrawer.draw_bounding_box(current_image, current_bounding_box, color=color)

        flat_images = BoundingBoxDrawer.pil_to_torch(pil_images)
        images = TensorFolder.fold(flat_images, initial_dimensions)

        return images

    @staticmethod
    def draw_3d_bounding_boxes(images: torch.Tensor, bounding_boxes: torch.Tensor, color=(255, 0, 0)) -> torch.Tensor:
        '''
        Draws bouding boxes on the given image tensors

        :param images: (..., 3, height, width) tensor with images
        :param bounding_boxes: (..., bounding_box_points, 2) tensor with bouding box points normalized in [0, 1]
        :param color: color for the boxes
        :return: (..., 3, height, width) tensor with images with bouding boxes on them
        '''

        flat_images, initial_dimensions = TensorFolder.flatten(images, -3)
        flat_bounding_boxes, _ = TensorFolder.flatten(bounding_boxes, -2)

        pil_images = BoundingBoxDrawer.torch_to_pil(flat_images)

        for idx in range(len(pil_images)):
            current_image = pil_images[idx]
            current_bounding_box = flat_bounding_boxes[idx].detach().cpu().numpy()

            BoundingBoxDrawer.draw_3d_bounding_box(current_image, current_bounding_box, color=color)

        flat_images = BoundingBoxDrawer.pil_to_torch(pil_images)
        images = TensorFolder.fold(flat_images, initial_dimensions)

        return images

    @staticmethod
    def draw_axes(images: torch.Tensor, axes_points: torch.Tensor) -> torch.Tensor:
        '''
        Draws axes on the given image tensors

        :param images: (..., 3, height, width) tensor with images
        :param axes_points: (..., 4, 2) axes projections normalized in [0, 1], in order origin, x_axis, y_axis, z_axis
        :return: (..., 3, height, width) tensor with images axes on them
        '''

        flat_images, initial_dimensions = TensorFolder.flatten(images, -3)
        flat_axes_points, _ = TensorFolder.flatten(axes_points, -2)

        pil_images = BoundingBoxDrawer.torch_to_pil(flat_images)

        for idx in range(len(pil_images)):
            current_image = pil_images[idx]
            current_axes = flat_axes_points[idx].detach().cpu().numpy()

            BoundingBoxDrawer.draw_axes_on_image(current_image, current_axes)

        flat_images = BoundingBoxDrawer.pil_to_torch(pil_images)
        images = TensorFolder.fold(flat_images, initial_dimensions)

        return images

