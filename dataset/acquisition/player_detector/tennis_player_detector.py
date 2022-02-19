from typing import Tuple, List

import torch
import torch.nn as nn
import torchvision

from utils.tensor_folder import TensorFolder


class TennisPlayerDetector(nn.Module):

    def __init__(self):
        super(TennisPlayerDetector, self).__init__()

        # Loads the model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
        self.model.eval()
        self.threshold = 0.75

        self.COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def point_in_trapetioid(self, x, y, points: torch.Tensor):
        '''
        Checks whether a point is contained within a trapetioid
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :param points: (top left, top right, bottom left, bottom right) tensor
        :return:
        '''

        # If points is higher than the upper base
        if y < min(points[0, 1].item(), points[1, 1].item()):
            return False

        # If point is lower than the lower base
        if y > max(points[2, 1].item(), points[3, 1].item()):
            return False

        # If x is on the left of the left boundary line
        delta_x = (points[2, 0] - points[0, 0]) / (points[2, 1] - points[0, 1])
        delta_y = (y - points[0, 1])
        if x < (points[0, 0] + delta_x * delta_y).item():
            return False

        # If x is on the right of the right boundary line
        delta_x = (points[3, 0] - points[1, 0]) / (points[3, 1] - points[1, 1])
        delta_y = (y - points[1, 1])
        if x > (points[1, 0] + delta_x * delta_y).item():
            return False

        return True

    def check_box_boundaries(self, box: Tuple, bounding_points: torch.Tensor):
        '''
        Checks whether the detection is compatible with the boundaries

        :param box: (left, top, right, bottom) tuple with current box
        :param bounding_points (6, 2) tensor with x and y image coordinates of detected points that form the boundaries of the
                             region where players are allowed (top left, top right, mid left, mid right, bottom left, bottom right)
        :return: 0 if not in boundaries
                 1 if in upper part of the field
                 2 if in lower part of the field
        '''

        base_x = (box[0] + box[2]) / 2
        base_y = box[3]

        # If the point is neighter in the upper, nor in the lower portion, discard the detection
        if self.point_in_trapetioid(base_x, base_y, bounding_points[:4]):
            return 1
        if self.point_in_trapetioid(base_x, base_y, bounding_points[2:]):
            return 2

        return 0

    def check_box_dimensions(self, box: Tuple, image_size: Tuple[int, int]):
        '''
        Checks whether the detection is compatible with the boundaries

        :param box: (left, top, right, bottom) tuple with current box
        :param image_size: (height, width) of the image
        :return: 0 if not valid dimension
                 1 if dimension is valid
        '''

        image_height = image_size[0]
        box_height = box[3] - box[1]

        # Position of the box inside the image
        box_bottom_position = box[3] / image_height

        # If the box is too small, discard it.
        # Accounts for heads that sometime appear at the bottom of the image
        if box_bottom_position > 0.6 and box_height / image_height < 55 / 560:
            return 0
        # But if the box is above half of the field, then make threshold less stringent to avoid losing player detections
        elif box_height / image_height < 25 / 560:
            return 0

        return 1

    def compute_center(self, box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def eliminate_duplicates(self, bounding_boxes: List, bounding_points: torch.Tensor, field_part: int):
        '''
        Returns a list with a single element

        :param bounding_boxes: list of (left, top, right, bottom) bounding boxes
        :param bounding_points (6, 2) tensor with x and y image coordinates of detected points that form the boundaries of the
                             region where players are allowed (top left, top right, mid left, mid right, bottom left, bottom right)
        :param field_part: 1 if players are in the upper part
                           2 if players are in the lower part
        :return:
        '''

        # Computes the reference line, a line positioned between the center net and the lowest barrier of the playing ground
        half_line = sum(bounding_points[2:4, 1].cpu().numpy().tolist()) / 2
        if field_part == 1:
            reference_line = sum(bounding_points[:2, 1].cpu().numpy().tolist()) / 2
            reference_line = (reference_line + half_line) / 2
        elif field_part == 2:
            reference_line = sum(bounding_points[-2:, 1].cpu().numpy().tolist()) / 2
            reference_line = (reference_line + half_line) / 2
        else:
            raise Exception(f"Invalid field part {field_part}")

        # Sorts the bounding boxes according to their distance to the reference line
        bounding_boxes.sort(key=lambda x: abs(x[3].item() - reference_line))
        # Returns the closest box
        return [bounding_boxes[0]]

    def forward(self, observations: torch.tensor, bounding_points: torch.tensor):
        '''
        Computes the bounding boxes for players in a field

        :param observations: (..., channels, height, width) tensor with observations
        :param: bounding_points (..., 6, 2) tensor with x and y image coordinates of detected points that form the boundaries of the
                             region where players are allowed (top left, top right, mid left, mid right, bottom left, bottom right)
        :return: (..., 4, 2) tensor with normalized bounding box coordinates (left, top, bottom, right) in [0, 1]
                 (..., 2) boolean tensor with True if that bounding box is valid or not
        '''

        flat_observations, initial_dimensions = TensorFolder.flatten(observations, dimensions=-3)
        flat_bounding_points, _ = TensorFolder.flatten(bounding_points, dimensions=-2)

        elements_count = flat_observations.size(0)
        image_height = flat_observations.size(-2)
        image_width = flat_observations.size(-1)

        # Computes positions one sequence at a time
        all_bounding_boxes = []
        all_is_valid = []

        with torch.no_grad():
            predictions = self.model(flat_observations)

        # Elaborates each batch element
        for current_prediction, current_bounding_points in zip(predictions, flat_bounding_points):

            pred_class = [self.COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(current_prediction['labels'].cpu().numpy())]
            pred_boxes = [(i[0], i[1], i[2], i[3]) for i in list(current_prediction['boxes'].detach().cpu().numpy())]
            pred_score = list(current_prediction['scores'].detach().cpu().numpy())
            filtered_preds = [pred_score.index(x) for x in pred_score if x > self.threshold]
            if len(filtered_preds) > 0:
                pred_t = filtered_preds[-1]
                pred_boxes = pred_boxes[:pred_t + 1]
                pred_class = pred_class[:pred_t + 1]
            else:
                pred_boxes = []
                pred_class = []

            #match_found = False
            matches = {
                1: [],  # Matches in upper part of the field
                2: [],  # Matches in lower part of the field
            }
            for idx in range(len(pred_boxes)):
                if pred_class[idx] == 'person':
                    check_result = self.check_box_dimensions(pred_boxes[idx], (image_height, image_width))
                    # Discards all boxes that are too small to be considered
                    if check_result:
                        check_result = self.check_box_boundaries(pred_boxes[idx], current_bounding_points)
                        if check_result != 0:
                            matches[check_result].append(pred_boxes[idx])

            is_valid = [True, True]
            # Eliminates duplicates and fills empty detections
            for idx in range(1, 3):
                if len(matches[idx]) > 1:
                    print(f"Warning: {matches[idx]} players in the same part of the field, using the one with y closest to the reference line")
                    matches[idx] = self.eliminate_duplicates(matches[idx], current_bounding_points, idx)
                if len(matches[idx]) == 0:
                    print(f"Warning: no players in part {idx} of the field")
                    # If detection is empty, this bounding box is not valid
                    is_valid[idx - 1] = False
                    matches[idx].append([0.0, 0.0, 0.0, 0.0])

            # Forms the current tensors, there is exactly one tensor per match [0]
            current_bounding_boxes = torch.stack([torch.tensor(matches[2][0]), torch.tensor(matches[1][0])], dim=-1)  # (4, 2) tensor
            current_is_valid = torch.tensor(list(reversed(is_valid)))  # (2) tensor

            all_bounding_boxes.append(current_bounding_boxes)
            all_is_valid.append(current_is_valid)

        all_bounding_boxes = torch.stack(all_bounding_boxes, dim=0)
        all_is_valid = torch.stack(all_is_valid, dim=0)

        folded_bounding_boxes = TensorFolder.fold(all_bounding_boxes, initial_dimensions)
        folded_is_valid = TensorFolder.fold(all_is_valid, initial_dimensions)

        return folded_bounding_boxes, folded_is_valid



