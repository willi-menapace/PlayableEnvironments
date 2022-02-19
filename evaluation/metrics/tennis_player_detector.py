from typing import Tuple

import torch
import torch.nn as nn
import torchvision


class TennisPlayerDetector(nn.Module):

    def __init__(self):
        super(TennisPlayerDetector, self).__init__()

        # Loads the model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
        self.model.eval()
        self.threshold = 0.8

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

    def compute_normalized_center(self, box: torch.Tensor, image_size: Tuple[int, int], device):
        '''

        :param box: (left, top, right, bottom) Tensor with current box
        :param image_size: (height, width) of the image
        :return: (rows, columns) numpy array with the box center
        '''

        return torch.as_tensor(((box[1] + box[3]) / 2 / image_size[0], (box[0] + box[2]) / 2 / image_size[1]), device=device)

    def compute_normalized_boxes(self, box: torch.Tensor, image_size: Tuple[int, int], device):
        '''

        :param box: (left, top, right, bottom) Tensor with current box
        :param image_size: (height, width) of the image
        :return: (rows, columns) numpy array with the box center
        '''

        box = box.clone()
        size_tensor = torch.as_tensor([image_size[1], image_size[0], image_size[1], image_size[0]], device=device)

        return box / size_tensor

    def forward(self, observations, batching=8):
        '''
        Computes the mean squared error between the reference and the generated observations

        :param observations: (bs, channels, height, width) tensor with observations. Must be normalized
        :param batching: maximum batch size to use during the computation
        :return: bs lists with (2) tensors with x and y coordinates of the detection
                 bs lists with (4) tensors with (left, top, right, bottom) normalized coordinates of the detected box
        '''

        batch_size = observations.size(0)
        image_size = observations.size(-2), observations.size(-1)

        # Computes positions one sequence at a time
        all_predicted_centers = []
        all_predicted_boxes = []

        # Computes a portion of the batch at a time of maximum size 'batching'
        current_end_idx = batching
        while True:
            current_end_idx = min(current_end_idx, batch_size)
            with torch.no_grad():
                predictions = self.model(observations[current_end_idx - batching:current_end_idx])

            for current_prediction in predictions:
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

                current_predicted_centers = []
                current_predicted_boxes = []
                for idx in range(len(pred_boxes)):
                    if pred_class[idx] == 'person':
                        if self.check_box_dimensions(pred_boxes[idx], image_size):
                            current_predicted_centers.append(self.compute_normalized_center(pred_boxes[idx], image_size, device=observations.device))
                            current_predicted_boxes.append(self.compute_normalized_boxes(torch.as_tensor(pred_boxes[idx], device=observations.device), image_size, device=observations.device))

                all_predicted_centers.append(current_predicted_centers)
                all_predicted_boxes.append(current_predicted_boxes)

            if current_end_idx == batch_size:
                break
            current_end_idx += batching

        return all_predicted_centers, all_predicted_boxes



