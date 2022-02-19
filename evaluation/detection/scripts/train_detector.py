# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from evaluation.detection.engine import train_one_epoch, evaluate
import evaluation.detection.utils as utils
import evaluation.detection.transforms as T
from evaluation.detection.multicamera_video_dataset_detection_adapter import MulticameraVideoDatasetDetectionAdapter
from evaluation.detection.object_detector import get_object_detection_model


def torch_to_pil(img):
    return torchvision.transforms.ToPILImage()(img).convert('RGB')


def plot_img_bbox(img, target, path):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height

    img = torch_to_pil(img)

    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.savefig(path)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    checkpoints_path = "checkpoints/detection_model_minecraft"
    Path(checkpoints_path).mkdir(exist_ok=True)
    output_latest_filename = os.path.join(checkpoints_path, f"latest.pth.tar")
    max_steps_per_epoch = 1000
    batch_size = 8

    dataset_path = "data/minecraft_v1"
    dataset_path_train = os.path.join(dataset_path, "train")
    dataset_path_test = os.path.join(dataset_path, "test")
    boxes_expansion_factor = (2.6, 1.0)  # (rows, cols)

    resize_height = 288
    resize_width = 512

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda')

    # our dataset has two classes only - background and dynamic object
    num_classes = 2

    # Creates the datasets
    dataset = MulticameraVideoDatasetDetectionAdapter(dataset_path_train, get_transform(train=True), (resize_height, resize_width), boxes_expansion_factor)
    dataset_test = MulticameraVideoDatasetDetectionAdapter(dataset_path_test, get_transform(train=False), (resize_height, resize_width), boxes_expansion_factor)

    # Saves sample annotations
    print("Saving sample images")
    for i in range(1, 1000, 100):
        sample_image, sample_annotation = dataset[i]
        plot_img_bbox(sample_image, sample_annotation, f"results/sample_detection_{i:05d}.pdf")

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset = torch.utils.data.Subset(dataset, indices[:-50])
    #dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_object_detection_model(num_classes)
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Load previous state if present
    if os.path.isfile(output_latest_filename):
        print("Previous checkpoint detected, loading checkpoint")
        loaded_state = torch.load(output_latest_filename)
        model.load_state_dict(loaded_state["model"])
        lr_scheduler.load_state_dict(loaded_state["lr_scheduler"])
        optimizer.load_state_dict(loaded_state["optimizer"])

    # move model to the right device
    model.to(device)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"- [{epoch:05d}] Train")
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, max_steps_per_epoch=max_steps_per_epoch)
        # update the learning rate
        lr_scheduler.step()

        status_dictionary = {
            "model": model.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        output_filename = os.path.join(checkpoints_path, f"epoch_{epoch:05d}.pth.tar")
        torch.save(status_dictionary, output_filename)
        torch.save(status_dictionary, output_latest_filename)

        print(f"- [{epoch:05d}] Evaluation")
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("Training finished")


if __name__ == "__main__":
    main()