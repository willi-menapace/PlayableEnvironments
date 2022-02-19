import torch
from torch.utils.data import DataLoader

from dataset.batching import single_batch_elements_collate_fn
from dataset.video_dataset import MulticameraVideoDataset
from model.utils.object_ids_helper import ObjectIDsHelper
from utils.average_meter import AverageMeter
from utils.drawing.image_helper import ImageHelper


class Evaluator:
    '''
    Helper class for model evaluation
    '''

    def __init__(self, config, dataset: MulticameraVideoDataset, logger, logger_prefix="test"):
        self.config = config
        self.logger = logger
        self.logger_prefix = logger_prefix
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"], shuffle=False, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)
        self.imaging_dataloader = DataLoader(dataset, batch_size=self.config["evaluation"]["batching"]["batch_size"], shuffle=True, collate_fn=single_batch_elements_collate_fn, num_workers=self.config["evaluation"]["batching"]["num_workers"], pin_memory=True)

        self.bounding_box_color = (255, 0, 0)
        self.ground_truth_bounding_box_color = (0, 0, 255)

        # Parameters for extra cameras for evaluation
        self.extra_cameras_rotations = torch.as_tensor(self.config["evaluation"]["extra_cameras"]["camera_rotations"], dtype=torch.float, device="cuda")
        self.extra_cameras_translations = torch.as_tensor(self.config["evaluation"]["extra_cameras"]["camera_translations"], dtype=torch.float, device="cuda")
        self.extra_cameras_focals = torch.as_tensor(self.config["evaluation"]["extra_cameras"]["camera_focals"], dtype=torch.float, device="cuda")

        # Helper for handling the relationships between object ids and their models
        self.object_id_helper = ObjectIDsHelper(self.config)

        # Helper for loggin the images
        self.image_helper = ImageHelper(self.config, logger, logger_prefix)

    def evaluate(self, model, step: int, log_only_global: bool=True):
        '''
        Evaluates the performances of the given model

        :param model: The model to evaluate
        :param step: The current step
        :param log_only_global: If true, images only for the global scene are rendered
        :return:
        '''

        loss_averager = AverageMeter()
        self.logger.print(f"- Saving sample images")
        # Saves sample images
        with torch.no_grad():
            for idx, batch in enumerate(self.imaging_dataloader):

                # Performs inference
                batch_tuple = batch.to_tuple()
                observations, actions, rewards, dones, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes = batch_tuple

                render_results = model.module.render_full_frame_from_observations(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, perturb=False, upsample_factor=1.0)

                ground_truth_observations = observations
                ground_truth_observations = ImageHelper.normalize(ground_truth_observations, (-1, +1))

                # Gets attention maps
                object_attention_maps = render_results["object_attention"]
                object_attention_maps = [torch.cat([current_tensor] * 3, dim=-3) for current_tensor in object_attention_maps]  # Transforms it in a 3 channel tensor

                reconstructed_bounding_boxes = render_results["reconstructed_bounding_boxes"]
                reconstructed_3d_bounding_boxes = render_results["reconstructed_3d_bounding_boxes"]
                projected_axes = render_results["projected_axes"]

                # Gets object crops
                object_crops = render_results["object_crops"]

                self.image_helper.save_images_from_results(render_results, step, "", ground_truth_observations, object_attention_maps, object_crops, bounding_boxes, reconstructed_bounding_boxes, reconstructed_3d_bounding_boxes, projected_axes, log_only_global=log_only_global)

                # Renders extra cameras
                scene_encodings = render_results["scene_encoding"]
                batch_size = observations.size(0)
                observations_count = observations.size(1)
                height = observations.size(-2)
                width = observations.size(-1)

                if self.extra_cameras_rotations.size(0) > 0:
                    # Adds the batch_size and observations_count dimensions
                    extra_camera_rotations = self.extra_cameras_rotations.unsqueeze(0).unsqueeze(0).repeat([batch_size, observations_count, 1, 1])
                    extra_camera_translations = self.extra_cameras_translations.unsqueeze(0).unsqueeze(0).repeat([batch_size, observations_count, 1, 1])
                    extra_cameras_focals = self.extra_cameras_focals.unsqueeze(0).unsqueeze(0).repeat([batch_size, observations_count, 1])
                    image_size = (height, width)
                    object_rotation_parameters = scene_encodings["object_rotation_parameters"]
                    object_translation_parameters = scene_encodings["object_translation_parameters"]
                    object_style = scene_encodings["object_style"]
                    object_deformation = scene_encodings["object_deformation"]
                    object_in_scene = scene_encodings["object_in_scene"]

                    render_results = model.module.render_full_frame_from_scene_encoding(extra_camera_rotations, extra_camera_translations, extra_cameras_focals, image_size,
                                                                                        object_rotation_parameters, object_translation_parameters, object_style, object_deformation,
                                                                                        object_in_scene, perturb=False, upsample_factor=1.0)

                    self.image_helper.save_images_from_results(render_results, step, "extra_cameras", log_only_global=log_only_global)

                break

        return


def evaluator(config, dataset: MulticameraVideoDataset, logger, logger_prefix="test"):
    return Evaluator(config, dataset, logger, logger_prefix)
