import torch

from dataset.video_dataset import MulticameraVideoDataset
from evaluation.evaluator import Evaluator
from utils.average_meter import AverageMeter
from utils.drawing.image_helper import ImageHelper


class ImageDecoderEvaluator(Evaluator):
    '''
    Helper class for model evaluation
    '''

    def __init__(self, config, dataset: MulticameraVideoDataset, logger, logger_prefix="test"):

        super(ImageDecoderEvaluator, self).__init__(config, dataset, logger, logger_prefix)

        # Number of rays to sample for each image
        self.samples_per_image = config["training"]["samples_per_image"]

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

                # Renders with the image decoder to produce the main reconstruction result
                render_results = model.module(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, samples_per_image=self.samples_per_image, perturb=False, upsample_factor=1.0)

                # Gathers dimensions of the observations
                batch_size = observations.size(0)
                observations_count = observations.size(1)
                height = observations.size(-2)
                width = observations.size(-1)

                ground_truth_observations = observations
                ground_truth_observations = ImageHelper.normalize(ground_truth_observations, (-1, +1))

                # Gets attention maps
                object_attention_maps = render_results["object_attention"]
                object_attention_maps = [torch.cat([current_tensor] * 3, dim=-3) for current_tensor in object_attention_maps]  # Transforms it in a 3 channel tensor

                reconstructed_bounding_boxes = render_results["reconstructed_bounding_boxes"]

                # Gets object crops
                object_crops = render_results["object_crops"]

                self.image_helper.save_decoded_images_from_results(render_results, step, "", ground_truth_observations,  bounding_boxes, reconstructed_bounding_boxes)

                # Renders extra cameras
                scene_encodings = render_results["scene_encoding"]

                print("Warning, remove evaluation decoder disabling")
                use_dec = model.module.use_image_decoder
                model.module.use_image_decoder = False
                # Renders with classical nerf to produce additional results, eg depth maps
                render_results = model.module.render_full_frame_from_observations(observations, camera_rotations, camera_translations, focals, bounding_boxes, bounding_boxes_validity, global_frame_indexes, video_frame_indexes, video_indexes, perturb=False, upsample_factor=1.0)
                model.module.use_image_decoder = use_dec

                # Gets attention maps
                object_attention_maps = render_results["object_attention"]
                object_attention_maps = [torch.cat([current_tensor] * 3, dim=-3) for current_tensor in object_attention_maps]  # Transforms it in a 3 channel tensor

                self.image_helper.save_images_from_results(render_results, step, "", ground_truth_observations, object_attention_maps, object_crops, bounding_boxes, reconstructed_bounding_boxes, log_only_global=log_only_global)

                # Renders with the extra cameras using the image decoder
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

                    render_results = model.module(extra_camera_rotations, extra_camera_translations, extra_cameras_focals, image_size,
                                           object_rotation_parameters, object_translation_parameters, object_style, object_deformation,
                                           object_in_scene, samples_per_image=self.samples_per_image, perturb=False, mode="scene_encodings")

                    self.image_helper.save_decoded_images_from_results(render_results, step, "extra_cameras")

                break

        return


def evaluator(config, dataset: MulticameraVideoDataset, logger, logger_prefix="test"):
    return ImageDecoderEvaluator(config, dataset, logger, logger_prefix)
