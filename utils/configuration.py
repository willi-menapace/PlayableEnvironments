import collections

import yaml
import os
from pathlib import Path

from utils.dict_wrapper import DictWrapper


class Configuration:
    '''
    Represents the configuration parameters for running the process
    '''

    def __init__(self, path):
        '''
        Initializes the configuration with contents from the specified file
        :param path: path to the configuration file in json format
        '''

        # Loads configuration file
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = DictWrapper(config)

    def get_config(self):
        return self.config

    def check_config(self):
        '''
        Raises an exception if the configuration is invalid and creates auxiliary fields
        :return:
        '''

        if not os.path.isdir(self.config["data"]["data_root"]):
            raise Exception(f"Data directory {self.config['data']['data_root']} does not exist")

        self.config["logging"]["output_directory"] = os.path.join(self.config["logging"]["output_root"], self.config["logging"]["run_name"])
        self.config["logging"]["checkpoints_root_directory"] = os.path.join(self.config["logging"]["checkpoints_root"], self.config["logging"]["run_name"])
        self.config["logging"]["playable_model_checkpoints_directory"] = os.path.join(self.config["logging"]["checkpoints_root_directory"], "playable_model_checkpoints")

        self.config["logging"]["output_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "images")
        self.config["logging"]["reconstructed_dataset_directory"] = os.path.join(self.config["logging"]["output_directory"], "reconstructed_dataset")
        self.config["logging"]["reconstructed_camera_manipulation_dataset_directory"] = os.path.join(self.config["logging"]["output_directory"], "reconstructed_camera_manipulation_dataset")
        self.config["logging"]["reconstructed_playability_dataset_directory"] = os.path.join(self.config["logging"]["output_directory"], "reconstructed_playability_dataset")
        self.config["logging"]["reconstructed_playability_legacy_dataset_directory"] = os.path.join(self.config["logging"]["output_directory"], "reconstructed_legacy_playability_dataset")
        self.config["logging"]["camera_trajectory_dataset_directory"] = os.path.join(self.config["logging"]["output_directory"], "camera_trajectory_dataset")
        self.config["logging"]["camera_trajectory_amt_directory"] = os.path.join(self.config["logging"]["output_directory"], "camera_trajectory_amt")
        self.config["logging"]["evaluation_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "evaluation_images")
        self.config["logging"]["style_storage_directory"] = os.path.join(self.config["logging"]["output_directory"], "style_storage")
        self.config["logging"]["teaser_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "teaser_images")
        self.config["logging"]["style_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "style_images")
        self.config["logging"]["playability_qualitatives_directory"] = os.path.join(self.config["logging"]["output_directory"], "playability_qualitatives")
        self.config["logging"]["camera_motion_grid_directory"] = os.path.join(self.config["logging"]["output_directory"], "camera_motion_grid")

        # Checks whether it is necessary or not to split the dataset
        if not "dataset_splits" in self.config["data"]:
            self.config["data"]["dataset_style"] = "splitted"
        else:
            self.config["data"]["dataset_style"] = "flat"
            if len(self.config["data"]["dataset_splits"]) != 3:
                raise Exception("Dataset splits must speficy exactly 3 elements")
            if sum(self.config["data"]["dataset_splits"]) != 1.0:
                raise Exception("Dataset splits must sum to 1.0")

        # If crop is not specified set the key anyways
        if not "crop" in self.config["data"]:
            self.config["data"]["crop"] = None

        # If evaluation frequency is not specified set the key to always evaluate after each epoch
        if not "eval_freq" in self.config["evaluation"]:
            self.config["evaluation"]["eval_freq"] = 0

        if not "max_evaluation_batches" in self.config["evaluation"]:
            self.config["evaluation"]["max_evaluation_batches"] = None
        if not "max_evaluation_batches" in self.config["playable_model_evaluation"]:
            self.config["playable_model_evaluation"]["max_evaluation_batches"] = None

        if not "max_steps_per_epoch" in self.config["training"]:
            self.config["training"]["max_steps_per_epoch"] = 10000

        # If the patching parameters are not set, set them to 0
        if not "patches_per_image" in self.config["training"]:
            self.config["training"]["patches_per_image"] = 0
        if not "patch_size" in self.config["training"]:
            self.config["training"]["patch_size"] = 0
        if not "perceptual_features" in self.config["training"]:
            self.config["training"]["perceptual_features"] = 5

        # By default do not use the head selection loss
        if not "head_selection_cross_entropy_loss_lambda" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["head_selection_cross_entropy_loss_lambda"] = 0.0

        # By default do not use camera parameters correction
        if not "enable_camera_parameters_offsets" in self.config["model"]:
            self.config["model"]["enable_camera_parameters_offsets"] = False
            self.config["training"]["camera_parameters_learning_rate"] = 0.0
        if not "camera_parameters_memory_size" in self.config["model"]:
            self.config["model"]["camera_parameters_memory_size"] = 1

        if not "perceptual_object_masking" in self.config["training"]:
            self.config["training"]["perceptual_object_masking"] = "none"

        # By default do not use the pose consistency loss
        if not "pose_consistency_lambda" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["pose_consistency_lambda"] = 0.0

        # By default use the same samples per image for pose consistency
        if not "pose_consistency_samples_per_image" in self.config["training"]:
            self.config["training"]["pose_consistency_samples_per_image"] = self.config["training"]["samples_per_image"]

        # By default use the same samples per image for keypoint consistency
        if not "keypoint_consistency_samples_per_image" in self.config["training"]:
            self.config["training"]["keypoint_consistency_samples_per_image"] = self.config["training"]["samples_per_image"]

        # By default do not use the keypoint consistency loss and the keypoint consistency score threshold
        if not "keypoint_consistency_loss_lambda" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["keypoint_consistency_loss_lambda"] = 0.0
        if not "keypoint_consistency_loss_threshold" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["keypoint_consistency_loss_threshold"] = 0.0

        # By default do not use the keypoint opacity loss and the keypoint consistency score threshold
        if not "keypoint_opacity_loss_lambda" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["keypoint_opacity_loss_lambda"] = 0.0
        if not "keypoint_opacity_loss_threshold" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["keypoint_opacity_loss_threshold"] = 0.0
        # By default do not use annealing on the keypoint opacity loss
        if not "keypoint_opacity_loss_max_steps" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["keypoint_opacity_loss_max_steps"] = 0

        # By default apply sigmoid to the model outputs
        if not "apply_activation" in self.config["model"]:
            self.config["model"]["apply_activation"] = True

        # By default use l2 for the features reconstruction loss
        if not "autoencoder_features_reconstruction_loss_type" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["autoencoder_features_reconstruction_loss_type"] = "l2"

        # By default do not normalize the features reconstruction loss
        if not "autoencoder_features_reconstruction_loss_normalize" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["autoencoder_features_reconstruction_loss_normalize"] = False

        has_autoencoder = "autoencoder" in self.config["model"]

        # By default the downsampled factor is computed based on the number of downsampling layers
        if has_autoencoder and not "downsample_factor" in self.config["model"]["autoencoder"]:
            downsampling_layers_count = self.config["model"]["autoencoder"]["downsampling_layers_count"]
            current_stride = 1
            self.config["model"]["autoencoder"]["downsample_factor"] = []
            # If multiple downsampling layers are present, create a list of strides
            if isinstance(downsampling_layers_count, collections.Sequence):
                for current_downsampling_layers_count in downsampling_layers_count:
                    current_stride = current_stride * (2 ** current_downsampling_layers_count)
                    self.config["model"]["autoencoder"]["downsample_factor"].append(current_stride)
            # Otherwise compute the stride for the only layer
            else:
                self.config["model"]["autoencoder"]["downsample_factor"] = downsampling_layers_count ** 2

        # By default do not exclude the encoder from training
        if has_autoencoder and not "exclude_encoder" in self.config["model"]["autoencoder"]:
                self.config["model"]["autoencoder"]["exclude_encoder"] = False

        # By default do not use patching
        if not "patch_size" in self.config["training"]:
            self.config["training"]["patch_size"] = 0
        # By default do not align sampled rays
        if not "align_grid" in self.config["training"]:
            self.config["training"]["align_grid"] = False
        # By default do not crop tensors to patch before loss computation
        if not "crop_to_patch" in self.config["training"]:
            self.config["training"]["crop_to_patch"] = False

        # By default do not align sampled rays
        if not "image_save_interval" in self.config["training"]:
            self.config["training"]["image_save_interval"] = 100

        # By default do not freeze the batch norm layers when the rest of the autoencoder is frozen
        if has_autoencoder and "also_freeze_bn" not in self.config["model"]["autoencoder"]:
            self.config["model"]["autoencoder"]["also_freeze_bn"] = False

        # By default do not use radial weights
        if "use_radial_weights" not in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["use_radial_weights"] = False

        # By default do not use profiling
        if "enable_profiling" not in self.config["training"]:
            self.config["training"]["enable_profiling"] = False

        # By eliminate rays of the static model that intersect with dynamic objects
        if "fix_object_overlaps" not in self.config["model"]:
            self.config["model"]["fix_object_overlaps"] = True

        # Use by default the plain fvd evaluator for fvd
        if "dataset_fvd_reconstruction_evaluator" not in self.config["evaluation"]:
            self.config["evaluation"]["dataset_fvd_reconstruction_evaluator"] = "evaluation.reconstructed_dataset_fvd_evaluator"

        # Sets default optimizer betas for adam
        if "betas" not in self.config["playable_model_training"]:
            self.config["playable_model_training"]["betas"] = (0.9, 0.999)

        # By default do not fix discriminator lr scheduler update bug
        if "fix_discriminator_lr_update" not in self.config["playable_model_training"]:
            self.config["playable_model_training"]["fix_discriminator_lr_update"] = False

        # By default do not fix discriminator lr scheduler update bug
        if "detach_translation" not in self.config["playable_model"]:
            self.config["playable_model"]["detach_translation"] = False

        # By default do not use acmv
        if "acmv_lambda" not in self.config["playable_model_training"]["loss_weights"]:
            self.config["playable_model_training"]["loss_weights"]["acmv_lambda"] = 0.0

        # By default use a bounding box that results in no normalization
        if "discriminator_bounding_box" not in self.config["playable_model"]:
            self.config["playable_model"]["discriminator_bounding_box"] = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]

        # By default do not enable anomaly detection
        if "detect_anomaly" not in self.config["playable_model"]:
            self.config["playable_model"]["detect_anomaly"] = False

        # By default does not train adversarially
        if "gan_loss_lambda" not in self.config["playable_model_training"]["loss_weights"]:
            self.config["playable_model_training"]["loss_weights"]["gan_loss_lambda"] = 0.0

        # By default the gan lambda for the discriminator is the same as for the generator
        if "discriminator_gan_loss_lambda" not in self.config["playable_model_training"]["loss_weights"]:
            self.config["playable_model_training"]["loss_weights"]["discriminator_gan_loss_lambda"] = self.config["playable_model_training"]["loss_weights"]["gan_loss_lambda"]

        # By default do not care about camera in acmv
        if "use_camera_relative_acmv" not in self.config["playable_model_training"]:
            self.config["playable_model_training"]["use_camera_relative_acmv"] = False

        # By default no rotation axis is needed for acmv computation
        if "acmv_rotation_axis" not in self.config["playable_model_training"]:
            self.config["playable_model_training"]["acmv_rotation_axis"] = None

        # By default use the detector saved under the checkpoints path
        if "minecraft_detector_weights_filename" not in self.config["evaluation"]:
            self.config["evaluation"]["minecraft_detector_weights_filename"] = "checkpoints/detection_model_minecraft/latest.pth.tar"

        return True

    def create_directory_structure(self):
        '''
        Creates directories as required by the configuration file
        :return:
        '''

        Path(self.config["logging"]["output_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["checkpoints_root_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["playable_model_checkpoints_directory"]).mkdir(parents=True, exist_ok=True)

        Path(self.config["logging"]["output_images_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["reconstructed_dataset_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["reconstructed_camera_manipulation_dataset_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["reconstructed_playability_dataset_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["reconstructed_playability_legacy_dataset_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["evaluation_images_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["camera_trajectory_amt_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["camera_trajectory_dataset_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["style_storage_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["teaser_images_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["style_images_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["playability_qualitatives_directory"]).mkdir(parents=True, exist_ok=True)
        Path(self.config["logging"]["camera_motion_grid_directory"]).mkdir(parents=True, exist_ok=True)








