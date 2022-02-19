import yaml
import os
from pathlib import Path

from utils.dict_wrapper import DictWrapper


class AutoencoderConfiguration:
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
        self.config["logging"]["evaluation_images_directory"] = os.path.join(self.config["logging"]["output_directory"], "evaluation_images")

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

        if not "max_steps_per_epoch" in self.config["training"]:
            self.config["training"]["max_steps_per_epoch"] = 10000

        if not "perceptual_features" in self.config["training"]:
            self.config["training"]["perceptual_features"] = 5

        # By default do not use L2 regularization
        if not "encoded_observations_squared_l2_norm_loss_lambda" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["encoded_observations_squared_l2_norm_loss_lambda"] = 0.0

        # By default use KL loss lambda == 1
        if not "KL_loss_lambda" in self.config["training"]["loss_weights"]:
            self.config["training"]["loss_weights"]["KL_loss_lambda"] = 1.0

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
        Path(self.config["logging"]["evaluation_images_directory"]).mkdir(parents=True, exist_ok=True)



