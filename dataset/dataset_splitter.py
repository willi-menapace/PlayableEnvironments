import os
from typing import Dict


class DatasetSplitter:
    '''
    Helper class for dataset splitting
    '''

    @staticmethod
    def generate_splits(config) -> Dict:
        '''
        Computes the subsets of directory to include in the train, validation and test splits

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        base_path = config["data"]["data_root"]
        return {
            "train": (os.path.join(base_path, "train"), config["training"]["batching"]),
            "validation": (os.path.join(base_path, "val"), config["evaluation"]["batching"]),
            "test": (os.path.join(base_path, "test"), config["evaluation"]["batching"])
        }

    @staticmethod
    def generate_dataset_reconstruction_splits(config) -> Dict:
        '''
        Generates the datasets to use for the dataset reconstruction task

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        base_path = config["data"]["data_root"]
        return {
            # Train dataset is not reconstructed
            "validation": (os.path.join(base_path, "val"), config["evaluation"]["reconstructed_dataset_batching"]),
            "test": (os.path.join(base_path, "test"), config["evaluation"]["reconstructed_dataset_batching"])
        }

    @staticmethod
    def generate_camera_manipulation_dataset_reconstruction_splits(config) -> Dict:
        '''
        Generates the datasets to use for the dataset reconstruction task

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        base_path = config["evaluation"]["reconstructed_camera_manipulation_dataset_path"]
        return {
            # Train dataset is not reconstructed
            # Validation dataset is not reconstructed
            "test": (os.path.join(base_path, "test"), config["evaluation"]["reconstructed_camera_manipulation_dataset_batching"])
        }

    @staticmethod
    def generate_playability_dataset_reconstruction_splits(config) -> Dict:
        '''
        Generates the datasets to use for the dataset reconstruction task

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        base_path = config["data"]["data_root"]
        return {
            # Train dataset is not reconstructed
            "validation": (os.path.join(base_path, "val"), config["playable_model_evaluation"]["reconstructed_dataset_batching"]),
            "test": (os.path.join(base_path, "test"), config["playable_model_evaluation"]["reconstructed_dataset_batching"])
        }

    @staticmethod
    def generate_evaluate_reconstructed_dataset_splits(config, selector: str) -> Dict:
        '''
        Generates the datasets to use for when evaluating the dataset reconstruction

        :param config: the configuration file
        :param selector: 'reference' if the reference dataset is desired, 'generated' if the generated dataset is desired
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        if selector == "reference":
            base_path = config["data"]["data_root"]
        elif selector == "generated":
            base_path = config["logging"]["reconstructed_dataset_directory"]
        else:
            raise Exception(f"Unknown dataset selector {selector}")
        return {
            # Train dataset is not reconstructed
            #"validation": (os.path.join(base_path, "val"), config["evaluation"]["reconstructed_dataset_evaluation_batching"]),
            "test": (os.path.join(base_path, "test"), config["evaluation"]["reconstructed_dataset_evaluation_batching"])
        }

    @staticmethod
    def generate_evaluate_reconstructed_camera_manipulation_dataset_splits(config, selector: str) -> Dict:
        '''
        Generates the datasets to use for when evaluating the camera manipulation dataset reconstruction

        :param config: the configuration file
        :param selector: 'reference' if the reference dataset is desired, 'generated' if the generated dataset is desired
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        if selector == "reference":
            base_path = config["evaluation"]["reconstructed_camera_manipulation_dataset_path"]
        elif selector == "generated":
            base_path = config["logging"]["reconstructed_camera_manipulation_dataset_directory"]
        else:
            raise Exception(f"Unknown dataset selector {selector}")
        return {
            # Train dataset is not reconstructed
            # Validation dataset is not reconstructed
            "test": (os.path.join(base_path, "test"), config["evaluation"]["reconstructed_dataset_evaluation_batching"])
        }

    @staticmethod
    def generate_evaluate_reconstructed_playability_dataset_splits(config, selector: str) -> Dict:
        '''
        Generates the datasets to use for when evaluating the dataset reconstruction

        :param config: the configuration file
        :param selector: 'reference' if the reference dataset is desired, 'generated' if the generated dataset is desired
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        if selector == "reference":
            base_path = config["data"]["data_root"]
        elif selector == "generated":
            base_path = config["logging"]["reconstructed_playability_dataset_directory"]
        else:
            raise Exception(f"Unknown dataset selector {selector}")
        return {
            # Train dataset is not reconstructed
            #"validation": (os.path.join(base_path, "val"), config["evaluation"]["reconstructed_dataset_evaluation_batching"]),
            "test": (os.path.join(base_path, "test"), config["evaluation"]["reconstructed_dataset_evaluation_batching"])
        }

    @staticmethod
    def generate_evaluate_camera_trajectory_dataset_splits(config, selector: str) -> Dict:
        '''
        Generates the datasets to use for when evaluating the camera trajectorydataset reconstruction

        :param config: the configuration file
        :param selector: 'reference' if the reference dataset is desired, 'generated' if the generated dataset is desired
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        if selector == "reference":
            base_path = config["data"]["data_root"]
        elif selector == "generated":
            base_path = config["logging"]["camera_trajectory_dataset_directory"]
        else:
            raise Exception(f"Unknown dataset selector {selector}")
        return {
            # Train dataset is not reconstructed
            #"validation": (os.path.join(base_path, "val"), config["evaluation"]["reconstructed_dataset_evaluation_batching"]),
            "test": (os.path.join(base_path, "test"), config["evaluation"]["reconstructed_dataset_evaluation_batching"])
        }

    @staticmethod
    def generate_playable_model_splits(config) -> Dict:
        '''
        Computes the subsets of directory to include in the train, validation and test splits

        :param config: the configuration file
        :return: dictionary with a list of directories for each split
        '''

        dataset_style = config["data"]["dataset_style"]
        if dataset_style != "splitted":
            raise Exception("Only the 'splitted' dataset style is supported")

        base_path = config["data"]["data_root"]
        return {
            "train": (os.path.join(base_path, "train"), config["playable_model_training"]["batching"]),
            "validation": (os.path.join(base_path, "val"), config["playable_model_evaluation"]["batching"]),
            "test": (os.path.join(base_path, "test"), config["playable_model_evaluation"]["batching"])
        }


