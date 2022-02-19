from typing import Tuple


class ObjectIDsHelper:
    '''
    Helper class for handling object ids and their correspondences with models
    '''

    def __init__(self, config):
        self.config = config

        # The number of models modeling static objects
        self.static_object_models_count = config["model"]["static_object_models"]
        # The number of models modeling the different object categories
        self.object_models_count = len(config["model"]["object_models"])
        # The number of models modeling dynamic objects
        self.dynamic_object_models_count = self.object_models_count - self.static_object_models_count

        self.object_parameters_encoders_configs = self.config["model"]["object_parameters_encoder"]
        self.object_models_configs = self.config["model"]["object_models"]
        self.object_encoders_configs = self.config["model"]["object_encoders"]

        # Computes the correspondences between object ids and models
        self.model_idx_by_object_idx_map = {}
        self.model_idx_by_dynamic_object_idx_map = {}
        self.first_object_idx_by_model_idx_map = {}

        current_object_idx = 0
        current_dynamic_object_idx = 0
        for model_idx in range(self.object_models_count):
            self.first_object_idx_by_model_idx_map[model_idx] = current_object_idx
            for _ in range(self.objects_count_by_model_idx(model_idx)):
                self.model_idx_by_object_idx_map[current_object_idx] = model_idx
                current_object_idx += 1

                if self.is_dynamic(model_idx):
                    self.model_idx_by_dynamic_object_idx_map[current_dynamic_object_idx] = model_idx
                    current_dynamic_object_idx += 1

        # Number of objects for each object category
        self.dynamic_objects_count = current_dynamic_object_idx
        self.objects_count = current_object_idx
        self.static_objects_count = self.objects_count - self.dynamic_objects_count

    def is_static(self, model_idx: int) -> bool:
        '''
        True if the id of the model corresponds to a static object

        :param model_idx: The id of the model
        :return: True if the id of the model corresponds to a static object
        '''

        return model_idx < self.static_object_models_count

    def is_dynamic(self, model_idx: int) -> bool:
        '''
        True if the id of the model corresponds to a dynamic object

        :param model_idx: The id of the model
        :return: True if the id of the model corresponds to a dynamic object
        '''

        return not self.is_static(model_idx)

    def objects_count_by_model_idx(self, model_idx: int) -> int:
        '''
        Computes the number of objects represented by the model with given idx
        :param model_idx: The id of the model
        :return:
        '''

        return self.object_parameters_encoders_configs[model_idx]["objects_count"]

    def objects_count_by_animation_model_idx(self, model_idx: int) -> int:
        '''
        Computes the number of objects represented by the animation model with given idx
        :param model_idx: The id of the model
        :return:
        '''

        # Animation model indexes start after the static models
        return self.object_parameters_encoders_configs[self.static_object_models_count + model_idx]["objects_count"]

    def model_idx_by_object_idx(self, object_idx) -> int:
        '''
        Computes the id of the model associated with the given object
        :param object_idx: The id of the object
        :return:
        '''

        return self.model_idx_by_object_idx_map[object_idx]

    def model_idx_by_dynamic_object_idx(self, object_idx) -> int:
        '''
        Computes the id of the model associated with the given dynamic object
        :param object_idx: The id of the object
        :return:
        '''

        return self.model_idx_by_dynamic_object_idx_map[object_idx]

    def animation_model_idx_by_dynamic_object_idx(self, object_idx) -> int:
        '''
        Computes the id of the animation model associated with the given dynamic object
        :param object_idx: The id of the object
        :return:
        '''

        return self.model_idx_by_dynamic_object_idx_map[object_idx] - self.model_idx_by_dynamic_object_idx_map[0]

    def object_idx_by_dynamic_object_idx(self, dynamic_object_idx) -> int:
        '''
        Computes the id of the object associated the given dynamic object
        :param object_idx: The id of the dynamic object
        :return:
        '''

        object_idx = dynamic_object_idx + self.static_objects_count
        if object_idx >= self.objects_count:
            raise Exception(f"The provided object id {dynamic_object_idx} is out of range")

        return object_idx

    def dynamic_object_idx_by_object_idx(self, object_idx) -> int:
        '''
        Computes the id of the dynamic object associated the given object
        :param object_idx: The id of the object
        :return:
        '''

        dynamic_object_id = object_idx - self.static_objects_count
        if dynamic_object_id < 0:
            raise Exception(f"The provided object id {object_idx} does not correspond to a dynamic object")

        return dynamic_object_id

    def dynamic_object_idx_range_by_model_idx(self, model_idx) -> Tuple[int, int]:
        '''
        Computes the range [begin_idx, end_idx) of ids for the dynamic objects associated with a certain model
        :param model_idx: The id of the model
        :return:
        '''

        if not self.is_dynamic(model_idx):
            raise Exception(f"Model id {model_idx} does not refer to a dynamic object")


        first_object_id = self.first_object_idx_by_model_idx_map[model_idx]
        first_dynamic_object_id = self.dynamic_object_idx_by_object_idx(first_object_id)
        objects_count = self.objects_count_by_model_idx(model_idx)
        end_idx = first_dynamic_object_id + objects_count

        return first_dynamic_object_id, end_idx
