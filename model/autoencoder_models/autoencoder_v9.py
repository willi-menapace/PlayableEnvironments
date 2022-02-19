from typing import Dict

from model.autoencoder_models.autoencoder_v7 import AutoencoderV7
from model.autoencoder_models.decoder_v7 import DecoderV7
from model.autoencoder_models.encoder_v5 import EncoderV5


class AutoencoderV9(AutoencoderV7):
    '''
    An autoencoder v7 with bottleneck blocks at each downsampling step and no activation on skip connections
    Resnet blocks are inserted to break long sequences of ininterrupted upsampling layers
    '''

    def __init__(self, model_config: Dict):
        super(AutoencoderV9, self).__init__(model_config)

        self.encoder = EncoderV5(model_config)
        self.decoder = DecoderV7(model_config)


def model(model_config):
    '''
    Istantiates a decoder with the given parameters
    :param model_config:
    :return:
    '''
    return AutoencoderV9(model_config)


