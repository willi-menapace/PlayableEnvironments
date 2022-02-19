from typing import Dict

from model.autoencoder_models.autoencoder_v7 import AutoencoderV7
from model.autoencoder_models.decoder_v6 import DecoderV6
from model.autoencoder_models.encoder_v4 import EncoderV4


class AutoencoderV8(AutoencoderV7):
    '''
    An autoencoder v7 with bottleneck blocks at each downsampling step and no activation on skip connections
    '''

    def __init__(self, model_config: Dict):
        super(AutoencoderV8, self).__init__(model_config)

        self.encoder = EncoderV4(model_config)
        self.decoder = DecoderV6(model_config)


def model(model_config):
    '''
    Istantiates a decoder with the given parameters
    :param model_config:
    :return:
    '''
    return AutoencoderV8(model_config)


