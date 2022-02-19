import numpy as np

from PIL import Image
from sklearn.manifold import TSNE


class TensorDisplayer:

    @staticmethod
    def show_image_tensor(tensor, range=[0.0, 1.0]):
        '''
        Displays a given torch tensor

        :param tensor: (3, height, width) tensor to display
        :param range: range of the values in the tensor for normalization
        :return:
        '''

        np_tensor = tensor.detach().cpu().numpy()
        np_tensor = (np_tensor + range[0]) / (range[1] - range[0])
        if np_tensor.shape[-1] != 3:
            np_tensor = np.moveaxis(np_tensor, 0, -1)
        np_tensor = (np_tensor * 255).astype(np.uint8)

        pil_tensor = Image.fromarray(np_tensor)
        pil_tensor.show()

    @staticmethod
    def reduce_dimensionality(features: np.ndarray):
        '''
        Reduces the dimensionality of the features to 2
        :param features: (..., dimensions)
        :return: (..., 2)
        '''

        dimensions = features.shape[1]
        if dimensions != 1:
            features = TSNE(n_jobs=14).fit_transform(features)
        else:
            zeros_copy = np.zeros_like(features)
            features = np.concatenate([features, zeros_copy], axis=1)

        return features


















