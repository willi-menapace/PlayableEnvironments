import os

import matplotlib.pyplot as plt
import torch


class AutoencoderFeaturesDrawer:

    @staticmethod
    def draw_features(features: torch.Tensor, output_dir: str) -> torch.Tensor:
        '''
        Draws bouding boxes on the given image tensors. Must be single camera

        :param features: (features_count, height, width) tensor with images
        :param output_dir: the directory where to save output

        '''

        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Minimum and maximum along height and width
        minimum = features.min(dim=2)[0].min(dim=1)[0].unsqueeze(-1).unsqueeze(-1)
        maximum = features.max(dim=2)[0].max(dim=1)[0].unsqueeze(-1).unsqueeze(-1)

        # normalizes the features in [0, 1]
        normalized_features = (features - minimum) / (maximum - minimum)

        features_count = normalized_features.size(0)
        for feature_idx in range(features_count):
            current_features = normalized_features[feature_idx]
            numpy_features = current_features.detach().cpu().numpy()

            # a colormap and a normalization instance
            cmap = plt.cm.jet
            # map the normalized data to colors
            # image is now RGBA (512x512x4)
            mapped_numpy_features = cmap(numpy_features)

            current_filename = os.path.join(output_dir, f"{feature_idx:05d}.png")

            # save the image
            plt.imsave(current_filename, mapped_numpy_features)
            plt.close()
