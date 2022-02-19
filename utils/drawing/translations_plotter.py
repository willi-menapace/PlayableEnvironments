import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils.tensor_folder import TensorFolder


class TranslationsPlotter:

    @staticmethod
    def plot_translations(translations: torch.Tensor, reconstructed_translations: torch.Tensor, excluded_axis: int, output_directory: str, prefix="", xlim=(-18, 18), ylim=(-18, 18)):
        '''
        Computes statistics about the vectors associated with each action

        :param translations: (..., observations_count, 3) array with translations
        :param reconstructed_translations: (..., observations_count, 3) array with reconstructed translations
        :param excluded_axis: axis to exclude from plotting
        :param output_directory: the directory where to output the plots
        :param prefix: string with which to start the name of each file
        '''

        Path(output_directory).mkdir(exist_ok=True, parents=True)

        translations, _ = TensorFolder.flatten(translations, -2)
        reconstructed_translations, _ = TensorFolder.flatten(reconstructed_translations, -2)

        all_axis = set(range(3))
        selected_axes = all_axis - set([excluded_axis])
        selected_axes = list(sorted(selected_axes))

        # Excludes one of the axes for plotting in 2D
        translations = translations[..., selected_axes]
        reconstructed_translations = reconstructed_translations[..., selected_axes]

        elements_count = translations.size(0)
        for element_idx in range(elements_count):
            current_translations = translations[element_idx].detach().cpu().numpy()
            current_reconstructed_translations = reconstructed_translations[element_idx].detach().cpu().numpy()

            current_translations_x = current_translations[..., 0]
            current_translations_y = current_translations[..., 1]
            current_reconstructed_translations_x = current_reconstructed_translations[..., 0]
            current_reconstructed_translations_y = current_reconstructed_translations[..., 1]

            plt.plot(current_translations_x, current_translations_y, linestyle='-', marker='o', markersize=2, label=f"GT")
            plt.plot(current_reconstructed_translations_x, current_reconstructed_translations_y, linestyle='--', marker='o', markersize=2, label=f"Reconstructed")
            plt.legend()
            plt.xlim(xlim)
            plt.ylim(ylim)

            current_filename = os.path.join(output_directory, f"{prefix}translations_{element_idx}.pdf")
            plt.savefig(current_filename, dpi=600)
            plt.close()


if __name__ == "__main__":
    translations = torch.zeros((3, 10, 3), dtype=torch.float)
    reconstructed_translations = torch.zeros((3, 10, 3), dtype=torch.float) + 1

    TranslationsPlotter.plot_translations(translations, reconstructed_translations, 2, "reconstructed_translations")