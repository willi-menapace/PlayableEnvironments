from statistics import mean
from typing import Dict

import numpy as np
from scipy.stats import kurtosis


class ActionVariance:

    def __init__(self):
        pass

    def __call__(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int, object_idx=None) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :param object_idx: the object idx. None if inapplicable
        :return: results dictionary
        '''

        if actions.shape[0] == 0:
            return {}

        object_suffix = ""
        if object_idx is not None:
            object_suffix = f"_{object_idx}"

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        all_avg_variances = []
        results = {}
        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_vectors = vectors[actions == action_idx, :]
            current_vectors_count = current_vectors.shape[0]

            mean_vector = np.mean(current_vectors, axis=0)
            variance_vector = np.var(current_vectors, axis=0)
            kurtosis_vector = kurtosis(current_vectors, axis=0)
            avg_variance = np.mean(variance_vector)
            all_avg_variances.append(float(avg_variance))

            results[f"action_variance{object_suffix}/mean_vector/{action_idx}"] = mean_vector.tolist()
            results[f"action_variance{object_suffix}/kurtosis/{action_idx}"] = kurtosis_vector.tolist()
            results[f"action_variance{object_suffix}/quantiles/{action_idx}"] = np.quantile(current_vectors, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], axis=0).tolist()
            results[f"action_variance{object_suffix}/variance_vector/{action_idx}"] = variance_vector.tolist()
            results[f"action_variance{object_suffix}/avg_variance/{action_idx}"] = float(avg_variance)
            results[f"action_variance{object_suffix}/frequency/{action_idx}"] = float(current_vectors_count / vectors_count)

        results[f"action_variance{object_suffix}/avg_variance/mean"] = mean(all_avg_variances)

        global_mean_vector = np.mean(vectors, axis=0)
        #global_kurtosis_vector = kurtosis(vectors, axis=0)
        global_variance_vector = np.var(vectors, axis=0)
        global_avg_variance = np.mean(global_variance_vector)

        results[f"action_variance{object_suffix}/mean_vector/global"] = global_mean_vector.tolist()
        #results[f"action_variance/kurtosis/global"] = global_kurtosis_vector.tolist()
        results[f"action_variance{object_suffix}/quantiles/global"] = np.quantile(vectors, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], axis=0).tolist()
        results[f"action_variance{object_suffix}/variance_vector/global"] = global_variance_vector.tolist()
        results[f"action_variance{object_suffix}/avg_variance/global"] = float(global_avg_variance)
        results[f"action_variance{object_suffix}/delta_mse"] = results[f"action_variance{object_suffix}/avg_variance/mean"] / results[f"action_variance{object_suffix}/avg_variance/global"]

        return results

if __name__ == "__main__":

    actions = np.asarray([0, 1, 0])
    vectors = np.asarray([
        [3.3, 4.4],
        [1.0, 2.2],
        [3.2, 4.1],
    ])

    action_variance = ActionVariance()
    results = action_variance(actions, vectors, 2)

    print(results)