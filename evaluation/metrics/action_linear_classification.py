import traceback
from typing import Dict

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsOneClassifier


class ActionClassificationScore:

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

        results = {}

        try:
            linear_results = self.compute_linear_results(actions, vectors, actions_count, object_suffix)
            results = dict(results, **linear_results)
        except Exception as e:
            print(f"Could not compute accuracy results due to error: {e}")
            print(traceback.format_exc())

        try:
            rbf_results = self.compute_rbf_results(actions, vectors, actions_count, object_suffix)
            results = dict(results, **rbf_results)
        except Exception as e:
            print(f"Could not compute accuracy results due to error: {e}")
            print(traceback.format_exc())

        try:
            poly_results = self.compute_poly_results(actions, vectors, actions_count, object_suffix)
            results = dict(results, **poly_results)
        except Exception as e:
            print(f"Could not compute accuracy results due to error: {e}")
            print(traceback.format_exc())

        try:
            linear_ovo_results = self.compute_ovo_results(actions, vectors, actions_count, object_suffix)
            results = dict(results, **linear_ovo_results)
        except Exception as e:
            print(f"Could not compute accuracy results due to error: {e}")
            print(traceback.format_exc())

        return results

    def compute_ovo_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int, object_suffix: str) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = OneVsOneClassifier(svm.LinearSVC(random_state=0, max_iter=10000))

        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results[f"linear_ovo{object_suffix}/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"linear_ovo{object_suffix}/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

    def compute_linear_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int, object_suffix: str) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = svm.LinearSVC(max_iter=10000)
        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results[f"linear{object_suffix}/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"linear{object_suffix}/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

    def compute_rbf_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int, object_suffix: str) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = svm.SVC(max_iter=10000)
        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results[f"rbf{object_suffix}/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"rbf{object_suffix}/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results

    def compute_poly_results(self, actions: np.ndarray, vectors: np.ndarray, actions_count: int, object_suffix: str) -> Dict:
        '''
        Computes statistics about the vectors associated with each action

        :param actions: (...) array with actions in [0, actions_count)
        :param vectors: (..., vector_size) array with vectors corresponding to actions
        :param actions_count: the number of actions
        :return: results dictionary
        '''

        vector_size = vectors.shape[-1]

        actions = np.reshape(actions, (-1))
        vectors = np.reshape(vectors, (-1, vector_size))

        vectors_count = vectors.shape[0]

        # Trains the svm on the movement vectors to predict the actions
        clf = svm.SVC(kernel="poly", max_iter=10000)
        clf.fit(vectors, actions)
        predicted_actions = clf.predict(vectors)

        results = {}
        results[f"poly{object_suffix}/action_accuracy"] = float(accuracy_score(actions, predicted_actions))

        for action_idx in range(actions_count):
            # If no actions of this category are present, skip it
            if (actions == action_idx).sum() == 0:
                continue

            current_actions = actions[actions == action_idx]
            current_predicted_actions = predicted_actions[actions == action_idx]
            results[f"poly{object_suffix}/action_accuracy/{action_idx}"] = float(accuracy_score(current_actions, current_predicted_actions))

        return results


if __name__ == "__main__":

    actions = np.random.randint(0, 4, size=(1000))
    vectors = np.random.random([1000, 3])

    results = ActionClassificationScore()(actions, vectors, 5)

    print(results)