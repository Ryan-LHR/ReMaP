# -*-coding:utf-8-*-
import re
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from baselines.surprise.stable_kde import StableGaussianKDE

def create_gaussian_kde(activations, removed_neurons):
    """construct kde (implementation of simple-tip)"""
    if activations.shape[1] == 0:  # remove all neurons
        warnings.warn(
            (
                # f"The passed min_var threshold {min_var_threshold} and/or"
                f"The passed min_var threshold {'min_var_threshold'} and/or"
                f"the automatic removal of numerically unstable features"
                f"led to the removal of all ATs. This instance of LSA will"
                f"thus always return density 0"
            ),
            UserWarning,
        )
        return None
    else:
        try:
            return StableGaussianKDE(activations.transpose()), removed_neurons

        except (np.linalg.LinAlgError, ValueError) as e:
            if "-th leading minor of the array is not positive definite" in str(
                    e
            ) or "numerical imprecision in covariance matrix" in str(e):
                problematic_row = int(re.findall("\\d*", str(e))[0]) - 1
                # remove neurons
                original_indexes = np.delete(
                    np.arange(activations.shape[1]), removed_neurons
                )
                problematic_index = original_indexes[problematic_row]

                warnings.warn(
                    f"Dropping AT {problematic_index}, as leading to numerical error.",
                    UserWarning,
                    1,
                )

                removed_neurons.append(problematic_index)
                return create_gaussian_kde(activations, removed_neurons)
            else:
                warnings.warn(f"Problem regarding KDE fitting", UserWarning)
                raise e

def subsample_arrays(
    subsampling: Union[int, float], arrays: Tuple[np.ndarray], seed: int
) -> Tuple[np.ndarray]:
    """Subsample multiple arrays using the sample sampling indexes for all.
    (take example of simple-tip)"""

    array_lengths = arrays.shape[0]
    assert all(
        a.shape[0] == arrays[0].shape[0] for a in arrays
    ), "All arrays must have the same number of samples"

    if subsampling == 1.0:
        return arrays
    elif isinstance(subsampling, int) and subsampling > 0:
        num_samples = min(subsampling, array_lengths)
    elif 0 < subsampling < 1:
        num_samples = int(subsampling * array_lengths)
    else:
        raise ValueError(
            "subsampling must be a float between 0 and 1"
            " (share of training data),"
            "or a positive int declaring the number of samples"
        )
    rng = np.random.RandomState(seed)
    indexes = rng.choice(np.arange(array_lengths), num_samples, replace=False)

    'original'
    # sub_arrays: List[np.ndarray] = [a[indexes] for a in arrays]
    # return tuple(sub_arrays)

    'Modified'
    sub_array = arrays[indexes]
    return sub_array


class KmeansDiscriminator:
    """(take example of simple-tip)"""
    def __init__(
        self,
        training_data,
        potential_k: Iterable[int],
        subsampling: Union[int, float] = 1.0,
        subsampling_seed: int = 0,
        n_init: int = 10,
        max_iter: int = 300,
    ):
        # training_data = _flatten_layers(training_data)
        training_data = subsample_arrays(
            subsampling, training_data, seed=subsampling_seed
        )

        self.best_score = -np.inf
        self.best_k = None
        self.best_clusterer = None

        for i in potential_k:
            kmeans = KMeans(n_clusters=i, n_init=n_init, max_iter=max_iter)
            cluster_labels = kmeans.fit_predict(training_data)
            silhouette_avg = silhouette_score(training_data, cluster_labels)
            if silhouette_avg > self.best_score:
                self.best_score = silhouette_avg
                self.best_k = i
                self.best_clusterer = kmeans

    def __call__(
        self, activations
    ) -> np.ndarray:
        return self.best_clusterer.predict(activations)


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.
    (take example of CertPri)

    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])
