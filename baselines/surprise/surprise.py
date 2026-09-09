# -*-coding:utf-8-*-
import random
from collections import defaultdict

# import numpy as np
# from tqdm import tqdm
# from scipy.stats import gaussian_kde
# import sklearn
from sklearn.covariance import EmpiricalCovariance
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader

from baselines.coverage.coverage_utils import *
from baselines.surprise.surprise_utils import *
from utils.dufp.dufp_utils import get_labels_and_classes


def get_clusters_of_per_class(train_ats, cand_ats, train_y, cand_y):
    """Get Clusters (for pc methods)"""

    # get clusters of train_ats
    train_ats_cluster = defaultdict(list)
    for at, label in zip(train_ats, train_y):
        train_ats_cluster[label].append(at)
    # convert to ndarray
    train_ats_cluster = {label: np.array(ats) for label, ats in train_ats_cluster.items()}

    # get clusters of cand indices
    cand_indices_cluster = defaultdict(list)
    for i, (at, label) in enumerate(zip(cand_ats, cand_y)):
        cand_indices_cluster[label].append(i)

    return train_ats_cluster, cand_indices_cluster

def get_ats(model, dataloder, total_neurons, device):
    """Get Activation Traces (ATs)"""
    act_values = get_activation_values(model, dataloder, device)
    num_samples = len(next(iter(next(iter(act_values.values())).values())))

    # Print the activation values of first input
    # print(f"\nThe activation values of input {0}: {get_sample_activations(act_values, 0)}")

    ats = np.zeros((num_samples, total_neurons), dtype=float)
    for s in tqdm(range(num_samples), desc="Capture ATs"):
        idx = 0
        for layer_name, neurons in act_values.items():
            for neuron_id in neurons:
                val = act_values[layer_name][neuron_id][s]
                ats[s, idx] = val
                idx += 1
    print('Done capture ATs\n')
    return ats

def get_layer_ats(model, dataloder, model_name, device):
    """Get ATs of target layer"""
    act_values = get_activation_values(model, dataloder, device)
    num_samples = len(next(iter(next(iter(act_values.values())).values())))

    # Select the second last layer
    if model_name == "MNIST-LeNet5":
        layer_name = 'fc2'
    elif model_name == "SVHN-VGG16":
        layer_name = 'fc2'
    elif model_name in ["FM-ResNet20", "C10-ResNet20"]:
        layer_name = 'avg_pool'
    elif model_name == "IM100Test-deit_base_patch16_224":
        layer_name = 'hf_model.classifier'
    elif model_name == "ModelNet40-DGCNN":
        layer_name = 'linear3'
    elif model_name == "ESC50-AST":
        layer_name = 'model.audio_spectrogram_transformer.encoder.layer.11.output.dense'
    else:
        raise ValueError(f"Model {model_name} not found")

    target_layer_neruons = len(act_values[layer_name])

    ats = np.zeros((num_samples, target_layer_neruons), dtype=float)
    for s in tqdm(range(num_samples), desc="Capture ATs of target layer"):
        idx = 0
        neurons = act_values[layer_name]
        for neuron_id in neurons:
            val = act_values[layer_name][neuron_id][s]
            ats[s, idx] = val
            idx += 1
    print('Done capture ATs\n')
    return ats

def get_lsa_surprise(train_ats, cand_ats, var_threshold, max_features=None):
    """
    LSA (take example of simple-tip)
    Args:
        var_threshold (float): The variance threshold of filtered neurons
        max_features (int or float): Optional Parameter, default is None in our paper (?)
    Return:
        surprise_values (ndarray): Surprise value of each sample, with total size of (num_sample,)
    """

    'Step 1: Calculate filtered neurons'
    # Figure out which neurons to remove (as they have low variance)
    if var_threshold is not None and var_threshold > 0:
        # calculate var by column ()
        at_var = np.var(train_ats, axis=0)
        removed_neurons = np.where(at_var < var_threshold)[0]

    if max_features is not None:
        if max_features < 1:  # as a ratio
            num_features = min(
                max_features * train_ats.shape[1], train_ats.shape[1]
            )
        else:
            num_features = min(max_features, train_ats.shape[1])

        dropped_columns = np.argsort(np.var(train_ats, axis=0))[:-num_features]
        removed_neurons = list(int(x) for x in dropped_columns)

    # remove filtered neurons (remove in _create_gaussian_kde)
    # if removed_neurons is not None and len(removed_neurons) > 0:
    #     cand_ats = np.delete(cand_ats, removed_neurons, axis=1)
        # train_ats = np.delete(train_ats, removed_neurons, axis=1)

    'Step 2: Construct kde'
    kde, removed_neurons = create_gaussian_kde(train_ats, removed_neurons)
    print(f"Done construct KDE")

    # if need, remove filtered neurons
    kde_size = kde.d
    if cand_ats.shape[1] != kde_size:
        if removed_neurons is not None and len(removed_neurons) > 0:
            cand_ats = np.delete(cand_ats, removed_neurons, axis=1)

    'Step 3: Get surprise values'
    if kde is None:
        surprise_values = np.zeros(shape=(cand_ats.shape[0],))
    else:
        density = kde.evaluate(cand_ats.transpose())
        surprise_values = -np.log(density)

    return surprise_values

def get_pc_lsa_surprise(train_ats, cand_ats, train_y, cand_y,
                        var_threshold, max_features=None):
    """PC-LSA"""

    'Step 1: Initialize (Clustering)'
    train_ats_cluster, cand_indices_cluster = \
        get_clusters_of_per_class(train_ats, cand_ats, train_y, cand_y)

    'Step 2: Calculate filtered neurons'
    # Figure out which neurons to remove (as they have low variance)
    removed_neurons = set()
    if var_threshold is not None and var_threshold > 0:
        for label, ats in train_ats_cluster.items():
            at_var = np.var(ats, axis=0)
            removed = np.where(at_var < var_threshold)[0]
            removed_neurons.update(removed)

    removed_neurons = sorted(removed_neurons)

    if max_features is not None:
        if max_features < 1:  # as a ratio
            num_features = min(
                max_features * train_ats.shape[1], train_ats.shape[1]
            )
        else:
            num_features = min(max_features, train_ats.shape[1])

        dropped_columns = np.argsort(np.var(train_ats, axis=0))[:-num_features]
        removed_neurons = list(int(x) for x in dropped_columns)

    # remove filtered neurons (remove in _create_gaussian_kde)
    # if removed_neurons is not None and len(removed_neurons) > 0:
    #     cand_ats = np.delete(cand_ats, removed_neurons, axis=1)
        # train_ats = np.delete(train_ats, removed_neurons, axis=1)

    'Step 3: Construct KDEs'
    removed_neurons_dict = {}
    kdes = {}
    for label, ats in train_ats_cluster.items():
        kde, remo_ns = create_gaussian_kde(ats, removed_neurons)
        kdes[label] = kde
        removed_neurons_dict[label] = remo_ns
    print(f"Done construct KDEs")

    'Step 4: Get surprise values'
    surprise_values = np.zeros(cand_ats.shape[0])
    for label, indices in cand_indices_cluster.items():
        kde = kdes[label]
        ats = cand_ats[indices]
        if kde is None:
            pc_surprise_values = np.zeros(shape=(ats.shape[0],))
        else:
            # if need, remove filtered neurons
            kde_size = kde.d
            if ats.shape[1] != kde_size:
                if removed_neurons is not None and len(removed_neurons) > 0:
                    ats = np.delete(ats, removed_neurons, axis=1)

            density = kde.evaluate(ats.transpose())
            pc_surprise_values = -np.log(density)

        surprise_values[indices] = pc_surprise_values

    return surprise_values

def get_pc_mlsa_surprise(train_ats, cand_ats, train_y, cand_y, num_components, seed):
    """PC-MLSA (take example of simple-tip)"""

    'Step 1: Initialize (Clustering)'
    train_ats_cluster, cand_indices_cluster = \
        get_clusters_of_per_class(train_ats, cand_ats, train_y, cand_y)

    'Step 2: Construct GMMs'
    gmms = {}
    for label, ats in train_ats_cluster.items():
        gmm = GaussianMixture(n_components=num_components, random_state=seed)
        gmm.fit(ats)
        gmms[label] = gmm
    print(f"Done construct GMMs")

    'Step 3: Get surprise values'
    surprise_values = np.zeros(cand_ats.shape[0])
    for label, indices in cand_indices_cluster.items():
        gmm = gmms[label]
        ats = cand_ats[indices]
        # Like LSA, MLSA is defined as negative log likelihood
        log_likelihood = gmm.score_samples(ats)
        pc_surprise_values = -log_likelihood

        surprise_values[indices] = pc_surprise_values

    return surprise_values

def get_dsa_surprise(train_ats, cand_ats, train_y, cand_y, subsampling, seed):
    """DSA, namely PC-DSA"""

    'Step 1: Initialize (Clustering)'
    train_ats_cluster, cand_indices_cluster = \
        get_clusters_of_per_class(train_ats, cand_ats, train_y, cand_y)

    'Step 2: Get distance values'
    surprise_values = np.zeros(cand_ats.shape[0])
    for label, indices in tqdm(cand_indices_cluster.items(), desc='Get distance'):
        # get ats of train (given class and other classes)
        train_ats_given_class = train_ats_cluster[label]
        other_labels = list(set(train_ats_cluster.keys()) - {label})
        train_ats_other_class = np.concatenate(
            [train_ats_cluster[l] for l in other_labels], axis=0
        )
        # subsampling
        train_ats_given_class = subsample_arrays(subsampling, train_ats_given_class, seed=seed)
        train_ats_other_class = subsample_arrays(subsampling, train_ats_other_class, seed=seed)

        # for each data point
        for i in indices:
            at = cand_ats[i]
            dist_a, dot_a = find_closest_at(at, train_ats_given_class)
            dist_b, _ = find_closest_at(dot_a, train_ats_other_class)
            distance = dist_a / dist_b

            surprise_values[indices] = distance

    return surprise_values

def get_pc_mdsa_surprise(train_ats, cand_ats, train_y, cand_y):
    """PC-MDSA (same as MDSA, take example of simple-tip)"""

    'Step 1: Initialize (Clustering)'
    train_ats_cluster, cand_indices_cluster = \
        get_clusters_of_per_class(train_ats, cand_ats, train_y, cand_y)

    'Step 2: Construct covariance matrixes'
    matrices = {}
    for label, ats in train_ats_cluster.items():
        matrix = EmpiricalCovariance()
        matrix.fit(ats)
        matrices[label] = matrix

    'Step 3: Get surprise values'
    surprise_values = np.zeros(cand_ats.shape[0])
    for label, indices in cand_indices_cluster.items():
        matrix = matrices[label]
        ats = cand_ats[indices]
        pc_distances = matrix.mahalanobis(ats)

        surprise_values[indices] = pc_distances

    return surprise_values

def get_pc_mmdsa_surprise(train_ats, cand_ats, train_y, cand_y, seed):
    """
    PC-MMDSA (take example of simple-tip)
    Args
        potential_k (list): num of k-means clusters
        subsampling (Union(int, float))
    """
    'Step 1: Initialize (Cluster and Set parameters)'
    random.seed(seed)
    np.random.seed(seed)
    train_ats_cluster, cand_indices_cluster = \
        get_clusters_of_per_class(train_ats, cand_ats, train_y, cand_y)

    potential_k = range(2, 6)
    subsampling = 0.3  # same as simple-tip
    # subsampling = 1.0  # not perform sub-sampling
    n_init: int = 10
    max_iter: int = 300

    'Step 2: Get K-Means Modals'
    # get discriminator & Construct covariance matrixes
    discriminators_list = {}
    matrices_list = {}
    for label, ats in tqdm(train_ats_cluster.items(), desc='Get K-Means Modals'):
        # (1) get discriminator
        discriminator = KmeansDiscriminator(
            training_data=ats,
            potential_k=potential_k,
            n_init=n_init,
            max_iter=max_iter,
            subsampling=subsampling,
            subsampling_seed=seed,
        )
        discriminators_list[label] = discriminator

        # (2) for each modal, construct matrix
        modal_indexes: np.ndarray = discriminator(ats)
        matrices = {}
        for modal_id in np.unique(modal_indexes):
            modal_ats = ats[modal_indexes == modal_id]
            matrix = EmpiricalCovariance()
            matrix.fit(modal_ats)
            matrices[modal_id] = matrix

        matrices_list[label] = matrices

    'Step 3: Get surprise values'
    surprise_values = np.zeros(cand_ats.shape[0])
    for label, indices in cand_indices_cluster.items():
        ats = cand_ats[indices]
        discriminator = discriminators_list[label]

        # get modal index in each cluster
        modal_indexes = discriminator(ats)

        indices = np.array(indices)
        for modal_id in np.unique(modal_indexes):
            modal_ats = ats[modal_indexes == modal_id]
            matrix = matrices_list[label][modal_id]
            pc_modal_distances = matrix.mahalanobis(modal_ats)

            # get global indices
            global_indices = indices[modal_indexes == modal_id]
            surprise_values[global_indices] = pc_modal_distances

    return surprise_values

def prioritize_by_surprise_cam(surprise_values, NUM_SC_BUCKETS, LOWER, UPPER, seed):
    """
    Surprise in CAM strategy

    Args:
        surprise_values (ndarray): Surprise value of each sample, with total size of (num_sample,)
        NUM_SC_BUCKETS (int): Buckets number of Surprise Coverage.
        upper (int or float): Upper bound of Surprise value.
        seed (int): random seed.
    """

    'Step 1: Initialize'
    num_samples = surprise_values.shape[0]
    remaining_samples = set(range(num_samples))
    act_state = set()
    prioritized_indices = []
    # get intervals
    if UPPER == 'Adaptive':
        # get max surprise as UPPER
        UPPER = np.max(surprise_values)
        print('Get Adaptive Surprise UPPER')
    interval_width = (UPPER - LOWER) / NUM_SC_BUCKETS
    intervals = [LOWER + i * interval_width for i in range(NUM_SC_BUCKETS + 1)]

    'Step 2: Precompute coverage sets for each sample'
    sample_coverages = [set() for _ in range(num_samples)]
    for s in tqdm(range(num_samples), desc="Precomputing coverage sets"):
        surprise = surprise_values[s]

        for i in range(NUM_SC_BUCKETS):
            lower_bound, upper_bound = intervals[i], intervals[i + 1]
            if lower_bound <= surprise < upper_bound:
                sample_coverages[s].add(f'sec-{i + 1}')
                break

    'Step 3: Iteratively select samples that add the most new coverage'
    for _ in tqdm(range(num_samples), desc="Prioritizing"):
        best_sample = None
        best_new_coverage = -1

        for s in remaining_samples:
            sample = sample_coverages[s]
            new_coverage = len(sample - act_state) / NUM_SC_BUCKETS
            if new_coverage > best_new_coverage:
                best_new_coverage = new_coverage
                best_sample = s
        if best_sample is None:
            raise ValueError("best sample not found")
        if best_new_coverage == 0:
            print(f'Reached the highest coverage at the {len(prioritized_indices)}-th sample')
            break

        prioritized_indices.append(best_sample)
        act_state.update(sample_coverages[best_sample])
        remaining_samples.remove(best_sample)

    'Step 4: Get final prioritization result'
    # extend remain samples by surprise value
    remaining_indices = list(remaining_samples)

    # remaining_indices_sorted = sorted(remaining_indices,
    #                                   key=lambda i: surprise_values[i], reverse=True)
    # seed = 42

    rng = np.random.default_rng(seed)
    remaining_indices_sorted = rng.permutation(remaining_indices).tolist()

    prioritized_indices.extend(remaining_indices_sorted)

    final_coverage = len(act_state) / NUM_SC_BUCKETS
    print(f'final Surprise Coverage is {final_coverage}')

    return prioritized_indices

def prioritize_by_surprise(method, model, model_name, train_loader, cand_loader,
                           cand_labels, seed, device, args):
    """Prioritization based on Surprise"""

    'Step 1: Initialize'
    NUM_SC_BUCKETS = 1000  # same as simple-tip, NSS, SA, DeepGini
    BATCH_SIZE = 256
    LOWER = 0  # same as simple-tip
    LAYER_LEVEL_SA = ['lsa', 'pc-lsa', 'pc-mlsa']  # layer-level sa methods (from sa)

    # To reset the batch size:
    # train_loader = DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    # cand_loader = DataLoader(cand_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_neurons = count_neurons(model, train_loader, device)

    if method not in LAYER_LEVEL_SA:
        train_ats = get_ats(model, train_loader, total_neurons, device)
        cand_ats = get_ats(model, cand_loader, total_neurons, device)
    else:
        UPPER = 2000  # same as NSS
        UPPER = 'Adaptive'  # same as simple-tip
        var_threshold = 1e-5  # same as LSA & CertPri
        # max_features = 300  # same as simple-tip (ignored in this study)
        train_ats = get_layer_ats(model, train_loader, model_name, device)
        cand_ats = get_layer_ats(model, cand_loader, model_name, device)

    if 'pc' in method:
        # train_y = train_loader.dataset.targets.tolist()  # true label
        # train_y = [label for _, label in train_loader.dataset]
        # cand_y = [label.item() for label in cand_labels]  # pred label
        # get true labels
        train_y, cand_y = get_labels_and_classes(
            args,
            model_name,
            train_loader,
            cand_loader,
        )

    'Step 2: Get surprise values'
    if method == 'lsa':
        surprise_values = get_lsa_surprise(train_ats, cand_ats,
                                           var_threshold, max_features=None)
    elif method == 'pc-lsa':
        surprise_values = get_pc_lsa_surprise(train_ats, cand_ats, train_y, cand_y,
                                              var_threshold, max_features=None)
    elif method == 'pc-mlsa':
        num_components = 3  # same as simple-tip
        surprise_values = get_pc_mlsa_surprise(train_ats, cand_ats,
                                               train_y, cand_y, num_components, seed)
    elif method == 'pc-dsa':
        UPPER = 2.0  # same as NSS
        UPPER = 'Adaptive'  # same as simple-tip
        subsampling = 0.3  # same as simple-tip
        # subsampling = 1.0
        surprise_values = get_dsa_surprise(train_ats, cand_ats,
                                           train_y, cand_y, subsampling, seed)
    elif method == 'pc-mdsa':
        UPPER = 'Adaptive'  # same as simple-tip
        surprise_values = get_pc_mdsa_surprise(train_ats, cand_ats,
                                               train_y, cand_y)
    elif method == 'pc-mmdsa':
        UPPER = 'Adaptive'  # same as simple-tip
        surprise_values = get_pc_mmdsa_surprise(train_ats, cand_ats,
                                                train_y, cand_y, seed)
    else:
        raise ValueError("Method not found")

    'Step 3: Prioritizing by surprise coverage'
    prioritized_indices = prioritize_by_surprise_cam(surprise_values, NUM_SC_BUCKETS,
                                                     LOWER, UPPER, seed)
    return prioritized_indices