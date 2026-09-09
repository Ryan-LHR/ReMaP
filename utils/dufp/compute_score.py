import numpy as np
from sklearn.preprocessing import Normalizer

from utils.dufp.density_computer.ablation_density import *
from utils.dufp.density_computer.posterior import KnnPosteriorDistanceComputer


def compute_score(
        features_train,  # intermediate output of target layer (train set)
        features_test,  # intermediate output of target layer (test set)
        truths_train,  # ground truth (train set)
        truths_test,  # ground truth (test set)
        labels_test,  # predict labels (test set)
        probs_test,  # predict probability (test set)
        num_classes,
        method,
        dist_type='euclidean',
        test_type=None,
        dens_type=None,
        **kwargs,
):
    """
    Margin
    """

    'Step 1: Initialize'
    # get l2 normalized vectors
    normalizer = Normalizer(norm='l2')
    Z_train = normalizer.transform(features_train)
    Z_test = normalizer.transform(features_test)

    'Step 2: Setup'
    classes = np.unique(truths_train)
    num_classes = classes.size
    dim = Z_train.shape[1]
    eps = 1e-12

    lam = kwargs['lam']

    'Step 3: Density Estimation'
    density_kwargs = {
        "truths_train": truths_train,
        "alpha": kwargs['alpha'],
    }
    if dens_type == 'knn_nll_post':
        Computer = KnnPosteriorDistanceComputer

    elif 'kde' in dens_type:
        Computer = KdeDensityComputer

    elif 'gaussian_single' in dens_type:
        Computer = GaussianDensityComputer

    elif 'gmm' in dens_type:
        Computer = GmmDensityComputer

    elif 'vmf' in dens_type:
        Computer = VmfDensityComputer
    #
    # elif 'fixedradius' in dens_type:
    #     Computer = FixedRadiusDensityComputer
    # else:
    #     Computer = KnnKllDistanceComputer

    else:
        raise ValueError(f"Density type {dens_type} not found!")
    dist_type = dens_type

    dist_computer = Computer(
        Z_train=Z_train,
        num_classes=num_classes,
        dist_type=dist_type,
        **density_kwargs,
    )
    D = dist_computer.compute(Z_test)

    'Step 4: Uncertainty Score Computation'
    # get predicted-class bayes post NLL
    pred_row_idx = labels_test
    rows = np.arange(D.shape[0])
    d_pred = D[rows, pred_row_idx]

    # (1) compute ambiguity score
    # get strongest competitor-class NLL
    D_mask = D.copy()
    D_mask[rows, pred_row_idx] = np.inf
    other_row_idx = D_mask.argmin(axis=1)
    d_other = D[rows, other_row_idx]

    margin = d_pred - d_other

    # (2) compute atypicality score
    prior_adj = dist_computer.get_prior_adjustment()  # (C,)
    d_pred_dens = d_pred + prior_adj[pred_row_idx]  # D_dens = D_A + adjustment

    # (3) hybrid uncertainty
    u_amb = (margin - margin.min()) / (margin.max() - margin.min() + eps)  # ambiguity score
    u_aty = (d_pred_dens - d_pred_dens.min()) / (d_pred_dens.max() - d_pred_dens.min() + eps)  # atypicality score

    # hybrid uncertainty
    score = (1 - lam) * u_amb + lam * u_aty
    return score



