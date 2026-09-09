import numpy as np
from sklearn.preprocessing import Normalizer

from utils.remap.adapt_geometry import prepare_space

from utils.dufp.density_computer.posterior import KnnPosteriorDistanceComputer


def compute_uncertainty(
        features_train,
        features_test,
        truths_train,
        labels_test,
        num_classes,
        alpha,
        return_certainty=False
):
    """
    Per-space uncertainty, and the local certainty that gates it.
    """

    normalizer = Normalizer(norm='l2')
    computer = KnnPosteriorDistanceComputer(
        Z_train=normalizer.transform(features_train), num_classes=num_classes,
        dist_type='knn_nll_post', truths_train=truths_train, alpha=alpha)
    D = computer.compute(normalizer.transform(features_test))

    rows = np.arange(D.shape[0])
    d_pred = D[rows, labels_test]
    D_mask = D.copy()
    D_mask[rows, labels_test] = np.inf
    competitor_idx = D_mask.argmin(axis=1)
    sigma = D[rows, competitor_idx] - d_pred
    if not return_certainty:
        return sigma

    posterior = np.exp(-computer.compute_nll_posterior(D))
    local_ct = (posterior.max(axis=1) - 1.0 / num_classes) / (1.0 - 1.0 / num_classes)
    return sigma, np.clip(local_ct, 0.0, 1.0)


def zscore(a, eps=1e-12):
    """
    Normalization
    """
    return (a - a.mean()) / (a.std() + eps)


def compute_global_reliability(features_train, truths_train, num_classes, alpha, holdout=0.2, seed=0,
                               beta=0.0, cov='within', name='reference',
                               paired_train=None, omega=0.0):
    """
    Global reliability.
    """
    y = np.asarray(truths_train)
    rng = np.random.RandomState(seed)

    idx = rng.permutation(len(y))
    n_eval = max(1, int(len(y) * holdout))
    eval_idx, fit_idx = idx[:n_eval], idx[n_eval:]

    # everything below is fitted on the fit split alone, or the transform sees the audit points
    pair_fit = None if paired_train is None else np.asarray(paired_train)[fit_idx]
    X_fit, X_eval = prepare_space(
        features_train[fit_idx], y[fit_idx],
        [features_train[fit_idx], features_train[eval_idx]],
        pair_fit, beta, cov, name, omega)
    normalizer = Normalizer(norm='l2')
    computer = KnnPosteriorDistanceComputer(
        Z_train=normalizer.transform(X_fit), num_classes=num_classes,
        dist_type='knn_nll_post', truths_train=y[fit_idx].tolist(), alpha=alpha)
    D = computer.compute(normalizer.transform(X_eval))

    global_r = np.mean(D.argmin(axis=1) == y[eval_idx])
    return float(global_r)


def compute_effective_trust(global_r, local_ct, relpow, form='mult'):
    """
    Effective trust.
    """
    if form == 'local':  # gate quantity only (drop global reliability r)
        return local_ct
    g = global_r ** relpow
    if form == 'mult':  # tau = g * kappa (default: logical conjunction)
        return g * local_ct
    if form == 'min':  # soft-AND
        return np.minimum(g, local_ct)
    if form == 'add':  # linear average (the disjunctive form we argue against)
        return 0.5 * (g + local_ct)
    if form == 'global':  # g only, no per-sample kappa (mirror of local, which is kappa only)
        return np.full_like(local_ct, g)
    raise ValueError(f"Trust form {form} not found!")
