"""
Task-conditioning of the borrowed reference space, applied before it is read.
"""
import numpy as np


def condition_features(features_fit, truths_fit, feature_list, beta=0.0, cov='within',
                       name='reference'):
    """
    Shrink the reference along its within-class directions, which carry no class signal.
    """
    if beta <= 0.0:
        return list(feature_list)

    X = np.asarray(features_fit, dtype=np.float64)
    if cov == 'total':
        C = np.cov(X - X.mean(axis=0), rowvar=False)
    else:
        y = np.asarray(truths_fit)
        centered = np.empty_like(X)
        for c in np.unique(y):  # centre each class on its own mean
            m = y == c
            centered[m] = X[m] - X[m].mean(axis=0)
        C = centered.T @ centered / max(1, len(X) - np.unique(y).size)

    d = C.shape[0]
    C = C * (d / (np.trace(C) + 1e-12))  # trace-normalize so beta is comparable across spaces
    C_beta = (1.0 - beta) * np.eye(d) + beta * C
    lam, V = np.linalg.eigh(C_beta)
    scale = np.power(np.clip(lam, 1e-8, None), -0.5)
    order = np.argsort(lam)[::-1]  # widest within-class directions first
    print(f"  [beta] {name} {cov} beta={beta}: top-5 directions scaled by "
          f"{np.array2string(scale[order[:5]], precision=3, floatmode='fixed')}, "
          f"tail-5 by {np.array2string(scale[order[-5:]], precision=3, floatmode='fixed')}"
          f"  (compressing the first, amplifying the last)")
    M = (V * scale) @ V.T
    return [np.asarray(f, dtype=np.float64) @ M for f in feature_list]


def _ridge_map(target, paired, ridge=1e-3):
    """Closed-form least squares from the DUT space into the reference space."""
    X = np.asarray(paired, dtype=np.float64)
    Y = np.asarray(target, dtype=np.float64)
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    G = Xc.T @ Xc
    G.flat[:: G.shape[0] + 1] += ridge * np.trace(G) / G.shape[0]
    return np.linalg.solve(G, Xc.T @ Yc)


def _inv_sqrt(C, floor=1e-8):
    lam, V = np.linalg.eigh(C)
    return (V * np.power(np.clip(lam, floor, None), -0.5)) @ V.T


def _sqrt(C, floor=0.0):
    lam, V = np.linalg.eigh(C)
    return (V * np.sqrt(np.clip(lam, floor, None))) @ V.T


def coral_map(target_fit, paired_fit, omega=0.5, ridge=1e-3):
    """
    Pull the reference's second-order structure toward the DUT's; omega=0 is the identity.
    """
    if omega <= 0.0:
        return None
    Y = np.asarray(target_fit, dtype=np.float64)
    X = np.asarray(paired_fit, dtype=np.float64)
    Yc, Xc = Y - Y.mean(axis=0), X - X.mean(axis=0)
    d = Yc.shape[1]
    W = _ridge_map(Y, X, ridge)  # lift the DUT covariance into the reference space
    S_lift = W.T @ (Xc.T @ Xc / len(X)) @ W
    S_lift = S_lift * (d / (np.trace(S_lift) + 1e-12))
    Stt = Yc.T @ Yc / len(Y)
    Stt.flat[:: d + 1] += ridge * np.trace(Stt) / d
    full = _inv_sqrt(Stt) @ _sqrt(S_lift + 1e-6 * np.eye(d))
    M = (1.0 - omega) * np.eye(d) + omega * full
    return M


def prepare_space(fit_ref, fit_truths, ref_list, paired_fit=None,
                  beta=0.0, cov='within', name='reference', omega=0.0):
    """
    Re-metricize the reference before it is read, everything fitted on fit_ref alone.
    """
    if omega > 0.0 and paired_fit is not None:  # coral first, so beta sees the realigned space
        M = coral_map(fit_ref, paired_fit, omega)
        if M is not None:
            ref_list = [np.asarray(f, dtype=np.float64) @ M for f in ref_list]
            fit_ref = ref_list[0]
    return condition_features(fit_ref, fit_truths, ref_list, beta, cov, name)
