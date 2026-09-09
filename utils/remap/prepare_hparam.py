"""
Training-set quantities and their caches: hparam is offline preprocessing and is
excluded from the reported cost, reliability is a step of the method and is counted.
"""
import json
from pathlib import Path

import numpy as np

from utils.remap.remap_utils import compute_global_reliability

HPARAM_SPLIT_SEED = 42  # held apart from args.seed


def cache_file(args, kind, teacher_name, stem):
    """
    Path of one cache entry, keyed by teacher and dataset like the teacher features.
    """
    save_dir = Path(args.data_dir) / "temp" / "remap" / kind / teacher_name / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{stem}.json"


def load_json(path, enabled=True):
    if not (enabled and path.exists()):
        return None
    with open(path) as f:
        return json.load(f)


def save_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def select_hparams_by_reliability(features_train, truths_train, num_classes, alpha,
                                  paired_train=None, cov='within', seed=HPARAM_SPLIT_SEED,
                                  beta_grid=(0.0, 0.25, 0.5, 0.75),
                                  omega_grid=(0.25, 0.5, 0.75),
                                  pick_beta=True, pick_omega=True):
    """
    Pick beta and omega by the reference's own held-out accuracy, using training labels only.
    """
    def r_at(b, w):
        return compute_global_reliability(features_train, truths_train, num_classes, alpha,
                                          seed=seed, beta=b, cov=cov,
                                          paired_train=paired_train, omega=w)

    omega = 0.0
    if pick_omega and paired_train is not None:  # omega first, the order it is applied in
        scores = [r_at(0.0, w) for w in omega_grid]
        omega = float(omega_grid[int(np.argmax(scores))])
        print(f"  [auto] r by omega: " + "  ".join(f"{w:.2f}:{s:.4f}"
                                                   for w, s in zip(omega_grid, scores))
              + f"  -> omega={omega}")
    beta = 0.0
    if pick_beta:
        scores = [r_at(b, omega) for b in beta_grid]  # on the space omega already realigned
        beta = float(beta_grid[int(np.argmax(scores))])
        print(f"  [auto] r by beta: " + "  ".join(f"{b:.2f}:{s:.4f}"
                                                 for b, s in zip(beta_grid, scores))
              + f"  -> beta={beta}")
    return beta, omega


def load_hparams(args, teacher_name, model_name, cov, alpha, beta_raw, omega_raw,
                 teacher_train, student_train, truths_train, num_classes):
    """
    Resolve beta and omega, selecting and caching them where either is given as 'auto'.
    """
    auto_beta, auto_omega = beta_raw == 'auto', omega_raw == 'auto'
    beta = 0.0 if auto_beta else float(beta_raw)
    omega = 0.0 if auto_omega else float(omega_raw)
    if not (auto_beta or auto_omega):
        return beta, omega

    # which knobs are auto is part of the key: picking beta alone leaves omega at 0
    picked = ('b' if auto_beta else '') + ('w' if auto_omega else '')
    path = cache_file(args, "hparam", teacher_name, f"{model_name}_{cov}_a{alpha}_auto-{picked}")
    hp = load_json(path, args.load_hparam)
    if hp is None:
        selected = select_hparams_by_reliability(
            teacher_train, truths_train, num_classes, alpha,
            paired_train=(student_train if auto_omega else None), cov=cov,
            pick_beta=auto_beta, pick_omega=auto_omega)
        hp = {"beta": selected[0], "omega": selected[1], "split_seed": HPARAM_SPLIT_SEED}
        save_json(path, hp)
        print(f"ReMaP hyperparameter is saved to: {path}")
    else:
        print(f"ReMaP hyperparameter is loaded from: {path}")
    if auto_beta:
        beta = float(hp["beta"])
    if auto_omega:
        omega = float(hp["omega"])
    print(f"  [auto] resolved beta={beta}, omega={omega}")
    return beta, omega


def load_reliability(args, teacher_name, model_name, cov, alpha, beta, omega, seed,
                     teacher_train, paired_train, truths_train, num_classes):
    """
    The reference's held-out accuracy, cached per experiment seed.
    """
    path = cache_file(args, "reliability", teacher_name,
                      f"{model_name}_{cov}_b{beta}_w{omega}_a{alpha}_seed{seed}")
    cached = load_json(path, args.load_reliability)  # pass -load_reliability False when timing
    if cached is not None:
        print(f"ReMaP reliability is loaded from: {path}")
        return float(cached["r"])
    r = compute_global_reliability(teacher_train, truths_train, num_classes, alpha,
                                   seed=seed, beta=beta, cov=cov,
                                   paired_train=paired_train, omega=omega)
    save_json(path, {"r": r})
    return r
