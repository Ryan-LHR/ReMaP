import numpy as np

from utils import extract_layer_output, flatten_layer_output, load_features
from utils.data_utils import get_modality
from utils.dufp.dufp_utils import get_target_layer_name, get_labels_and_classes, try_get_param
from utils.remap.adapt_geometry import prepare_space
from utils.remap.prepare_hparam import load_hparams, load_reliability
from utils.remap.remap_utils import *
from utils.remap.teacher_extractor import load_teacher_features

DEFAULT_TEACHER_BY_MODALITY = {
    'image': 'clip_vitl14',
    'pointcloud': 'uni3d',
    'audio': 'clap',
}


def prioritize_by_remap(
        model,
        model_name,
        train_loader,
        train_vectors,
        train_labels,
        cand_loader,
        cand_truths,
        cand_vectors,
        cand_labels,
        cand_correct,
        device,
        args,
):
    """
    ReMaP: Reference Model audited Prioritization.
    """

    '(1) Setup'
    y_train, y_cand = get_labels_and_classes(args, model_name, train_loader, cand_loader)
    num_classes = np.unique(y_train).size
    pred_labels_cand = np.array([t.cpu() for t in cand_labels])

    default_teacher = DEFAULT_TEACHER_BY_MODALITY[get_modality(args.dataset)]

    teacher_name = try_get_param(args.method, "teacher", str, default_teacher)
    fuse = try_get_param(args.method, "fuse", str, "rel")
    alpha = try_get_param(args.method, "alpha", float, 0.3)

    # Both default to 'auto': picked from the reference's own held-out accuracy and
    # cached (see prepare_hparam.py). Give a number to fix one, 0 to switch it off.
    beta_raw = try_get_param(args.method, "beta", str, "auto")
    cov = try_get_param(args.method, "cov", str, "within")
    omega_raw = try_get_param(args.method, "omega", str, "auto")

    print(f"\nReMaP {args.method}: teacher={teacher_name}, fuse={fuse}, alpha={alpha}, "
          f"beta={beta_raw}, cov={cov}, omega={omega_raw}")

    '(2) Get features of the DUT'
    layer_name = get_target_layer_name(model_name)
    if not args.load_feature:
        student_train = flatten_layer_output(extract_layer_output(model, train_loader, layer_name))
        student_cand = flatten_layer_output(extract_layer_output(model, cand_loader, layer_name))
    else:
        student_train = load_features(args, layer_name, label='Train', model=model, loader=train_loader)
        student_cand = load_features(args, layer_name, label=f'Candidate_{args.cand_type}', model=model, loader=cand_loader)

    '(3) Get features of the FM'
    cand_label = f'Candidate_{args.cand_type}'
    if args.cand_type == 'adversarial':
        cand_label = f'{cand_label}_{model_name}'
    teacher_train = load_teacher_features(args, 'Train', teacher_name, train_loader, device)
    teacher_cand = load_teacher_features(args, cand_label, teacher_name, cand_loader, device)

    '(4) Resolve auto hyperparameter, offline and cached'
    beta, omega = load_hparams(args, teacher_name, model_name, cov, alpha, beta_raw, omega_raw,
                               teacher_train, student_train, y_train, num_classes)

    '(4) Process the reference feature'
    teacher_train, teacher_cand = prepare_space(
        teacher_train, y_train, [teacher_train, teacher_cand],
        student_train, beta, cov, 'reference', omega)

    '(5) Per-space uncertainty and local certainty'
    sigma_s = compute_uncertainty(
        student_train, student_cand, y_train, pred_labels_cand, num_classes, alpha)
    sigma_t, local_ct = compute_uncertainty(
        teacher_train, teacher_cand, y_train, pred_labels_cand, num_classes, alpha,
        return_certainty=True)

    '(6) Trust-gated fusion'
    zs = zscore(sigma_s)
    zt = zscore(sigma_t)
    # Fusion
    if fuse == 'student_only':  # DUT's ambiguity alone
        score = -zs
    elif fuse == 'teacher_only':  # FM's ambiguity alone
        score = -zt

    elif fuse == 'rel':  # 'rel' (main / default)
        relpow = try_get_param(args.method, "relpow", float, 4.0)  # reliability exponent p
        trust = try_get_param(args.method, "trust", str, "global")  # tau combination form

        global_r = load_reliability(args, teacher_name, model_name, cov, alpha, beta, omega,
                                    int(getattr(args, 'seed', 0) or 0),
                                    teacher_train, student_train, y_train, num_classes)
        effe_trust = compute_effective_trust(global_r, local_ct, relpow, trust)
        print(f"  teacher reliability r={global_r:.4f}, g=r^{relpow}={global_r ** relpow:.4f}, trust={trust}")

        score = (1 - effe_trust) * (-zs) + effe_trust * (-zt)

    else:
        raise ValueError(f"Fuse type {fuse} not found!")

    '(7) Prioritization'  # higher score represents more fault-likely
    order = np.argsort(score)[::-1]
    return order
