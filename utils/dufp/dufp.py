import torch
import numpy as np

from utils import extract_layer_output, flatten_layer_output, load_features
from utils.dufp.compute_score import compute_score
from utils.dufp.dufp_utils import *


def prioritize_by_dufp(
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
    Implementation of our proposed DuFP
    """
    '(1) Setup'

    # get true labels
    y_train, y_cand = get_labels_and_classes(
        args,
        model_name,
        train_loader,
        cand_loader,
    )

    # get num of classes
    classes = np.unique(y_train)
    num_classes = classes.size

    # get data size
    train_size = len(y_train)
    cand_size = len(y_cand)

    # get target layer
    layer_name = get_target_layer_name(model_name)

    pred_labels_cand = np.array([t.cpu() for t in cand_labels])  # pred labels

    # get softmax outputs of candidate set
    prob_vectors_train = convert_vectors_to_probs(train_vectors)
    prob_vectors_cand = convert_vectors_to_probs(cand_vectors)

    # oc_wrong_indices_train, oc_correct_indices_train = \
    #     stats_overall(prob_vectors_train, y_train, 'Train')

    # oc_wrong_indices_cand, oc_correct_indices_cand = \
    #     stats_overall(prob_vectors_cand, y_cand, 'Cand')

    # return list(range(cand_size))  # for test

    '(2) Feature Extraction'
    if not args.load_feature:
        # extract
        train_support_output = extract_layer_output(model, train_loader, layer_name)
        cand_support_output = extract_layer_output(model, cand_loader, layer_name)
        # flatten layer outputs
        train_support_output = flatten_layer_output(train_support_output)
        cand_support_output = flatten_layer_output(cand_support_output)

    else:
        train_kwargs = {'model': model, 'loader': train_loader}
        cand_kwargs = {'model': model, 'loader': cand_loader}

        train_support_output = load_features(args, layer_name, label='Train', **train_kwargs)
        cand_support_output = load_features(args, layer_name, label=f'Candidate_{args.cand_type}', **cand_kwargs)

    print(f"The feature dimension of model {args.model} is {cand_support_output.shape[1]}")

    '(3) Uncertainty Score Computation'

    # 'Original implementation'
    # args.method = "dimp"

    # Set default param
    alpha = 0.3  # default value
    lam = 0.2 if 'IM100Test' not in model_name else 0.01  # default value
    dist_type = 'euclidean'  # default euclidean distance
    dens_type = 'knn_nll_post'

    test_type = None

    # Try to get param
    alpha = try_get_param(args.method, "alpha", float, alpha)
    lam = try_get_param(args.method, "lam", float, lam)
    dist_type = try_get_param(args.method, "dist", str, dist_type)

    test_type = try_get_param(args.method, "test", str, test_type, is_compound=True)
    dist_type = try_get_param(args.method, "dist", str, dist_type, is_compound=True)
    dens_type = try_get_param(args.method, "dens", str, dens_type, is_compound=True)
    dist_type = 'post'

    print(f"\nCurrent method is {args.method}, "
          f"\nalpha is: {alpha}, lam is {lam}"
          f"\ndistance type is: {dist_type}, ",
          f"\ndensity type is: {dens_type}, ",
          f"\ntest type is {test_type}")

    kwargs = {
        'model_name': model_name,
        'cand_type': args.cand_type,
        'alpha': alpha,
        'lam': lam,
    }

    score = compute_score(
        features_train=train_support_output,
        features_test=cand_support_output,
        truths_train=y_train,
        truths_test=y_cand,
        labels_test=pred_labels_cand,
        probs_test=prob_vectors_cand,

        num_classes=num_classes,
        method=args.method,
        dist_type=dist_type,
        dens_type=dens_type,
        test_type=test_type,
        **kwargs,
    )

    '(4) Prioritization'
    rank_lst = np.argsort(score)
    rank_lst = rank_lst[::-1]

    return rank_lst