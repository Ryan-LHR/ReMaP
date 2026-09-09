import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer

from utils import extract_layer_output, flatten_layer_output, load_features
from utils.dufp.dufp_utils import get_target_layer_name, try_get_param


def prioritize_by_lof(
        model,
        model_name,
        train_loader,
        cand_loader,
        device,
        args,
):
    """
    LOF baseline using global penultimate-layer feature outlierness.
    """
    k = try_get_param(args.method, "k", int, 20)
    metric = try_get_param(args.method, "metric", str, "euclidean")

    layer_name = get_target_layer_name(model_name)

    if not args.load_feature:
        train_support_output = extract_layer_output(model, train_loader, layer_name, device)
        cand_support_output = extract_layer_output(model, cand_loader, layer_name, device)

        train_support_output = flatten_layer_output(train_support_output)
        cand_support_output = flatten_layer_output(cand_support_output)
    else:
        train_kwargs = {'model': model, 'loader': train_loader}
        cand_kwargs = {'model': model, 'loader': cand_loader}

        train_support_output = load_features(args, layer_name, label='Train', **train_kwargs)
        cand_support_output = load_features(args, layer_name, label=f'Candidate_{args.cand_type}', **cand_kwargs)

    normalizer = Normalizer(norm='l2')
    features_train = normalizer.transform(train_support_output)
    features_cand = normalizer.transform(cand_support_output)

    if len(features_train) <= 1:
        raise ValueError("LOF requires at least two training samples.")
    k = min(k, len(features_train) - 1)

    print(f"\nCurrent method is {args.method}, "
          f"\nLOF k is: {k}, metric is: {metric}, layer is: {layer_name}")

    lof = LocalOutlierFactor(
        n_neighbors=k,
        metric=metric,
        novelty=True,
    )
    lof.fit(features_train)

    # score_samples returns the opposite LOF: lower values mean more abnormal.
    lof_scores = -lof.score_samples(features_cand)
    rank_lst = np.argsort(lof_scores)
    rank_lst = rank_lst[::-1]

    return rank_lst
