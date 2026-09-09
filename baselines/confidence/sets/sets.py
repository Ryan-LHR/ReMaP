"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "SETS: A Simple yet Effective DNN Test Selection Approach"

Note:
    The core SETS functions are adapted from the official implementation:
    https://github.com/GIST-NJU/SETS/blob/main/Source_code/SETS.py
    The wrapper integrates them with this project's feature, prediction, and
    per-budget selection pipeline.
--------------------------------------------------------------------------------------
"""

import re
import time

import numpy as np
from tqdm import tqdm

from utils.dufp.dufp_utils import convert_vectors_to_probs, get_target_layer_name
from utils.model_utils import extract_layer_output, flatten_layer_output
from utils.other_utils import load_features, save_point_print


def prioritize_by_sets(model, model_name, train_loader, cand_loader,
                       cand_vectors, budget_list, args):
    """Selection by the official SETS implementation."""

    'Step 1: Output Probabilities Extraction'
    softmax_cand_prob = convert_vectors_to_probs(cand_vectors)

    'Step 2: Feature Extraction'
    layer_name = get_target_layer_name(model_name)

    if not args.load_feature:
        cand_support_output = extract_layer_output(model, cand_loader, layer_name)
        cand_support_output = flatten_layer_output(cand_support_output)

    else:
        cand_kwargs = {'model': model, 'loader': cand_loader}
        cand_support_output = load_features(args, layer_name, label=f'Candidate_{args.cand_type}', **cand_kwargs)

    'Step 3: SETS Selection Under Different Budget'
    uncertainty, diversity, alpha = _parse_sets_params(args.method)
    index = list(range(len(softmax_cand_prob)))
    prioritized_dict = {}

    for budget in budget_list:
        budget_size = _get_budget_size(budget, len(softmax_cand_prob))
        save_point_print(
            f'Selecting under budget size={budget_size}, '
            f'uncertainty={uncertainty}, diversity={diversity}, alpha={alpha}'
        )

        selected_indices, _ = sets(
            budget_size,
            index,
            cand_support_output,
            softmax_cand_prob,
            uncertainty,
            diversity,
            alpha,
        )
        prioritized_dict[budget] = selected_indices

    return prioritized_dict


def _parse_sets_params(method):
    uncertainty = _get_param(method, 'uncertainty', 'maxp', str).lower()
    diversity = _get_param(method, 'diversity', 'gd', str).lower()
    alpha = _get_param(method, 'alpha', 3, float)

    if uncertainty in ['deepgini']:
        uncertainty = 'gini'

    if uncertainty not in ['maxp', 'gini']:
        raise ValueError(f"Unsupported SETS uncertainty metric: {uncertainty}")
    if diversity not in ['gd', 'std']:
        raise ValueError(f"Unsupported SETS diversity metric: {diversity}")
    if alpha < 1:
        raise ValueError("SETS alpha should be at least 1.")

    return uncertainty, diversity, alpha


def _get_param(method, key, default, value_type):
    pattern = rf"{key}\(([^)]+)\)"
    match = re.search(pattern, method)
    if match:
        return value_type(match.group(1))
    return default


def _get_budget_size(budget, data_size):
    if isinstance(budget, float) and 0 < budget <= 1:
        budget_size = int(data_size * budget)
    elif isinstance(budget, int) and budget >= 1:
        budget_size = budget
    else:
        raise ValueError(f"Unsupported budget: {budget}")

    budget_size = max(1, min(budget_size, data_size))
    return budget_size


def GD(IDs, features):
    selected_features = features[list(IDs)]
    dot_p = np.dot(selected_features, selected_features.T)
    sign, Log_det = np.linalg.slogdet(dot_p)
    return Log_det


def STD(IDs, features):
    x_sample = features[list(IDs)]
    std_f = np.std(x_sample, axis=0)
    L1norm = np.linalg.norm(std_f, 1)
    return L1norm


def gini_score(Output_probability):
    gini_scores = []
    for i in range(len(Output_probability)):
        sum_value = 0
        for j in range(len(Output_probability[0])):
            sum_value = sum_value + Output_probability[i][j] ** 2
        gini_scores.append(1 - sum_value)
    return gini_scores


def maxp_score(Output_probability):
    return [1 - max(prob) for prob in Output_probability]


def sets(size, index, features, output_probability, uncertainty, diversity, a):
    """
    Official SETS core adapted to accept project-provided feature/probability arrays.
    """
    start_time = time.time()

    features = _as_2d_numpy(features)
    output_probability = np.asarray(output_probability, dtype=np.float64)

    if uncertainty == "gini":
        un_scores = gini_score(output_probability)
    elif uncertainty == "maxp":
        un_scores = maxp_score(output_probability)
    else:
        raise ValueError(f"Unsupported SETS uncertainty metric: {uncertainty}")

    sorted_indices = sorted(index, key=lambda i: un_scores[i], reverse=True)

    top_percent_count = max(1, int(a * size))
    if a * size > len(index):
        top_percent_count = len(index)

    filtered_indices = sorted_indices[:top_percent_count]
    chunks = [filtered_indices[i::size] for i in range(size)]

    S = []
    current_gd = 0

    for chunk in tqdm(chunks, desc="SETS Selection"):
        max_gd_delta = -float('inf')
        best_index = -1
        gd_deltas = []
        gd_datas = []

        if len(chunk) == 0:
            continue

        for i in chunk:
            if diversity == "gd":
                new_gd = GD(S + [i], features)
            elif diversity == "std":
                new_gd = STD(S + [i], features)
            else:
                raise ValueError(f"Unsupported SETS diversity metric: {diversity}")

            gd_datas.append(new_gd)
            gd_delta = new_gd - current_gd
            gd_deltas.append(gd_delta)

        min_gd = min(gd_deltas)
        max_gd = max(gd_deltas)
        if max_gd - min_gd > 0:
            normalized_gd_deltas = [(gd - min_gd) / (max_gd - min_gd + 0.5) for gd in gd_deltas]
        else:
            normalized_gd_deltas = [0] * len(gd_deltas)

        for idx, i in enumerate(chunk):
            current_un = un_scores[i]
            objective_value = current_un * normalized_gd_deltas[idx]

            if objective_value > max_gd_delta:
                max_gd_delta = objective_value
                best_index = i

        if best_index != -1:
            S.append(best_index)
            ind = chunk.index(best_index)
            current_gd = gd_datas[ind]

    end_time = time.time()
    execution_time = end_time - start_time

    return S, execution_time


def _as_2d_numpy(features):
    if hasattr(features, 'detach'):
        features = features.detach().cpu().numpy()

    features = np.asarray(features, dtype=np.float64)
    features = np.nan_to_num(features, copy=False)

    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    if features.ndim != 2:
        raise ValueError(f"SETS features should be 2-D after flattening, got shape {features.shape}.")

    return features
