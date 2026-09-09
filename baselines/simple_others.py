# -*-coding:utf-8-*-
import random

from utils import save_point_print


def prioritize_by_random(seed, candidate_set):
    """Random Prioritization"""
    random.seed(seed)
    num_samples = len(candidate_set)
    prioritized_indices = random.sample(range(num_samples), num_samples)

    return prioritized_indices

def prioritize_by_best(cand_correct, seed):
    """
    Best Selection
    Within each group (True or False), the order is randomly shuffled based on a given seed.
    (only consider correctness, not consider diversity)!
    """
    # Separate indices
    miscls_indices = [i for i, correct in enumerate(cand_correct) if not correct]
    correct_indices = [i for i, correct in enumerate(cand_correct) if correct]

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(miscls_indices)
    rng.shuffle(correct_indices)

    # Concat
    prioritized_indices = miscls_indices + correct_indices

    return prioritized_indices

def selection_by_best_ratio(cand_correct, budget_list, method_name, seed):
    """
    Best Selection
    Randomly Select the given ratio of misclassified samples
    """
    from utils.dufp.dufp_utils import try_get_param
    ratio = 0.25  # default
    ratio = try_get_param(method_name, "ratio", float, ratio)
    print(f"Current misclassified ratio is: {ratio}")

    # Separate indices
    miscls_indices = [i for i, correct in enumerate(cand_correct) if not correct]
    correct_indices = [i for i, correct in enumerate(cand_correct) if correct]

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(miscls_indices)
    rng.shuffle(correct_indices)

    data_size = len(cand_correct)
    prioritized_dict = {}
    for i, budget in enumerate(budget_list):
        budget_size = int(data_size * budget)
        wrong_size = int(budget_size * ratio)
        correct_size = budget_size - wrong_size
        save_point_print(f'Selecting under budget size={budget_size}')

        # Relax ratio if one group is insufficient, but keep budget_size unchanged
        wrong_take = min(wrong_size, len(miscls_indices))
        correct_take = min(correct_size, len(correct_indices))
        remaining = budget_size - (wrong_take + correct_take)

        if remaining > 0:
            extra_wrong = min(remaining, len(miscls_indices) - wrong_take)
            wrong_take += extra_wrong
            remaining -= extra_wrong

        if remaining > 0:
            extra_correct = min(remaining, len(correct_indices) - correct_take)
            correct_take += extra_correct
            remaining -= extra_correct

        if remaining > 0:
            print(f"[Warning]: Not enough samples to fill budget {budget}!")
            prioritized_dict[budget] = []
            continue

        # Concat
        prioritized_indices = miscls_indices[:wrong_take] + correct_indices[:correct_take]
        prioritized_dict[budget] = prioritized_indices

    return prioritized_dict


def prioritize_by_best_diverse(cand_truths, cand_labels, cand_correct, seed):
    """
    Best Selection
    (consider the naive diversity, namely fault_type)
    """

    pri_pred_labels = [cand_labels[i].item() for i in prioritized_indices]
    pri_true_labels = [cand_truths[i] for i in prioritized_indices]
    return prioritized_indices