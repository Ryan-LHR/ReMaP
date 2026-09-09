import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from baselines.coverage.coverage_utils import *

def sort_ctm(coverage, num_samples, seed):
    """
    Sort for CTM strategy
    Detail: Deterministic lexsort for non-zero coverage,
            Random shuffle for zero-coverage samples
    """

    coverage = coverage.detach().cpu()
    nonzero_mask = coverage > 0
    nonzero_idx = torch.nonzero(nonzero_mask, as_tuple=False).flatten().numpy()

    prioritized_indices = []

    # Deterministic lexsort for non-zero coverage (primary: -coverage, secondary: idx)
    if nonzero_idx.size > 0:
        cov_nonzero = coverage.numpy()[nonzero_idx]
        order = np.lexsort((nonzero_idx, -cov_nonzero))
        prioritized_nonzero = nonzero_idx[order].tolist()
        prioritized_indices.extend(prioritized_nonzero)

    # Random shuffle for zero-coverage samples
    remaining_samples = set(range(num_samples)) - set(prioritized_indices)
    prioritized_indices = extend_indices(prioritized_indices, remaining_samples, seed)

    return prioritized_indices


def extend_indices(prioritized_indices, remaining_samples, seed):
    """
    Eextend the prioritized_indices by remaining_samples
    Detail: When cam strategy stop because reaching highest coverage,
            random select remaining samples to prioritized indices
    """
    remaining_list = list(remaining_samples)  # convert a set to list
    random.seed(seed)
    random.shuffle(remaining_list)
    prioritized_indices.extend(remaining_list)

    return prioritized_indices

def prioritize_by_nac_ctm(total_neurons, act_values, t, seed):
    """
    NAC in CTM strategy

    Args:
        model (torch.nn.Module): neural network model
        total_neurons (int): number of neurons
        act_values (dict): neuron activation values
        t (float): threshold of NAC
    """
    # get number of samples
    num_samples = len(next(iter(next(iter(act_values.values())).values())))

    coverage = torch.zeros(num_samples)

    for s in tqdm(range(num_samples), desc="Prioritizing"):
        count = 0
        for layer_name in act_values:
            for neuron_id in act_values[layer_name]:
                val = act_values[layer_name][neuron_id][s]
                if val > t:
                    count += 1
        coverage[s] = count / total_neurons

    prioritized_indices = sort_ctm(coverage, num_samples, seed)
    return prioritized_indices

def prioritize_by_nac_cam(total_neurons, act_values, seed, t):
    """NAC in CAM strategy"""

    'Step 1: Initialize'
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    remaining_samples = set(range(num_samples))
    act_state = set()
    prioritized_indices = []

    'Step 2: Precompute coverage sets for each sample'
    # (sample_coverages contain the neurons each sample covered)
    sample_coverages = [set() for _ in range(num_samples)]
    for s in tqdm(range(num_samples)):
        for layer_name, neurons in act_values.items():
            for neuron_id in neurons:
                val = act_values[layer_name][neuron_id][s]
                if val > t:
                    sample_coverages[s].add((layer_name, neuron_id))

    'Step 3: Iteratively select samples that add the most new coverage'
    for _ in tqdm(range(num_samples), desc="Prioritizing"):
        best_sample = None
        best_new_coverage = -1

        for s in remaining_samples:
            sample = sample_coverages[s]
            new_coverage = len(sample - act_state) / total_neurons
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
    prioritized_indices = extend_indices(prioritized_indices, remaining_samples, seed)
    final_coverage = len(act_state) / total_neurons
    print(f'final NAC is {final_coverage}')

    return prioritized_indices

def prioritize_by_nbc_ctm(total_neurons, act_values, act_ranges, seed):
    """NBC in CTM strategy"""
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    coverage = torch.zeros(num_samples)

    for s in tqdm(range(num_samples), desc="Prioritizing"):
        count = 0
        for layer_name in act_values:
            for neuron_id in act_values[layer_name]:
                val = act_values[layer_name][neuron_id][s]
                # get upper and lower bound of current neuron
                lower_bound, upper_bound = act_ranges[layer_name][neuron_id]
                if val > upper_bound:
                    count += 1
                elif val < lower_bound:
                    count += 1

        coverage[s] = count / (2 * total_neurons)

    prioritized_indices = sort_ctm(coverage, num_samples, seed)

    return prioritized_indices

def prioritize_by_nbc_cam(total_neurons, act_values, act_ranges, seed):
    """NBC in CAM strategy"""

    'Step 1: Initialize'
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    remaining_samples = set(range(num_samples))
    act_state = set()
    prioritized_indices = []

    'Step 2: Precompute coverage sets for each sample'
    sample_coverages = [set() for _ in range(num_samples)]
    for s in tqdm(range(num_samples)):
        for layer_name, neurons in act_values.items():
            for neuron_id in neurons:
                val = act_values[layer_name][neuron_id][s]
                lower_bound, upper_bound = act_ranges[layer_name][neuron_id]
                if val > upper_bound:
                    sample_coverages[s].add((layer_name, neuron_id, 'upper'))
                elif val < lower_bound:
                    sample_coverages[s].add((layer_name, neuron_id, 'lower'))

    'Step 3: Iteratively select samples that add the most new coverage'
    for _ in tqdm(range(num_samples), desc="Prioritizing"):
        best_sample = None
        best_new_coverage = -1

        for s in remaining_samples:
            sample = sample_coverages[s]
            new_coverage = len(sample - act_state) / (2 * total_neurons)
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
    prioritized_indices = extend_indices(prioritized_indices, remaining_samples, seed)
    final_coverage = len(act_state) / (2 * total_neurons)
    print(f'final NBC is {final_coverage}')

    return prioritized_indices

def prioritize_by_snac_ctm(total_neurons, act_values, act_ranges, seed):
    """SNAC in CTM strategy"""
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    coverage = torch.zeros(num_samples)

    for s in tqdm(range(num_samples), desc="Prioritizing"):
        count = 0
        for layer_name in act_values:
            for neuron_id in act_values[layer_name]:
                val = act_values[layer_name][neuron_id][s]
                # get upper bound of current neuron
                _, upper_bound = act_ranges[layer_name][neuron_id]
                if val > upper_bound:
                    count += 1

        coverage[s] = count / total_neurons

    prioritized_indices = sort_ctm(coverage, num_samples, seed)

    return prioritized_indices

def prioritize_by_snac_cam(total_neurons, act_values, act_ranges, seed):
    """SNAC in CAM strategy"""

    'Step 1: Initialize'
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    remaining_samples = set(range(num_samples))
    act_state = set()
    prioritized_indices = []

    'Step 2: Precompute coverage sets for each sample'
    sample_coverages = [set() for _ in range(num_samples)]
    for s in tqdm(range(num_samples)):
        for layer_name, neurons in act_values.items():
            for neuron_id in neurons:
                val = act_values[layer_name][neuron_id][s]
                lower_bound, upper_bound = act_ranges[layer_name][neuron_id]
                if val > upper_bound:
                    sample_coverages[s].add((layer_name, neuron_id, 'upper'))

    'Step 3: Iteratively select samples that add the most new coverage'
    for _ in tqdm(range(num_samples), desc="Prioritizing"):
        best_sample = None
        best_new_coverage = -1

        for s in remaining_samples:
            sample = sample_coverages[s]
            new_coverage = len(sample - act_state) / total_neurons
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
    prioritized_indices = extend_indices(prioritized_indices, remaining_samples, seed)
    final_coverage = len(act_state) / total_neurons
    print(f'final SNAC is {final_coverage}')

    return prioritized_indices

def prioritize_by_kmnc_ctm(total_neurons, act_values, act_k_ranges, k, seed):
    """KMNC in CTM strategy"""
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    coverage = torch.zeros(num_samples)

    for s in tqdm(range(num_samples), desc="Prioritizing"):
        count = 0
        for layer_name in act_values:
            for neuron_id in act_values[layer_name]:
                val = act_values[layer_name][neuron_id][s]
                intervals = act_k_ranges[layer_name][neuron_id]

                for i in range(k):
                    lower_bound, upper_bound = intervals[i], intervals[i + 1]
                    if lower_bound <= val < upper_bound:
                        count += 1
                        break  # No need to check further intervals for this neuron

        coverage[s] = count / (k * total_neurons)

    prioritized_indices = sort_ctm(coverage, num_samples, seed)

    return prioritized_indices

def prioritize_by_kmnc_cam(total_neurons, act_values, act_k_ranges, seed, k):
    """KMNC in CAM strategy"""

    'Step 1: Initialize'
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    remaining_samples = set(range(num_samples))
    act_state = set()
    prioritized_indices = []

    'Step 2: Precompute coverage sets for each sample'
    sample_coverages = [set() for _ in range(num_samples)]
    for s in tqdm(range(num_samples)):
        for layer_name, neurons in act_values.items():
            for neuron_id in neurons:
                val = act_values[layer_name][neuron_id][s]
                intervals = act_k_ranges[layer_name][neuron_id]

                for i in range(k):
                    lower_bound, upper_bound = intervals[i], intervals[i + 1]
                    if lower_bound <= val < upper_bound:
                        sample_coverages[s].add((layer_name, neuron_id, f'sec-{i + 1}'))
                        break

    'Step 3: Iteratively select samples that add the most new coverage'
    for _ in tqdm(range(num_samples), desc="Prioritizing"):
        best_sample = None
        best_new_coverage = -1

        # for s in remaining_samples:
        for s in sorted(remaining_samples):
            sample = sample_coverages[s]
            new_coverage = len(sample - act_state) / (k * total_neurons)
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
    prioritized_indices = extend_indices(prioritized_indices, remaining_samples, seed)
    final_coverage = len(act_state) / (k * total_neurons)
    print(f'final KMNC is {final_coverage}')

    return prioritized_indices

def prioritize_by_tknc_cam(total_neurons, act_values, seed, k):
    """TKNC in CAM strategy"""

    'Step 1: Initialize'
    num_samples = len(next(iter(next(iter(act_values.values())).values())))
    remaining_samples = set(range(num_samples))
    act_state = set()
    prioritized_indices = []

    'Step 2: Precompute coverage sets for each sample'
    sample_coverages = [set() for _ in range(num_samples)]
    for s in tqdm(range(num_samples)):
        for layer_name, neurons in act_values.items():

            # get the neuron activations of sample s in given layer
            neuron_activations = [(neuron_id, neurons[neuron_id][s]) for neuron_id in neurons]
            top_k_neurons = sorted(neuron_activations, key=lambda x: x[1], reverse=True)[:k]

            for neuron_id, val in top_k_neurons:
                sample_coverages[s].add((layer_name, neuron_id))

    'Step 3: Iteratively select samples that add the most new coverage'
    for _ in tqdm(range(num_samples), desc="Prioritizing"):
        best_sample = None
        best_new_coverage = -1

        for s in remaining_samples:
            sample = sample_coverages[s]
            new_coverage = len(sample - act_state) / total_neurons
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
    prioritized_indices = extend_indices(prioritized_indices, remaining_samples, seed)
    final_coverage = len(act_state) / total_neurons
    print(f'final tknc is {final_coverage}')

    return prioritized_indices


def prioritize_by_coverage(method, model, train_loader, cand_loader, seed, strategy=None):
    """Prioritization based on Coverage"""
    FINE_COVERAGE = ['nbc', 'kmnc', 'snac']
    BATCH_SIZE = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # To reset the batch size:
    # train_loader = DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)
    # cand_loader = DataLoader(cand_loader.dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_neurons = count_neurons(model, train_loader, device)

    act_values = get_activation_values(model, cand_loader, device)
    # Print the activation values of first input
    print(f"\nThe activation values of input {0}: {get_sample_activations(act_values, 0)}")

    if strategy == None:
        method, strategy = get_method_and_strategy(method)

    if strategy == 'ctm':
        'Coverage-Total Method'
        if method == 'nac':
            t = 0.5  # same as NSS
            prioritized_indices = prioritize_by_nac_ctm(total_neurons, act_values, t, seed)

        elif method in FINE_COVERAGE:
            act_ranges = get_activation_ranges(model, train_loader, device)
            if method == 'nbc':
                prioritized_indices = prioritize_by_nbc_ctm(total_neurons, act_values, act_ranges, seed)
            elif method == 'snac':
                prioritized_indices = prioritize_by_snac_ctm(total_neurons, act_values, act_ranges, seed)
            elif method == 'kmnc':
                k = 1000  # same as NSS
                act_k_ranges = get_k_ranges(act_ranges, k)
                prioritized_indices = prioritize_by_kmnc_ctm(total_neurons, act_values, act_k_ranges, k, seed)
        else:
            raise ValueError("Method Not Found")

    elif strategy == 'cam':
        'Coverage-Additional Method'
        if method == 'nac':
            t = 0.5
            prioritized_indices = prioritize_by_nac_cam(total_neurons, act_values, seed, t)

        elif method in FINE_COVERAGE:
            act_ranges = get_activation_ranges(model, train_loader, device)
            if method == 'nbc':
                prioritized_indices = prioritize_by_nbc_cam(total_neurons, act_values, act_ranges, seed)
            elif method == 'snac':
                prioritized_indices = prioritize_by_snac_cam(total_neurons, act_values, act_ranges, seed)
            elif method == 'kmnc':
                k = 1000
                act_k_ranges = get_k_ranges(act_ranges, k)
                prioritized_indices = prioritize_by_kmnc_cam(total_neurons, act_values, act_k_ranges, seed, k)

        elif method == 'tknc':
            k = 3  # same as DeepGauge
            k = 2  # same as DeepGauge & simple-tip
            # k = 1
            prioritized_indices = prioritize_by_tknc_cam(total_neurons, act_values, seed, k)

        else:
            raise ValueError("Method Not Found")

    else:
        raise ValueError("Strategy Not Found (choice: cam or ctm)")

    return prioritized_indices

