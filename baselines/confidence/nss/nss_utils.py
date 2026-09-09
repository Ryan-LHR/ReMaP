import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from tqdm import tqdm

from utils import load_loader
from baselines.confidence.nss.mutation import BenignMutation
from utils.data_utils import show_images
from utils.model_utils import extract_layer_output

def create_subset_dataset(cand_loader, ratio, seed=None):
    """
    Randomly selected a small ratio of original candidate set
    """

    original_dataset = cand_loader.dataset
    dataset_size = len(original_dataset)
    subset_size = int(dataset_size * ratio)

    # generate random indices
    random.seed(seed)
    selected_indices = random.sample(range(dataset_size), subset_size)

    # selected subset
    selected_set = Subset(original_dataset, selected_indices)

    return selected_set

def get_mutated_dataset(selected_set, mutation_types, params):
    """
    Iterate over each sample and randomly select a mutation type for mutation
    """
    mutated_inputs = []
    labels = []

    mutator = BenignMutation(mutation_types, params)

    for input, label in tqdm(selected_set, desc='Mutate inputs'):
        mutated_input = mutator(input)
        mutated_inputs.append(mutated_input)
        labels.append(label)

    mutated_inputs = torch.stack(mutated_inputs)
    labels_tensor = torch.tensor(labels)
    mutated_dataset = TensorDataset(mutated_inputs, labels_tensor)

    # show images for check mutation
    # show_images(selected_set, num_images=20)
    # show_images(mutated_dataset, num_images=20)

    return mutated_dataset


def identity_sensitive_neurons(model, ori_set, mutated_set, batch_size, layer_name, k, dataset_name):
    """Identity sensitive neurons"""

    'Step 1: Setup'
    if dataset_name in ['imagenet_100']:
        ori_loader = load_loader(dataset_name, ori_set, batch_size)
        mut_loader = load_loader(dataset_name, mutated_set, batch_size)
    else:
        ori_loader = DataLoader(dataset=ori_set, shuffle=False, batch_size=batch_size)
        mut_loader = DataLoader(dataset=mutated_set, shuffle=False, batch_size=batch_size)

    'Step 2: Capture output of target layer'
    ori_neuron_acts = extract_layer_output(model, ori_loader, layer_name)
    mut_neuron_acts = extract_layer_output(model, mut_loader, layer_name)

    # convert ndarray to tensor
    ori_neuron_acts = torch.from_numpy(ori_neuron_acts)
    mut_neuron_acts = torch.from_numpy(mut_neuron_acts)
    # flatten
    ori_neuron_acts = ori_neuron_acts.view(ori_neuron_acts.size(0), -1)
    mut_neuron_acts = mut_neuron_acts.view(mut_neuron_acts.size(0), -1)

    data_size, num_neurons = ori_neuron_acts.shape
    print(f"\nNum of total neurons is {num_neurons}")
    neuron_sensitivity = torch.zeros(num_neurons, device=ori_neuron_acts.device)

    'Step 3: Calculate neuron sensitivity'
    for i in range(data_size):
        sensitivity = torch.abs(mut_neuron_acts[i] - ori_neuron_acts[i])
        neuron_sensitivity += sensitivity

    'Step 4: Select sensitive neurons'
    num_selected = int(num_neurons * k)
    _, top_indices = torch.topk(neuron_sensitivity, num_selected)
    print(f'Num of Sensitive neurons is {num_selected}')

    return top_indices

def get_TNSScore(model, ori_set, mut_set, batch_size, layer_name, sensitive_neurons, dataset_name):
    """get TNSScore"""

    'Step 1: Setup'
    if dataset_name in ['imagenet_100']:
        ori_loader = load_loader(dataset_name, ori_set, batch_size)
        mut_loader = load_loader(dataset_name, mut_set, batch_size)
    else:
        ori_loader = DataLoader(dataset=ori_set, shuffle=False, batch_size=batch_size)
        mut_loader = DataLoader(dataset=mut_set, shuffle=False, batch_size=batch_size)

    'Step 2: Capture output of target layer'
    ori_neuron_acts = extract_layer_output(model, ori_loader, layer_name)
    mut_neuron_acts = extract_layer_output(model, mut_loader, layer_name)
    # convert ndarray to tensor
    ori_neuron_acts = torch.from_numpy(ori_neuron_acts)
    mut_neuron_acts = torch.from_numpy(mut_neuron_acts)
    # flatten
    ori_neuron_acts = ori_neuron_acts.view(ori_neuron_acts.size(0), -1)
    mut_neuron_acts = mut_neuron_acts.view(mut_neuron_acts.size(0), -1)

    'Step 3: Calculate TNSScore'
    num_samples = ori_neuron_acts.size(0)
    tnsscore = torch.zeros(num_samples, device=ori_neuron_acts.device)

    for i in range(num_samples):
        ori_sample = ori_neuron_acts[i, sensitive_neurons]
        mut_sample = mut_neuron_acts[i, sensitive_neurons]

        diff = torch.abs(mut_sample - ori_sample)
        tnsscore[i] = diff.sum()

    return tnsscore
