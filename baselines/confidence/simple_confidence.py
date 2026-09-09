# -*-coding:utf-8-*-

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.datasets import ProbVecDataset


def prioritize_by_deepgini(prob_loader, device):
    """DeepGini"""
    metric_values = torch.tensor([], dtype=torch.float32)
    metric_values = metric_values.to(device)
    for batch_idx, prob_batch in enumerate(prob_loader):
        prob_batch = prob_batch.to(device)
        gini_values = 1 - torch.sum(prob_batch ** 2, dim=1)
        'print data for validate'
        # if batch_idx == 0:
        #     print(f"Original prediction vectors for the first batch: {prob_batch[:3]}"
        #           f"DeepGini values for the first batch: {gini_values[:3]}")
        metric_values = torch.cat((metric_values, gini_values), dim=0)

    # The higher deepgini value corresponding to higher uncertainty
    uncertainty = metric_values
    prioritized_indices = torch.argsort(uncertainty, descending=True)
    prioritized_indices = prioritized_indices.tolist()

    return prioritized_indices

def prioritize_by_maxp(prob_loader, device):
    """MaxP"""
    metric_values = torch.tensor([], dtype=torch.float32)
    metric_values = metric_values.to(device)
    for batch_idx, prob_batch in enumerate(prob_loader):
        prob_batch = prob_batch.to(device)
        max_values, _ = torch.max(prob_batch, dim=1)
        # if batch_idx == 0:
        #     print(f"Original prediction vectors for the first batch: {prob_batch[:3]}"
        #           f"MaxP values for the first batch: {max_values[:3]}")
        metric_values = torch.cat((metric_values, max_values), dim=0)

    # The higher maxp value corresponding to lower uncertainty
    uncertainty = -1 * metric_values
    prioritized_indices = torch.argsort(uncertainty, descending=True)
    prioritized_indices = prioritized_indices.tolist()

    return prioritized_indices

def prioritize_by_margin(prob_loader, device):
    """Margin"""
    metric_values = torch.tensor([], dtype=torch.float32)
    metric_values = metric_values.to(device)
    for batch_idx, prob_batch in enumerate(prob_loader):
        prob_batch = prob_batch.to(device)
        top2_values, _ = torch.topk(prob_batch, 2, dim=1)
        margin_values = top2_values[:, 0] - top2_values[:, 1]
        # margin_values = top2_values[:, 0] / top2_values[:, 1]
        # if batch_idx == 0:
        #     print(f"Original prediction vectors for the first batch: {prob_batch[:3]}"
        #           f"Margin values for the first batch: {margin_values[:3]}")
        metric_values = torch.cat((metric_values, margin_values), dim=0)

    # The higher margin value corresponding to lower uncertainty
    uncertainty = -1 * metric_values
    prioritized_indices = torch.argsort(uncertainty, descending=True)
    prioritized_indices = prioritized_indices.tolist()

    return prioritized_indices

def prioritize_by_entropy(prob_loader, device):
    """Entropy"""
    metric_values = torch.tensor([], dtype=torch.float32)
    metric_values = metric_values.to(device)
    for batch_idx, prob_batch in enumerate(prob_loader):
        prob_batch = prob_batch.to(device)
        entropy_values = -torch.sum(prob_batch * torch.log(prob_batch), dim=1)
        # if batch_idx == 0:
        #     print(f"Original prediction vectors for the first batch: {prob_batch[:3]}"
        #           f"Entropy values for the first batch: {entropy_values[:3]}")
        metric_values = torch.cat((metric_values, entropy_values), dim=0)

    # The higher entropy value corresponding to higher uncertainty
    uncertainty = metric_values
    prioritized_indices = torch.argsort(uncertainty, descending=True)
    prioritized_indices = prioritized_indices.tolist()

    return prioritized_indices

def prioritize_by_softmax(prob_loader):
    """Softmax"""

    return

def prioritize_by_confidence(method, cand_vectors, device):
    """Prioritization based on Confidence"""

    # validate whether cand_vectors is composed of probability vectors
    prob_loader = get_prob_loader(cand_vectors)

    if method == 'deepgini':
        prioritized_indices = prioritize_by_deepgini(prob_loader, device)
    elif method == 'maxp':
        prioritized_indices = prioritize_by_maxp(prob_loader, device)
    elif method == 'margin':
        prioritized_indices = prioritize_by_margin(prob_loader, device)
    elif method == 'entropy':
        prioritized_indices = prioritize_by_entropy(prob_loader, device)
    # elif method == 'softmax':
    #     prioritized_indices = prioritize_by_softmax(prob_loader, device)
    else:
        raise ValueError("Method not found")

    return prioritized_indices


def is_probability_vector(tensor):
    """validate whether tensor is a probability vector"""
    # Check if all values are between 0 and 1
    if torch.all((tensor >= 0) & (tensor <= 1)):
        # Check if the sum of all values is 1
        if torch.isclose(tensor.sum(), torch.tensor(1.0)):
            return True
    return False

def process_with_softmax(logit_vectors):
    """Convert each tensor to probabilities using softmax"""
    softmax = torch.nn.Softmax(dim=-1)
    prob_vectors = [softmax(logit) for logit in logit_vectors]

    return prob_vectors

def get_prob_loader(cand_vectors):
    """get dataloader of prediction probability vectors"""
    is_probability = True
    for i, vec in enumerate(cand_vectors[:5]):
        if is_probability_vector(vec):
            print(f"cand_vectors[{i}] is a valid probability vector.")
        else:
            print(f"cand_vectors[{i}] {vec} is NOT a valid probability vector with sum {vec.sum()}.")
            is_probability = False

    if not is_probability:
        prob_vectors = process_with_softmax(cand_vectors)
        print(f"cand_vectors[0] now is a valid probability vector: {prob_vectors[0]}.")
    else:
        prob_vectors = cand_vectors
    prob_loader = DataLoader(ProbVecDataset(prob_vectors), batch_size=1024, shuffle=False)

    return prob_loader

