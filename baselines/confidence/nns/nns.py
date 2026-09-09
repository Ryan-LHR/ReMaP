"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Our Own Implementation
Reference Paper     : "In Defense of Simple Techniques for Neural Network Test Case Selection"

Note:
    This implementation is developed based on the descriptions and algorithms (e.g., pseudocode)
    provided in the original paper. No official code was released by the authors.

--------------------------------------------------------------------------------------
"""

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from baselines.confidence.simple_confidence import get_prob_loader, ProbVecDataset, prioritize_by_deepgini


def construct_distance_mat(feature_loader, k):
    """get top k distance and indices from features"""

    num_samples = len(feature_loader.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize matrix
    topk_distances = torch.full((num_samples, k), float('inf'), dtype=torch.float16, device=device)
    topk_indices = torch.full((num_samples, k), -1, dtype=torch.long, device=device)

    # Get a global feature vector
    all_vectors = []
    for vector in tqdm(feature_loader, desc="Loading candidate vectors"):
        all_vectors.append(vector)
    all_vectors = torch.cat(all_vectors, dim=0).to(device)

    all_vectors_normalized = all_vectors / all_vectors.norm(dim=1, keepdim=True)

    # Compute distance
    for i in tqdm(range(num_samples), desc="Computing Top-K cosine distances"):
        vec_i = all_vectors[i].unsqueeze(0)
        vec_i_normalized = vec_i / vec_i.norm(dim=1, keepdim=True)
        similarities = torch.matmul(vec_i_normalized, all_vectors_normalized.T).squeeze(0)
        distances = 1 - similarities

        # exclude it self
        distances[i] = float('inf')

        # get top k nearest neighbors
        topk_dist, topk_idx = torch.topk(distances, k, largest=False)
        topk_distances[i] = topk_dist.to(dtype=torch.float16)
        topk_indices[i] = topk_idx

    print("Top-K distances tensor shape:", topk_distances.shape)
    print("Top-K indices tensor shape:", topk_indices.shape)
    print("First sample Top-10 distances:", topk_distances[0])
    print("First sample Top-10 indices:", topk_indices[0])

    return topk_distances, topk_indices


def prioritize_by_nns(cand_vectors):
    """NNS"""

    'Step 0: Initialize'
    k = 10  # same as NNS & FAST
    weight = 0.5  # same as NNS & FAST
    # weight = 1  # degrades to the original uncertainty-based method
    uncertainty_type = 'deepgini'  # same as NNS & FAST
    weighted = True
    weight_function = 'inverse'
    # weight_function = 'exp'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    'Step 1: Output Probabilities Extraction'
    prob_loader = get_prob_loader(cand_vectors)

    'Step 2: Construct Distance Matrix'
    topk_distances, topk_indices = construct_distance_mat(prob_loader, k)

    'Step 3: Get Smoothed Probability Vectors'
    # (1) Get tensor
    if isinstance(cand_vectors, list):
        cand_vectors_tensor = torch.stack(cand_vectors).float().to(device)

    # (2) Get probabilities of top k inputs
    topk_prob_vectors = cand_vectors_tensor[topk_indices]

    # (3) Calculate P_kNN
    if weighted:
        # Weighted sum
        if weight_function == 'inverse':
            epsilon = 1e-8
            weights = 1.0 / (topk_distances.float() + epsilon)
        elif weight_function == 'exp':
            weights = torch.exp(-topk_distances.float())
        else:
            raise ValueError("weight_function must be either 'inverse' or 'exp'")
        # Normalize weights
        weights = weights / weights.sum(dim=1, keepdim=True)
        # Expand weights to match the dimensions of topk_vectors
        weights = weights.unsqueeze(2)

        p_kNN = (weights * topk_prob_vectors).sum(dim=1)
    else:
        # Simple average
        p_kNN = topk_prob_vectors.mean(dim=1)

    p_M = cand_vectors_tensor

    # (4) Calculate P_smoothed
    p_smoothed = weight * p_M + (1 - weight) * p_kNN

    # (5) Get smoothed dataloder
    p_smoothed_np = p_smoothed.cpu().numpy().astype(np.float32)
    smoothed_dataset = ProbVecDataset(p_smoothed_np)
    smoothed_prob_loader = DataLoader(smoothed_dataset, batch_size=1024, shuffle=False)

    'Step 4: Prioritize by Uncertainty'
    if uncertainty_type == 'deepgini':
        prioritized_indices = prioritize_by_deepgini(smoothed_prob_loader, device)
    else:
        raise ValueError("Uncertainty Type not found")

    return prioritized_indices