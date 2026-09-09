import os

import torch
from torch.utils.data import Dataset, Subset
import numpy as np

class VectorDataset(Dataset):
    """
    Custom dataset for vectors (probability, intermediate feature)
    No labels
    """
    def __init__(self, vectors):
        """
        Args:
            prob_vectors (list or tensor): List of prediction vectors (or a tensor of shape [N, D])
        """
        self.vectors = vectors

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        v = self.vectors[idx]
        return v

ProbVecDataset = VectorDataset


class CustomTensorDataset(Dataset):
    """
    Custom dataset for tensor inputs and ndarray labels
    (for most dataset, such as mnist, cifar10)
    """
    def __init__(self, inputs, labels):
        """
        Args:
            inputs (tensor)
            labels (np.ndarray)
        """
        self.inputs = inputs
        self.labels = labels
        # ensure each label is a int object.
        if isinstance(inputs, torch.Tensor):
            self.labels = np.array(self.labels)

        self.targets = self.labels

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]

        return x, y

    def __len__(self):
        return len(self.inputs)


