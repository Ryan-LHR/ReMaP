import os

import torch
from torch.utils.data import DataLoader

from utils.datasets import CustomTensorDataset


def dataset_to_tensors(dataset, batch_size=512, num_workers=0, device=None):
    """
    Concatenate all the samples in the dataset into two tensors: inputs_tensor, labels_tensor
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_inputs, all_labels = [], []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        all_inputs.append(inputs)
        all_labels.append(labels)

    inputs_tensor = torch.cat(all_inputs,  dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    return inputs_tensor, labels_tensor

def save_tensor_dataset(train_set, test_set, dataset_name, path_to_data):
    """

    """
    train_tensor_path = os.path.join(path_to_data, f"{dataset_name}/train", "train_set_tensor.pt")
    test_tensor_path = os.path.join(path_to_data, f"{dataset_name}/test", "test_set_tensor.pt")

    train_inputs_tensor, train_labels_tensor = dataset_to_tensors(train_set, batch_size=512, device=None)
    torch.save((train_inputs_tensor, train_labels_tensor), train_tensor_path)

    test_inputs_tensor, test_labels_tensor = dataset_to_tensors(test_set, batch_size=512, device=None)
    torch.save((test_inputs_tensor, test_labels_tensor), test_tensor_path)

    print(f"===> Saving processed Tensors to: "
          f"\n   {train_tensor_path}"
          f"\n   {test_tensor_path}")


def load_tensor_dataset(dataset_name, path_to_data):
    """

    """
    train_tensor_path = os.path.join(path_to_data, f"{dataset_name}/train", "train_set_tensor.pt")
    test_tensor_path = os.path.join(path_to_data, f"{dataset_name}/test", "test_set_tensor.pt")

    print(f"===> Loading processed Tensors from: "
          f"\n   {train_tensor_path}"
          f"\n   {test_tensor_path}")

    inputs_t, labels_t = torch.load(train_tensor_path)
    train_set = CustomTensorDataset(inputs_t, labels_t)

    inputs_t, labels_t = torch.load(test_tensor_path)
    test_set = CustomTensorDataset(inputs_t, labels_t)

    return train_set, test_set

