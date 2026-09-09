# -*-coding:utf-8-*-
import os

import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset, Dataset
from torchvision import transforms
from timm.data import create_dataset
from PIL import Image
from tqdm import tqdm

from utils.candidate.corrupted.corrupted_imagenet_utils import SaveImageDataset

def process_to_tensor(dataset_name, inputs_path, labels_path, tensor_path):
    """Process ndarray to tensor (in image dataset)"""

    # load processed ndarray
    print(f"===> Loading processed ndarray from: {inputs_path}, {labels_path}")
    inputs_np = np.load(inputs_path)
    labels_np = np.load(labels_path)

    if dataset_name == 'fashion_mnist':
        inputs_np = np.expand_dims(inputs_np, axis=-1)

    print(f"===> Processing ndarray to Tensors")
    print("     Shape of raw inputs:", inputs_np.shape)
    print("     Shape of raw labels:", labels_np.shape)

    # convert to tensor
    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32).permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels_np)
    print("     Shape of tensor inputs:", inputs_tensor.shape)
    print("     Shape of tensor labels:", labels_tensor.shape)

    # normalization
    inputs_tensor = inputs_tensor.float() / 255.0

    # save to .pt file
    save_dir = os.path.dirname(tensor_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save((inputs_tensor, labels_tensor), tensor_path)

    return

def process_cifar10_c(inputs_dir, labels_path, tensor_path,
                      corruption_type, severity,
                      dataset_name, seed=42):
    """
    Load cifar10-c, and random select the same size samples with original test set
    """
    # (1) Setup
    if dataset_name == 'cifar_10':
        total_size = 10000
    elif dataset_name == 'svhn':
        total_size = 26032
    else:
        raise ValueError(f"Dataset {dataset_name} not found!")


    # load ndarray
    labels_np = np.load(labels_path)

    # get paths for corrupted data with various severity and type
    path_list = [
        os.path.join(inputs_dir, f'{t}_severity_{s}.npy')
        for t in corruption_type
        for s in severity
    ]

    # (2) Random selection from multiple files
    # allocate select size for each file
    num_files = len(path_list)
    base_samples_per_file = total_size // num_files
    remainder = total_size % num_files

    # e.g. for 5 files, size=10000 => base=2000, remainder=0
    # If remainder>0, we can add +1 to that many files to distribute leftover
    indices_per_file = [base_samples_per_file] * num_files
    for i in range(remainder):
        indices_per_file[i] += 1
    assert sum(indices_per_file) == total_size, f"Total sum {sum(indices_per_file)} " \
                                                f"is not equal to total size {total_size}"

    # random arrange all indices
    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(total_size)
    # all_indices = list(range(total_size))  # for test

    # iterate sample
    inputs_list = []
    labels_list = []
    indices_list = []
    start = 0
    for i, path in enumerate(path_list):
        # load and sample
        inputs_subset = np.load(path)
        sample_size = indices_per_file[i]
        # indices = rng.choice(total_size, size=sample_size, replace=False)
        indices = all_indices[start: start + sample_size]
        start += sample_size

        inputs_subset = inputs_subset[indices]
        labels_subset = labels_np[indices]

        inputs_list.append(inputs_subset)
        labels_list.append(labels_subset)
        indices_list.append(indices)

    inputs_np = np.concatenate(inputs_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)

    print(f"===> Processing ndarray to Tensors")
    print("     Shape of raw inputs:", inputs_np.shape)
    print("     Shape of raw labels:", labels_np.shape)

    # (3) Convert to tensor
    inputs_tensor = torch.tensor(inputs_np, dtype=torch.float32).permute(0, 3, 1, 2)
    labels_tensor = torch.tensor(labels_np)
    print("     Shape of tensor inputs:", inputs_tensor.shape)
    print("     Shape of tensor labels:", labels_tensor.shape)

    # normalization
    inputs_tensor = inputs_tensor.float() / 255.0

    # (4) Save to .pt file
    save_dir = os.path.dirname(tensor_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save((inputs_tensor, labels_tensor), tensor_path)

    return

def create_imagenet_c(path_to_data, corruption_type, severity, save_dir, dataset_name, seed=42):
    """
    Load imagenet-C, and random select the same size samples with original test set
    """
    # (1) Setup
    print('create imagenet-c')
    if dataset_name == 'imagenet_100':
        total_size = 5000
    else:
        raise ValueError

    dataset_dir = os.path.join(path_to_data, f'imagenet_100/corrupted')
    # get paths for corrupted data with various severity and type
    path_list = [
        os.path.join(dataset_dir, f'raw/{t}_severity_{s}')
        for t in corruption_type
        for s in severity
    ]

    # (2) Random selection from multiple files
    # allocate select size for each file
    num_files = len(path_list)
    base_samples_per_file = total_size // num_files
    remainder = total_size % num_files

    # e.g. for 5 files, size=10000 => base=2000, remainder=0
    # If remainder>0, we can add +1 to that many files to distribute leftover
    indices_per_file = [base_samples_per_file] * num_files
    for i in range(remainder):
        indices_per_file[i] += 1
    assert sum(indices_per_file) == total_size, f"Total sum {sum(indices_per_file)} " \
                                                f"is not equal to total size {total_size}"

    # random arrange all indices
    rng = np.random.default_rng(seed)
    all_indices = rng.permutation(total_size)

    # iterate sample
    indices_list = []
    subset_list = []
    start = 0
    for i, corrupted_dir in enumerate(path_list):

        file_set = create_dataset(
            root=corrupted_dir,
            name='',
            split='validation',
            download=False,
            load_bytes=False,
            class_map='',
        )
        # sample
        sample_size = indices_per_file[i]
        # indices = rng.choice(total_size, size=sample_size, replace=False)
        indices = all_indices[start: start + sample_size]
        start += sample_size

        subset_list.append(Subset(file_set, indices))
        indices_list.append(indices)

        # (3) Save img to a new folder
        save_dataset = SaveImageDataset(
            root=corrupted_dir,
            save_path=save_dir,
            transform=transforms.Compose([])
        )
        subset = Subset(save_dataset, indices)
        distorted_dataset_loader = torch.utils.data.DataLoader(
            subset, batch_size=1, shuffle=False, num_workers=1)

        for _ in tqdm(distorted_dataset_loader): continue

    # () Create new dataset from save folder
    # dataset = create_dataset(
    #     root=save_dir,
    #     name='',
    #     split='validation',
    #     download=False,
    #     load_bytes=False,
    #     class_map='',
    # )
    # return dataset
    return


CORR_ABBR = {
    'brightness': 'bri',
    'contrast': 'con',
    'defocus_blur': 'db',
    'elastic_transform': 'et',
    'fog': 'fog',
    'frost': 'fro',
    'gaussian_noise': 'gn',
    'glass_blur': 'gb',
    'impulse_noise': 'in',
    'jpeg_compression': 'jc',
    'motion_blur': 'mb',
    'pixelate': 'pix',
    'shot_noise': 'sn',
    'snow': 'sno',
    'zoom_blur': 'zb',
}

def get_corruption_str(corruption_type):
    """
    Convert corruption list to a compact string using CORR_ABBR table.

    Examples:
        ['brightness', 'defocus_blur', 'fog'] -> 'bri-db-fog'
    """
    def to_abbr(name):
        return CORR_ABBR.get(name, name)

    if isinstance(corruption_type, (list, tuple)):
        str_corruption = '-'.join(to_abbr(n) for n in corruption_type)
    else:
        str_corruption = to_abbr(corruption_type)

    return str_corruption
