import os

import numpy as np
from torch.utils.data import TensorDataset

from utils.data_utils import CustomTensorDataset
from utils.candidate.corrupted.corrupted_utils import *

CORRUPTED_TENSOR_DATASETS = [
    'fashion_mnist', 'mnist', 'cifar_10', 'svhn'
]


def load_corrupted_data(dataset_name, path_to_data, load_type="process"):
    """
    Load corrupted dataset.

    Args:
        load_type (str): Specifies how to load the dataset. Options are:
            - "process": Process the raw data into tensor format or others.
            - "processed": Load the dataset from processed files(e.g., tensors).
    """
    print(f'\nloading corrupted dataset: {dataset_name}-c')
    inputs_file = 'None'
    labels_file = 'None'
    # (1) Set load path
    dataset_dir = os.path.join(path_to_data, f'{dataset_name}/corrupted')

    tensor_path = os.path.join(dataset_dir, 'tensor/data_corrupted.pt')

    if dataset_name == 'fashion_mnist':
        # load (same as simple-tip)
        inputs_file = 'fmnist-c-test.npy'
        labels_file = 'fmnist-c-test-labels.npy'
        load_type = 'processed'

    elif dataset_name == 'mnist':
        # load (same as datis)
        inputs_file = 'data_corrupted.npy'
        labels_file = 'label_corrupted.npy'
        load_type = 'processed'

    elif dataset_name in ['cifar_10', 'svhn']:
        # cifar10-c: downloaded from cifar10-c
        # svhn-c: construct following cifar10-c
        inputs_file = 'inputs'
        labels_file = 'labels/labels.npy'
        # severity = list(range(1, 6))
        severity = [1, 2]
        corruption_type = [
            'brightness',
            'contrast',
            'defocus_blur',
            'elastic_transform',
            'fog',
            'frost',
            'gaussian_noise',
            'glass_blur',
            'impulse_noise',
            'jpeg_compression',
            'motion_blur',
            'pixelate',
            'shot_noise',
            'snow',
            'zoom_blur',
        ]

        # load_type = 'raw'
        load_type = 'processed'
    elif dataset_name in ['imagenet_100']:
        severity = [1]
        # severity = [4]
        # severity = [5]
        corruption_type = [
            'brightness',  # good
            'contrast',  # moderate
            'defocus_blur',  # bad
            'elastic_transform',  # moderate
            'fog',  # moderate
            'frost',  # moderate
            'gaussian_noise',  # moderate moderate
            'glass_blur',  # moderate good
            'impulse_noise',  # moderate
            'jpeg_compression',  # moderate
            'motion_blur',  # moderate
            'pixelate',  # moderate good
            'shot_noise',  # moderate good
            'snow',  # moderate good
            'zoom_blur',  # bad
        ]
        str_corruption = get_corruption_str(corruption_type)
        save_dir = os.path.join(dataset_dir, f"sampled/{str_corruption}_severity_{severity}")  # orig
        # save_dir = os.path.join(dataset_dir, f"sampled/{str_corruption}_severity_{severity}_mixed")  # mixed

        # load_type = 'raw'
        load_type = 'processed'

    inputs_path = os.path.join(dataset_dir, 'ndarray', inputs_file)
    labels_path = os.path.join(dataset_dir, 'ndarray', labels_file)

    # (2) Process to Tensor
    if load_type == 'raw':
        if dataset_name in ['fashion_mnist', 'mnist']:
            process_to_tensor(dataset_name, inputs_path, labels_path, tensor_path)

        elif dataset_name in ['cifar_10', 'svhn']:
            process_cifar10_c(inputs_path, labels_path, tensor_path,
                              corruption_type, severity, dataset_name)

        elif dataset_name in ['imagenet_100']:
            create_imagenet_c(path_to_data, corruption_type, severity,
                                        save_dir, dataset_name)

    # (3) Loading Processed Dataset
    if dataset_name in CORRUPTED_TENSOR_DATASETS:
        print(f"===> Loading processed Tensors from: {tensor_path}")
        inputs_t, labels_t = torch.load(tensor_path)
        dataset = CustomTensorDataset(inputs_t, labels_t)

    elif dataset_name in ['imagenet_100']:
        print(f"===> Loading processed Images from: {save_dir}")
        dataset = create_dataset(
            root=save_dir,
            name='',
            split='validation',
            download=False,
            load_bytes=False,
            class_map='',
        )
    else:
        raise ValueError(f"This dataset {dataset_name} has no corrupted version")

    return dataset