import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils.other_utils import save_to_pickle, load_from_pickle, save_point_print
from utils.datasets import *

def load_dataset_to_ndarray(dataset):
    """
    Convert the inputs of torch.dataset from tensor to ndarray with size (N, C, H, W) and (N, F)

    Args:
        dataset (Dataset): PyTorch Dataset

    Return:
        np.ndarray: ndarray with size of (N, C, H, W)
    """
    all_inputs = []
    for idx in tqdm(range(len(dataset)), desc='Convert to ndarray'):
        input, label = dataset[idx]  # Dataset return (input, label)
        # ensure input type is PyTorch Tensor
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        # elif isinstance(input, Image.Image):
        #     input = np.array(input)
        elif isinstance(input, np.ndarray):
            None
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

        # Ensure the image shape is (C, H, W)
        if input.ndim == 3 and input.shape[0] in [1, 3]:
            all_inputs.append(input)
        elif input.ndim == 2:
            # If it's a grayscale image, add the channel dimension
            input = np.expand_dims(input, axis=0)
            all_inputs.append(input)

        elif input.ndim == 1:
            # input = np.expand_dims(input, axis=0)
            all_inputs.append(input)

        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

    # Stack all input, resulting in a shape of (N, C, H, W)
    x = np.stack(all_inputs, axis=0)
    return x

def show_images(dataset, num_images=10):
    """
    Display images from a PyTorch dataset using matplotlib
    """
    num_cols = 5
    num_rows = math.ceil(num_images / num_cols)
    plt.figure(figsize=(num_cols * 3, num_rows * 3))

    for i in range(num_images):
        image, label = dataset[i]

        # Ensure the image is on CPU and convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # If the tensor has shape like (C, H, W), we need to reshape/transpose for imshow
        # - grayscale: (1, H, W) -> (H, W)
        # - RGB: (3, H, W) -> (H, W, 3)
        if image.ndim == 3 and image.shape[0] == 1:
            # single-channel (grayscale)
            image = image[0]  # now shape is (H, W)
            cmap = "gray"
        elif image.ndim == 3 and image.shape[0] == 3:
            # likely multi-channel, e.g. (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            cmap = None
        else:
            # e.g. already (H, W)
            cmap = "gray" if image.ndim == 2 else None

        # create a subplot
        ax = plt.subplot(num_rows, num_cols, i + 1)
        ax.imshow(image, cmap=cmap)
        ax.set_title(f"Label: {label}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def get_dataset_statistics(dataset: Dataset, sample_ratio=0.1, seed=0, str=None):
    """
    Compute min, max, mean, and variance of given dataset.
    """
    print(f'\nCompute statistics values of {str}')
    # Show the first sample and label
    first_data, first_label = dataset[0]
    # if isinstance(first_data, torch.Tensor):
    #     first_data = first_data.cpu().detach().numpy()
    print("First sample data:\n", first_data)
    print("First label:", first_label)
    print("-----")

    # sample instances
    if seed is not None:
        random.seed(seed)

    num_samples = len(dataset)
    sample_size = int(num_samples * sample_ratio)
    sample_indices = random.sample(range(num_samples), sample_size)

    all_data = []

    for idx in sample_indices:
        data, _ = dataset[idx]
        if isinstance(data, torch.Tensor):
            data = data.cpu().detach().numpy()
        elif isinstance(data, Image.Image):
            data = np.array(data)
        all_data.append(data.ravel())

    all_data = np.concatenate(all_data, axis=0)

    # calculate
    data_min = float(all_data.min())
    data_max = float(all_data.max())
    data_mean = float(all_data.mean())
    data_var = float(all_data.var())

    print(f"Sampled {sample_size} items ({sample_ratio*100:.1f}%):")
    print(f"  min:  {data_min}")
    print(f"  max:  {data_max}")
    print(f"  mean: {data_mean}")
    print(f"  var:  {data_var}")

    return data_min, data_max, data_mean, data_var

def sample_subset(dataset, ratio=None, size=None, seed=42):
    """
    Randomly sample a subset from a given dataset using either ratio or an size.
    """

    # Initialize an independent RNG generator
    gen = torch.Generator()
    gen.manual_seed(seed)

    total_size = len(dataset)

    # Determine sample_size based on ratio or size
    if ratio is not None:
        sample_size = int(total_size * ratio)
    elif size is not None:
        sample_size = size
    else:
        raise ValueError("Either 'ratio' or 'size' must be provided.")

    # Create a list of randomly permuted indices
    indices = torch.randperm(total_size, generator=gen)[:sample_size]

    # Construct and return a Subset
    sampled_dataset = Subset(dataset, indices)
    return sampled_dataset

def extract_targets(dataset):
    """
    Extract targets from a torch.Dataset object
    """
    if hasattr(dataset, "targets"):
        return dataset.targets

    if isinstance(dataset, Subset):
        base = dataset
        while isinstance(base, Subset):
            base = base.dataset

        if hasattr(base, "targets"):
            return base.targets[dataset.indices]
        # if not hasattr(base, "targets"):
        #     raise AttributeError("Underlying dataset has no 'targets'.")

    targets = []
    for i in range(len(dataset)):
        sample = dataset[i]
        targets.append(sample[1])
    return targets

def load_truths(args, dataset, label):
    """Load and save dataset's truths"""

    save_dir = Path(args.data_dir)/"temp"/"dimp"/"truths"/args.model
    if "IM100Test" in args.model:  # for test
        save_dir = Path(args.data_dir) / "temp" / "dimp" / "truths" / "IM100-FastViT_S12"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir/f'{label}_truths.pkl')

    if not save_path.exists() or args.debug_mode:
        print(f"\nTruths is not found in: {save_path}")
        y = extract_targets(dataset)

        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        save_to_pickle(save_path, y)

    y = load_from_pickle(save_path)
    print(f"Truths is loaded from: {save_path}")

    return y

def extract_truths(dataset):
    """
    Extract targets from a torch.Dataset object
    """
    return [label for _, label in dataset]


def statistic_dataset(dataset, label=None):
    """
    Basic dataset statistics
    """
    labels = extract_truths(dataset)
    labels = [label for _, label in dataset]

    counter = Counter(labels)
    num_classes = len(counter)
    total_samples = len(labels)

    class_counts = dict(sorted(counter.items()))
    counts = list(class_counts.values())

    max_count = max(counts)
    min_count = min(counts)
    is_balanced = (max_count == min_count)

    print(f"\n---------Dataset Statistics: {label}---------"
          f"\nTotal samples: {total_samples}"
          f"\nNumber of classes : {num_classes}"
          f"\nSamples per class :")
    for c, n in class_counts.items():
        print(f"----Class {c:<3}: {n}")

    print(f"Class balanced: {is_balanced}")
    if not is_balanced:
        print(f"----Max / Min count: {max_count} / {min_count}")

    return

def downsample_debug(train_loader, test_loader, cand_set, cand_loader, args):
    """Downsample for debug"""

    '(1) Setup'
    from utils.load_data.load_data import load_loader
    seed = args.seed
    save_point_print('Debug mode is on !!!')
    args.save_result = False
    # args.load_prediction = False
    # args.load_feature = False
    args.data_dir = Path(args.data_dir)/"temp"/f"debug_{args.downsample_ratio}"

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    cand_size = len(cand_set)

    def _sample_size(size):
        downsample_size = int(size * args.downsample_ratio)
        return max(1, downsample_size)

    kwargs = {'model_file': args.model_file}
    train_set = sample_subset(train_loader.dataset, size=_sample_size(train_size), seed=seed)  # subsample for test
    train_loader = load_loader(args.dataset, train_set, args.batch_size, **kwargs)

    test_set = sample_subset(test_loader.dataset, size=_sample_size(test_size), seed=seed)  # subsample for test
    test_loader = load_loader(args.dataset, test_set, args.batch_size, **kwargs)

    cand_set = sample_subset(cand_set, size=_sample_size(cand_size), seed=seed)  # subsample for test
    cand_loader = load_loader(args.dataset, cand_set, args.batch_size, **kwargs)


    return train_loader, test_loader, cand_set, cand_loader


def get_modality(dataset_name):
    """Resolve a dataset name to its modality; an unregistered dataset is an error, never a default."""
    from utils.load_data.load_data import DATASET_MODALITY

    for modality, datasets in DATASET_MODALITY.items():
        if dataset_name in datasets:
            return modality
    raise ValueError(f"Unknown dataset {dataset_name}, register it in DATASET_MODALITY: {list(DATASET_MODALITY)}")
