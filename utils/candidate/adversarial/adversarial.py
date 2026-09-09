import torch
from torch.utils.data import ConcatDataset, Subset

from utils.data_utils import sample_subset
from utils.load_data.load_data import IMAGE_DATASETS
from utils.candidate.adversarial.adv_image import generate_adv_image, \
    load_adversarial_tensor_dataset, load_adversarial_im100

def load_adversarial_data(dataset_name, model_name, path_to_data, load_type, model, test_set, test_loader,
                          model_file, mix=True):
    """Load adversarial dataset"""
    print('\nloading adversarial dataset...')

    if dataset_name not in IMAGE_DATASETS:
        raise ValueError("Dataset not found!")
    # attack_types = ["fgsm", "bim", "pgd"]
    attack_types = ["fgsm", "bim", "pgd", "cw", "deepfool"]

    "(1) Generate"
    if load_type == 'not_generated':
        generate_adv_image(dataset_name, model_name, attack_types, model, test_loader, path_to_data)

    "(2) Load and Mix"
    if dataset_name == "imagenet_100":
        dataset = load_adversarial_im100(path_to_data, model_name, attack_types, mix)
    else:
        dataset = load_adversarial_tensor_dataset(dataset_name, model_name, path_to_data)
        if mix:
            dataset = concat_dataset(dataset, test_set)
    return dataset

def concat_dataset(cand_set, test_set, seed=42):
    """
    Construct a mixed dataset by taking half from test_set
    and half from cand_set
    """
    assert len(cand_set) == len(test_set), \
        f"Dataset size mismatch: cand_set={len(cand_set)}, test_set={len(test_set)}"
    total_size = len(test_set)
    split_a_size = total_size // 2
    # split_b_size = total_size - split_a_size

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=gen)

    idx_test = perm[:split_a_size]
    idx_cand = perm[split_a_size:]

    test_half = Subset(test_set, idx_test)
    cand_half = Subset(cand_set, idx_cand)

    cand_set_concat = ConcatDataset([test_half, cand_half])
    return cand_set_concat
