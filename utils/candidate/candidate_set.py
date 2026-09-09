# -*-coding:utf-8-*-
import os

import numpy as np
from torch.utils.data import DataLoader

from utils.load_data.load_data import load_loader
from utils.candidate.corrupted.corrupted import load_corrupted_data
from utils.candidate.adversarial.adversarial import load_adversarial_data
from utils.data_utils import show_images, get_dataset_statistics, sample_subset


def construct_candidate(dataset_name, path_to_data, test_set, cand_type='nominal',
                        cand_size="all", batch_size=None, include_test=True,
                        validate=False, args=None, **kwargs):
    """
    Construct candidate set
    """
    print(f'\n{cand_type} candidate set constructing...')

    # (1) Load Candidate Set
    if cand_type == 'nominal':
        # namely the original test set (same as datis, tdpr)
        cand_set = test_set

    elif cand_type == 'corrupted':
        load_type = "process_to_tensor"
        # load_type = "tensor"
        cand_set = load_corrupted_data(dataset_name, path_to_data, load_type)

    elif cand_type == 'adversarial':
        model = kwargs["model"]
        model_name = kwargs["model_name"]
        test_loader = kwargs["test_loader"]

        # load_type = "not_generated"
        load_type = "processed"
        mix = True
        # mix = False
        cand_set = load_adversarial_data(dataset_name, model_name, path_to_data,
                                         load_type, model, test_set, test_loader, args.model_file, mix=mix)
    else:
        raise ValueError("Candidate Type Not Found")

    # (2) Uniform data size same as original test set
    if len(cand_set) != len(test_set):
        print(f'Random sampling from original candidate set size ({len(cand_set)})'
              f' to size ({len(test_set)})')
        cand_set = sample_subset(cand_set, size=len(test_set))

    if cand_size != "all":
        cand_set = sample_subset(cand_set, size=int(cand_size))  # subsample for test

    print(f'The size of candidate set: {len(cand_set)}')

    # (3) Create Dataloader
    kwargs = {'model_file': args.model_file, 'cand_type': cand_type}
    cand_loader = load_loader(dataset_name, cand_set, batch_size, **kwargs)

    # (4) Validate the candidate set
    # validate = True
    validate = False
    if validate == True:
        show_images(test_set, num_images=20)
        show_images(cand_set, num_images=20)
        get_dataset_statistics(test_set, str='Test Set')
        get_dataset_statistics(cand_set, str='Candidate Set')

    return cand_set, cand_loader
