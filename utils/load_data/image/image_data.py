import os

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, MNIST, CIFAR10, SVHN

from utils.load_data.image.imagenet.imagenet import load_im_100
from utils.load_data.image.image_data_utils import *

def load_image_data(dataset_name, path_to_data, load_type):
    """
    Load Image Dataset
    Args:
    load_type (str): Specifies how to load the dataset. Options are:
        - "raw": Load raw dataset
        - "process": Process the raw data into tensor format or others.
        - "processed": Load the dataset from processed files(e.g., tensors).
    """
    if dataset_name == 'fashion_mnist':
        if load_type in ['raw', 'process']:
            train_set = FashionMNIST(root=os.path.join(path_to_data, "fashion_mnist/train"),
                                     train=True, download=True, transform=transforms.ToTensor())
            test_set = FashionMNIST(root=os.path.join(path_to_data, "fashion_mnist/test"),
                                    train=False, download=True, transform=transforms.ToTensor())
            if load_type == 'process':
                save_tensor_dataset(train_set, test_set, dataset_name, path_to_data)

        if load_type in ['process', 'processed']:
            train_set, test_set = load_tensor_dataset(dataset_name, path_to_data)

    elif dataset_name == 'mnist':
        if load_type in ['raw', 'process']:
            train_set = MNIST(root=os.path.join(path_to_data, "mnist/train"),
                              train=True, download=True, transform=transforms.ToTensor())
            test_set = MNIST(root=os.path.join(path_to_data, "mnist/test"),
                             train=False, download=True, transform=transforms.ToTensor())
            if load_type == 'process':
                save_tensor_dataset(train_set, test_set, dataset_name, path_to_data)

        if load_type in ['process', 'processed']:
            train_set, test_set = load_tensor_dataset(dataset_name, path_to_data)

    elif dataset_name == 'cifar_10':
        if load_type in ['raw', 'process']:
            train_set = CIFAR10(root=os.path.join(path_to_data, "cifar_10/train"),
                                train=True, download=True, transform=transforms.ToTensor())
            test_set = CIFAR10(root=os.path.join(path_to_data, "cifar_10/test"),
                               train=False, download=True, transform=transforms.ToTensor())
            if load_type == 'process':
                save_tensor_dataset(train_set, test_set, dataset_name, path_to_data)

        if load_type in ['process', 'processed']:
            train_set, test_set = load_tensor_dataset(dataset_name, path_to_data)

    elif dataset_name == 'svhn':
        if load_type in ['raw', 'process']:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            train_set = SVHN(root=os.path.join(path_to_data, "svhn/train"),
                             split='train', download=True, transform=transform)
            test_set = SVHN(root=os.path.join(path_to_data, "svhn/test"),
                            split='test', download=True, transform=transform)
            if load_type == 'process':
                save_tensor_dataset(train_set, test_set, dataset_name, path_to_data)

        if load_type in ['process', 'processed']:
            train_set, test_set = load_tensor_dataset(dataset_name, path_to_data)
        # 为测试集添加 class_to_idx 属性
        # trainset.class_to_idx = {str(i): i for i in range(10)}
        # testset.class_to_idx = {str(i): i for i in range(10)}

    elif dataset_name == 'imagenet_100':
        load_type = 'raw'
        # load_type = 'process_to_tensor'
        # load_type = 'tensor'
        train_set, test_set = load_im_100(path_to_data, load_type=load_type)


    else:
        raise ValueError("Dataset Not Found")

    return train_set, test_set
