from tqdm import tqdm
from torch.utils.data import DataLoader
from timm.data import create_loader

from utils.load_data.image.image_data import load_image_data
from utils.load_data.pointcloud.pointcloud_data import load_pointcloud_data
from utils.load_data.audio.audio_data import load_audio_data
from utils.load_data.image.imagenet.imagenet import im_loader_args
from utils.data_utils import statistic_dataset

IMAGE_DATASETS = ['fashion_mnist', 'mnist', 'cifar_10', 'svhn', 'imagenet_100']
POINTCLOUD_DATASETS = ['modelnet40']
AUDIO_DATASETS = ['esc50']

DATASET_MODALITY = {
    'image': IMAGE_DATASETS,
    'pointcloud': POINTCLOUD_DATASETS,
    'audio': AUDIO_DATASETS,
}


def load_data(dataset_name, path_to_data, batch_size, n_workers, args):
    """
    Load the dataset

    Args:
        dataset (str): dataset name
        path_to_data (str): path
        ...

    Returns:
        (train_loader, test_loader): dataloader and test loader
    """
    print('\nloading data...')

    if dataset_name in IMAGE_DATASETS:
        # load_type = 'raw'
        # load_type = 'process'
        load_type = 'processed'
        train_set, test_set = load_image_data(dataset_name, path_to_data, load_type=load_type)

    elif dataset_name in POINTCLOUD_DATASETS:
        train_set, test_set = load_pointcloud_data(dataset_name, path_to_data)
    elif dataset_name in AUDIO_DATASETS:
        train_set, test_set = load_audio_data(dataset_name, path_to_data)
    else:
        raise ValueError("Dataset Not Found")

    print(f'The size of train set: {len(train_set)}')
    print(f'The size of test set: {len(test_set)}')

    # statistic_dataset(train_set, label='Train')
    # statistic_dataset(test_set, label='Test')

    # Create Dataloader
    kwargs = {'model_file': args.model_file}
    train_loader = load_loader(dataset_name, train_set, batch_size, n_workers, **kwargs)
    test_loader = load_loader(dataset_name, test_set, batch_size, n_workers, **kwargs)

    return (train_loader, test_loader)

def load_loader(dataset_name, dataset, batch_size, n_workers=1, **kwargs):
    """
    Load the dataloader
    """

    common_loader_args = dict(
        num_workers=n_workers
    )

    if dataset_name == 'imagenet_100':
        im_loader_args["batch_size"] = batch_size
        # im_loader_args["batch_size"] = 256  # for test
        # im_loader_args["num_workers"] = n_workers
        im_loader_args["num_workers"] = 4
        # im_loader_args["num_workers"] = 1

        method = kwargs.get("method", None)
        if method == "fast":
            im_loader_args["num_workers"] = 1
        cand_type = kwargs.get('cand_type', None)
        if cand_type == 'adversarial':
            None
        print(f"Current num_workers is: {im_loader_args['num_workers']}")
        dataloader = create_loader(dataset, **im_loader_args)

    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **common_loader_args)

    return dataloader
