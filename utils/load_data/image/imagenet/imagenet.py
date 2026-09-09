import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from timm.data import create_dataset
from tqdm import tqdm

from .in_utils import *

# same config as fastvit
data_config = {
    # 'input_size': (3, 256, 256),  # for fastvit
    'input_size': (3, 224, 224),  # for 224 x 224
    # 'input_size': (3, 384, 384),
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'crop_pct': 0.9,
    'crop_mode': 'center'

}
im_loader_args = dict(
    input_size=data_config["input_size"],
    batch_size=256,
    use_prefetcher=True,
    interpolation=data_config["interpolation"],
    mean=data_config["mean"],
    std=data_config["std"],
    num_workers=4,
    crop_pct=0.9,
    # pin_memory=False,
    pin_memory=True,
    tf_preprocessing=False,

)

def load_im_100(path_to_data, load_type='raw'):
    """
    Load ImageNet-100 Dataset
    load_type:
        raw: load the raw jpg images.
    """
    # (1) load raw images
    dataset_dir = os.path.join(path_to_data, f'imagenet_100')

    if load_type in ['raw', 'process_to_tensor']:
        train_raw = os.path.join(dataset_dir, 'raw/train')
        test_raw = os.path.join(dataset_dir, 'raw/val')
        print(f"===> Loading raw images from \n{train_raw}, \n{test_raw}")

        train_set = create_dataset(
            root=train_raw,
            name='',
            split='validation',
            download=False,
            load_bytes=False,
            class_map=''
        )
        test_set = create_dataset(
            root=test_raw,
            name='',
            split='validation',
            download=False,
            load_bytes=False,
            class_map=''
        )
        if load_type == 'raw':
            return train_set, test_set

        elif load_type in ['process_to_tensor']:
            # from .shard_dataset import materialize_loader_to_shards
            from utils.load_data.load_data import load_loader
            cache_root = Path(dataset_dir) / "tensor"
            train_cache_dir = cache_root / "train"
            test_cache_dir = cache_root / "val"

            # Materialize
            train_cache_dir.mkdir(parents=True, exist_ok=True)
            test_cache_dir.mkdir(parents=True, exist_ok=True)
            train_loader = load_loader('imagenet_100', train_set, batch_size=128)
            test_loader = load_loader('imagenet_100', test_set, batch_size=128)

            dtype = torch.float32  # for test
            # dtype = torch.float16
            shard_size = 1024

            process_to_individual_tensors(train_loader, base_path=train_cache_dir)
            process_to_individual_tensors(test_loader, base_path=test_cache_dir)

            print(f"\n===> Loading processed Tensors from \n{train_cache_dir}, \n{test_cache_dir}")
            train_set = IndividualTensorDataset(train_cache_dir)
            test_set = IndividualTensorDataset(test_cache_dir)

            # materialize_loader_to_shards(
            #     train_loader,
            #     save_dir=train_cache_dir,
            #     shard_size=shard_size,
            #     save_dtype=dtype,
            # )
            # materialize_loader_to_shards(
            #     test_loader,
            #     save_dir=test_cache_dir,
            #     shard_size=shard_size,
            #     save_dtype=dtype,
            # )
            #
            # # Return cached datasets (no transforms needed anymore)
            # train_set = ShardTensorDataset(train_cache_dir, cache_shards=1)
            # test_set = ShardTensorDataset(test_cache_dir, cache_shards=1)

    elif load_type in ['tensor']:
        cache_root = Path(dataset_dir) / "tensor"
        train_cache_dir = cache_root / "train"
        test_cache_dir = cache_root / "val"
        print(f"\n===> Loading processed Tensors from \n{train_cache_dir}, \n{test_cache_dir}")
        train_set = IndividualTensorDataset(train_cache_dir)
        test_set = IndividualTensorDataset(test_cache_dir)
    else:
        raise ValueError(f"Load type {load_type} not found!")

    return train_set, test_set

def process_to_individual_tensors(dataloader, base_path):
    """
    将 dataloader 内的每个样本都单独存储到一个 .pt 文件中。
    保存格式： (image, label)
    文件命名： base_path/sample{i}.pt
    """
    os.makedirs(base_path, exist_ok=True)

    sample_idx = 0
    for images, labels in tqdm(dataloader, desc="Process to tensor"):
        # images.shape = [B, C, H, W], labels.shape = [B]
        # 逐条遍历
        for i in range(len(images)):
            image_i = images[i].cpu()
            label_i = labels[i].cpu()

            save_path = os.path.join(base_path, f"sample{sample_idx}.pt")
            torch.save((image_i, label_i), save_path)
            sample_idx += 1

    print(f"Total {sample_idx} samples saved to {base_path}/*.pt")


class IndividualTensorDataset(Dataset):
    """
    以“一条数据一个 .pt 文件”的方式存储后，通过该 Dataset 索引和加载。
    假设每个 .pt 文件都包含 (image, label) 两个tensor。
    """

    def __init__(self, base_path):
        """
        Args:
            base_path (str): 存放 sample{i}.pt 文件的目录
        """
        self.base_path = base_path
        self.files = []

        # 收集所有 sample{idx}.pt 文件
        for filename in os.listdir(base_path):
            if filename.endswith(".pt") and filename.startswith("sample"):
                self.files.append(os.path.join(base_path, filename))

        # 按文件名排序 => 对应 sample0, sample1, ...
        # 这样就可以和当初保存的顺序一致
        self.files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][6:]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image, label = torch.load(file_path)  # 加载 (image, label)
        label = label.item()
        return image, label