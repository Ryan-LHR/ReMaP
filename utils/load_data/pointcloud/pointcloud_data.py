import os
import glob

import numpy as np
import torch
from torch.utils.data import TensorDataset

from utils.datasets import CustomTensorDataset


def _load_modelnet40_h5(data_dir, split, num_points=1024):
    """Read ModelNet40 HDF5 files (modelnet40_ply_hdf5_2048 format) for a split.

    Each .h5 has 'data' [n, 2048, 3] and 'label' [n, 1]. Returns (points[n, num_points, 3], label[n]).
    The first num_points are kept (deterministic -> sample order stable for index-aligned teacher features).
    """
    import h5py
    files = sorted(glob.glob(os.path.join(data_dir, '**', f'*{split}*.h5'), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No ModelNet40 '{split}' .h5 files under {data_dir} "
            f"(expected modelnet40_ply_hdf5_2048: ply_data_{split}*.h5).")
    all_data, all_label = [], []
    for f in files:
        with h5py.File(f, 'r') as h5:
            all_data.append(h5['data'][:])
            all_label.append(h5['label'][:])
    data = np.concatenate(all_data, axis=0)[:, :num_points, :].astype('float32')
    label = np.concatenate(all_label, axis=0).reshape(-1).astype('int64')
    return torch.from_numpy(data), torch.from_numpy(label)
    # return torch.from_numpy(data), label


def load_pointcloud_data(dataset_name, path_to_data, num_points=1024):
    """Load a point-cloud dataset as (train_set, test_set) TensorDatasets of (points[N,3], label).

    ModelNet40 expects the HDF5 release under {path_to_data}/modelnet40/ (ply_data_train*.h5 /
    ply_data_test*.h5; the modelnet40_ply_hdf5_2048 subfolder layout is also found recursively).
    Loaders use shuffle=False so teacher features align with the student by index.
    """
    if dataset_name == 'modelnet40':
        data_dir = os.path.join(path_to_data, 'modelnet40')
        train_x, train_y = _load_modelnet40_h5(data_dir, 'train', num_points)
        test_x, test_y = _load_modelnet40_h5(data_dir, 'test', num_points)
        return CustomTensorDataset(train_x, train_y), CustomTensorDataset(test_x, test_y)
    raise ValueError(f"Unknown point-cloud dataset: {dataset_name}")
