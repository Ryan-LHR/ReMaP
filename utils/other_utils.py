import os
import re
import gc
import pickle
import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone

from termcolor import colored
import torch


def get_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else
                            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return device

def get_datetime():
    """Get datetime"""
    time_zone = timezone(timedelta(hours=8))  # beijing timezone
    now = datetime.now(time_zone)
    time_str = now.strftime("%Y%m%d_%H%M")

    return time_str

# functions for print
def save_point_print(text='Divider'):
    """print divider with text"""
    divider_long = f'{"":-^50}'

    print('\n' + divider_long)
    print(f'{text:-^50}')
    print(divider_long)


def print_msg_box(msg, indent=4, width=40, title=None):
    """Print message-box with optional title."""
    lines = msg.split('\n')
    space = " " * indent
    if not width:
        width = max(map(len, lines))
    box = f'╔{"═" * (width + indent * 2)}╗\n'  # upper_border
    if title:
        box += f'║{space}{title:^{width}}{space}║\n'  # title
        box += f'║{space}{"-" * len(title):^{width}}{space}║\n'  # underscore
    box += ''.join([f'║{space}{line:<{width}}{space}║\n' for line in lines])
    box += f'╚{"═" * (width + indent * 2)}╝'  # lower_border
    print(box)

def color_print(str, color, attrs=None, end='\n'):
    """Print with color"""
    print(colored(str, color, attrs=attrs), end=end)
    """
    Usage
    print(colored("load LeNet-5 model and MNIST data sets", "blue"))
    """


def setup_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    import random
    import numpy as np

    # Set random seeds for Python and NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Set random seeds for PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # Set random seeds for TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

def cleanup():
    """Clean up"""
    # cleanup ram
    gc.collect()

    # cleanup gpu
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# functions for save data
def save_to_pickle(path, data):
    """Save data to a pickle file"""
    folder_path, file_name = os.path.split(path)  # split directory and filename
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(path, 'wb') as file:
        pickle.dump(data, file)
        print("Save data to: %s" % path)

def load_from_pickle(path):
    """Load data from a pickle file"""
    with open(path, 'rb') as file:
        data_loaded = pickle.load(file)
        print("Load data from: %s" % path)
    return data_loaded

def load_prediction_results(args, label, **kwargs):
    """Load and save model's prediction results"""

    save_dir = Path(args.data_dir)/"temp"/"prediction"/args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir/f'{label}_prediction.pt')

    if not save_path.exists():
        print(f"Prediction results is not found in: {save_path}")
        save_prediction_results(save_path, label, **kwargs)

    prediction = torch.load(save_path)
    print(f"Prediction results is loaded from: {save_path}")

    return prediction

def save_prediction_results(save_path, label, **kwargs):
    """Save prediction results"""
    from utils import get_predictions

    model = kwargs['model']
    dataloader = kwargs['loader']
    device = kwargs['device']

    (pred_vectors, pred_labels, pred_correct) = get_predictions(
        model, dataloader, return_type='Tensor', is_print=True, data_name=label, device=device)

    torch.save((pred_vectors, pred_labels, pred_correct), save_path)
    print(f"Prediction results is saved to: {save_path}")
    return

def check_consistency(list_test, list_cand):
    """Check the consistency of prediction result"""

    assert len(list_test) == len(list_cand), \
        f"Length mismatch: test={len(list_test)}, cand={len(list_cand)}"

    for i, (t, c) in enumerate(zip(list_test, list_cand)):
        assert int(t.item()) == int(c.item()), \
            f"Mismatch at index {i}: test={int(t.item())}, cand={int(c.item())}"
    return

def load_features(args, layer_name, label, **kwargs):
    """Load and save model's features"""

    save_dir = Path(args.data_dir)/"temp"/"feature"/args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir/f'{label}_feature_{layer_name}.pt')

    if not save_path.exists():
        print(f"Feature representation is not found in: {save_path}")
        save_features(save_path, layer_name, **kwargs)

    support_output = torch.load(save_path)
    print(f"Feature representation is loaded from: {save_path}")

    return support_output

def save_features(save_path, layer_name, **kwargs):
    """Save features"""
    from utils import extract_layer_output, flatten_layer_output

    model = kwargs['model']
    dataloader = kwargs['loader']

    support_output = extract_layer_output(model, dataloader, layer_name)
    support_output = flatten_layer_output(support_output)

    # torch.save(support_output, save_path)
    torch.save(support_output, save_path, pickle_protocol=5)
    print(f"Feature representation is saved to: {save_path}")
    return support_output

def get_result_save_path(args):
    """
    Get the save path of Result Dict to csv
    """
    save_path_csv = os.path.join(args.save_dir_result,
                                 f"{args.method}_"
                                 f"{args.cand_type}_"
                                 f"{str(args.ex_time)}_"
                                 f"seed_{args.seed_list}.csv")
    return save_path_csv

def get_indices_save_path(args):
    """
    Get the save path of Prioritized Indices
    """
    save_path_indices = os.path.join(args.save_dir_result, 'prioritized_indices',
                                     f"{args.method}_"
                                     f"{args.cand_type}_"
                                     f"seed-{args.seed}.pkl")
    return save_path_indices

def get_models_save_path(args, test_acc):
    """
    Get the save path of Retrained models
    """
    from pathlib import Path
    model_save_dir = os.path.join(args.save_dir_result,
                                  "retrained_models",
                                  f"{args.method}")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_name = f"{Path(args.model_file).stem}" \
                      f"-budget({args.budget})" \
                      f"-split_seed({args.split_seed}" \
                      f"-seed({args.seed})" \
                      f"-acc({test_acc:.4f}).pth"
    save_path_models = os.path.join(model_save_dir, model_save_name)

    return save_path_models

def save_result_to_csv(save_path, result_dict, args):
    """Save result to a CSV file / 将数据保存至 CSV 文件"""
    if args.debug_mode:
        return

    folder_path, filename = os.path.split(save_path)  # 分割文件路径和文件名
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 写入csv文件 'a+'代表追加, 不覆盖原有数据
    with open(save_path, 'a+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['args', args])  # Params

        for key, value in result_dict.items():  # write results
            writer.writerow([key, value])
        writer.writerow(['------------------'])  # write divider

def save_result_dict(result_dict, args, result_type):
    """
    Get the save path of Result Dict pkl
    """
    if args.debug_mode:
        return

    if result_type == 'single':
        seed = args.seed
    elif result_type == 'avg':
        seed = args.seed_list

    if args.rq == "rq1-priorization":
        save_path = os.path.join(args.save_dir_result, "result_dicts",
                                     f"{args.method}-"
                                     f"{args.cand_type}-"
                                     f"{args.cand_size}-"
                                     f"seed_{seed}"
                                     f".pkl")
    elif args.rq == "rq-retrain":
        save_path = os.path.join(args.save_dir_result, "result_dicts",
                                 f"{args.method}-"
                                 f"{args.cand_type}-"
                                 f"budget_{args.budget}-"
                                 f"split_seed_{args.split_seed}-"
                                 f"seed_{seed}"
                                 f".pkl")
    else:
        raise ValueError("Can not find match RQ")
    save_to_pickle(save_path, result_dict)










