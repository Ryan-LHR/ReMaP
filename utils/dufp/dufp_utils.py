import re
from pathlib import Path

import numpy as np
import torch

from baselines.confidence.simple_confidence import is_probability_vector, process_with_softmax
from utils.other_utils import load_from_pickle, save_to_pickle
from utils.data_utils import extract_targets, load_truths


def get_target_layer_name(model_name):
    """
    Specify the second last layer(penultimate) of DNN under test
    """

    if model_name == "MNIST-LeNet5":
        layer_name = 'fc2'
    elif model_name == "SVHN-VGG16":
        layer_name = 'fc2'
    elif model_name in ["FM-ResNet20", "C10-ResNet20"]:
        layer_name = 'avg_pool'
    elif model_name == "IM100Test-deit_base_patch16_224":
        layer_name = 'hf_model.classifier'
    elif model_name == "ModelNet40-DGCNN":
        layer_name = 'dp2'  # nn.Dropout(0.5) -- identity in eval, so hook.output == penultimate [B,256]
    elif model_name == "ESC50-AST":
        layer_name = 'penultimate'  # AudioHFClassifier._MeanTimePool -- hook.output == hidden_states[-1].mean(1)

    else:
        raise ValueError("Model not found")

    return layer_name

def try_get_param(method, key, type, default, is_compound=False):
    try:
        if is_compound:
            return type(get_param_paren_nested(method, key))
        else:
            return type(get_param_paren(method, key))
    except:
        print(f"{key} is not given in method")
        return default


def get_param_paren(method: str, key: str, default=None) -> str:
    """
    Extract param from string
    e.g. "dimp_alpha(0.9)_beta(0.8)" -> get_param_paren(..., 'alpha') = '0.9'
    """
    pattern = rf"{key}\(([^)]+)\)"
    match = re.search(pattern, method)
    if match:
        return match.group(1)
    if default is not None:
        return default
    raise ValueError(f"{key} not found in: {method}")


def get_param_paren_nested(method: str, key: str, default=None) -> str:
    """
    Extract key(...) content with nested parentheses support.
    e.g. dist(euclid_neighbors(100)) -> 'euclid_neighbors(100)'
    """
    token = f"{key}("
    s = method.find(token)
    if s == -1:
        if default is not None:
            return default
        raise ValueError(f"{key} not found in: {method}")

    i = s + len(token)  # start after '('
    depth = 1
    while i < len(method) and depth:
        ch = method[i]
        depth += (ch == '(') - (ch == ')')
        i += 1

    if depth != 0:
        if default is not None:
            return default
        raise ValueError(f"Unbalanced parentheses for {key} in: {method}")

    return method[s + len(token): i - 1]


def get_labels_and_classes(
    args,
    model_name,
    train_loader,
    cand_loader,
):
    """
    Extract true labels for train and candidate sets, and infer class info.

    Returns:
        y_train (np.ndarray): true labels of training set
        y_cand (np.ndarray): true labels of candidate set
        classes (np.ndarray): unique class labels
        num_classes (int): number of classes
    """

    # Load saved truth (for large-scale datasets)
    if args.dataset in ["imagenet_100"]:
        y_train = load_truths(args, train_loader.dataset, label='Train')
        y_cand = load_truths(args, cand_loader.dataset, label=f'Candidate_{args.cand_type}')

    else:
        y_train = extract_targets(train_loader.dataset)
        y_cand = extract_targets(cand_loader.dataset)

        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()
            y_cand = y_cand.cpu().numpy()


    # Data type conversion
    if not isinstance(y_train, list):
        y_train = y_train.tolist()

    if not isinstance(y_cand, list):
        y_cand = y_cand.tolist()

    assert isinstance(y_train, list), \
        f"y_train must be a list, but got {type(y_train)}"

    return y_train, y_cand


def convert_vectors_to_probs(pred_vectors):
    """
    Convert pred vectors to probability vectors
    """
    # validate whether pred_vectors is composed of probability vectors
    for i, vec in enumerate(pred_vectors[:5]):
        if is_probability_vector(vec):
            print(f"pred_vectors[{i}] is a valid probability vector.")
            is_probability = True
        else:
            print(f"pred_vectors[{i}] {vec} is NOT a valid probability vector with sum {vec.sum()}.")
            is_probability = False

    if not is_probability:
        prob_vectors = process_with_softmax(pred_vectors)
    else:
        prob_vectors = pred_vectors

    prob_vectors = np.array([tensor.cpu().numpy() for tensor in prob_vectors])
    return prob_vectors