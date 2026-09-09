"""
-------------------------------- Implementation Info --------------------------------

Implementation Type : Transplanted Official Implementation
Reference Paper     : "Prioritizing Test Inputs for Deep Neural Networks via Mutation Analysis"

Note:
    This implementation is developed based on the descriptions and algorithms (e.g., pseudocode)
    provided in the original paper.

--------------------------------------------------------------------------------------
"""
import numpy as np
from tqdm import tqdm

IMAGE_MODELS = ["MNIST-LeNet5", "FM-ResNet20", "C10-ResNet20", "SVHN-VGG16"]

from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader, Subset

from baselines.confidence.prima.utils.core_prioritization_unit import core_prioritization_unit_new
from baselines.confidence.prima.utils.feature_csv_conclusion import features_csv_conclusion
from baselines.confidence.prima.utils.feature_extraction import feature_extraction
from baselines.confidence.prima.utils.mutate_model import get_mutate_model
from baselines.confidence.prima.utils.accquire_prob import accquire_prob
from baselines.confidence.prima.utils.select_area_perturbated_generator import perturb_image
from baselines.confidence.prima.utils.model_feature_extraction import model_feature_extraction
from baselines.confidence.prima.utils.xgboost_predict import rank_prima_tests
from baselines.confidence.prima.utils.xgboost_train import train_prima_xgboost

def prioritize_by_prima(model, model_name, train_set, train_vectors,
                       train_correct, cand_loader, cand_labels,
                        batch_size, data_dir, cand_type, seed):
    """
    PRIMA (official implementation of PRIMA)

    This implementation integrates PRIMA's mutation, feature extraction,
    and ranking stages within the PyTorch framework, eliminating the need
    to execute separate shell scripts for each stage.
    """

    'Step 1: Setup'
    np.random.seed(seed)
    _, val_loader = build_val_loader(train_set, cand_loader)

    num_samples = len(cand_loader.dataset)
    input_mutation_image = [
        'gauss', 'reverse', 'black', 'white', 'shuffle'
    ]

    model_mutation = [
        'GF', 'WS', 'NAI', 'NEB'
    ]

    # Set Layer for model mutation
    if model_name == "MNIST-LeNet5":
        layer_name = 'fc1'
        # layer_name = 'fc2'
    elif model_name == "SVHN-VGG16":
        layer_name = 'fc1'
        # layer_name = 'fc2'
    elif model_name in ["FM-ResNet20", "C10-ResNet20"]:
        layer_name = 'fc1'
    else:
        raise ValueError("Model not found")

    if model_name in IMAGE_MODELS:
        input_mutation = input_mutation_image
        perturb_func = perturb_image
    else:
        raise ValueError("Model not found")

    save_dir = Path(data_dir)/"temp"/"prima"/model_name / cand_type
    save_dir_train = save_dir /"train"
    save_dir_cand = save_dir /"cand"

    if save_dir.exists():
        print(f"\nDelete the directory {save_dir}")
        shutil.rmtree(save_dir)

    'Step 2: Extract train features'
    train_input_feat_csv, train_model_feat_csv = get_mutated_features(
        model, model_name, perturb_func, input_mutation,
        model_mutation, val_loader, layer_name,
        save_dir_train, seed, is_train=True
    )

    'Step 3: Learning to Rank'
    rank_model, scaler, feat_names = train_prima_xgboost(train_input_feat_csv, train_model_feat_csv)

    'Step 4: Extract test features'
    cand_input_feat_csv, cand_model_feat_csv = get_mutated_features(
        model, model_name, perturb_func, input_mutation,
        model_mutation, cand_loader, layer_name,
        save_dir_cand, seed, is_train=False
    )

    'Step 5: Prioritization'
    ranking_df = rank_prima_tests(
        model=rank_model,
        scaler=scaler,
        feature_names=feat_names,
        input_feat_csv=cand_input_feat_csv,
        model_feat_csv=cand_model_feat_csv,
        id_col=None,
    )

    prioritized_indices = ranking_df["sample_idx"].tolist()

    # Remove the directory if it exists
    if save_dir.exists():
        print(f"\nDelete the directory {save_dir}")
        shutil.rmtree(save_dir)

    return prioritized_indices

def get_mutated_features(model, model_name, perturb_func, input_mutation, model_mutation,
                         dataloader, layer_name, save_dir, seed, is_train):

    'Step 2: Input Mutation'
    for m_type in input_mutation:
        print(f"\nCurrent input mutation type is {m_type}")
        perturb_func(save_dir, m_type, dataloader, model_name)
        accquire_prob(model, model_name, save_dir, m_type, dataloader)
        feature_extraction(model_name, save_dir, m_type)

    input_feat_csv = features_csv_conclusion(model_name, save_dir, "input", dataloader, is_train=is_train)

    'Step 3: Model Mutation'
    for m_type in model_mutation:
        print(f"\nCurrent model mutation type is {m_type}")
        mutated_model = get_mutate_model(model, m_type, seed, target_layer=layer_name)
        core_prioritization_unit_new(mutated_model, dataloader, save_dir, m_type, model_name)
        model_feature_extraction(model_name, save_dir, m_type)

    model_feat_csv = features_csv_conclusion(model_name, save_dir, "model", dataloader, is_train=is_train)

    return input_feat_csv, model_feat_csv


def build_val_loader(train_set, test_loader, seed=42):
    """
    从 train_set 随机抽取与 test_set 数量相同的样本作为 val_set，
    并创建与 test_loader 配置一致的 val_loader。
    """

    # (1) 获取 test_set 大小
    test_set = test_loader.dataset
    val_size = len(test_set)

    # (2) 固定随机种子，确保可复现
    g = torch.Generator()
    g.manual_seed(seed)

    # (3) 从 train_set 中随机抽取 val_size 个 index
    indices = torch.randperm(len(train_set), generator=g)[:val_size].tolist()

    # (4) 构建 Subset
    val_set = Subset(train_set, indices)

    # (5) 生成 val_loader，并复用 test_loader 的 batch_size、shuffle、num_workers 等配置
    val_loader = DataLoader(
        val_set,
        batch_size=test_loader.batch_size,
        shuffle=False,
        num_workers=test_loader.num_workers,
        pin_memory=getattr(test_loader, 'pin_memory', False)
    )

    return val_set, val_loader
