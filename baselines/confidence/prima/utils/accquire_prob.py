import os
# import time
import sys
# import datetime
import numpy as np
# import keras
# from keras.applications.vgg19 import VGG19
# import math
# from keras.models import Model
# import random
# from keras.datasets import mnist
# from numpy import arange
# import argparse
# from keras.applications import vgg19,resnet50
# from keras.applications.vgg19 import preprocess_input
# import re
# from datautils import get_data,get_model,data_proprecessing

# exp_id = sys.argv[1]
# ptype = sys.argv[2]
# samples = len(get_data(exp_id)[0])
# print(samples)
# samples = int(float(sys.argv[3]))
# import tensorflow as tf


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# from keras.backend.tensorflow_backend import set_session
import random
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
# if __name__=="__main__":
from pathlib import Path

from tqdm import tqdm
import torch

from baselines.confidence.prima.utils.select_area_perturbated_generator import collect_x_y_from_dataloader

def predict_proba_on_dataloader(model, dataloader, device=None, apply_softmax=True):
    """
    对原始测试集 dataloader 做一次完整预测，返回 (N, num_classes) 的 numpy 概率矩阵。
    用于生成 ori_prob。
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_probs = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # outputs 通常是 logits，做 softmax 转成概率
            if apply_softmax:
                outputs = torch.softmax(outputs, dim=1)

            all_probs.append(outputs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


def predict_proba_on_array(model, x_all, device=None, is_image=True, apply_softmax=True):
    """
    对单个文件里的所有 mutants 做预测：
    x_all: numpy array 或 torch.Tensor，形状通常是 (num_mutants, ...)
    返回 (num_mutants, num_classes) 的 numpy 概率矩阵。
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if isinstance(x_all, np.ndarray):
        inputs = torch.from_numpy(x_all).float()
    elif isinstance(x_all, torch.Tensor):
        inputs = x_all
    else:
        raise TypeError(f"Unsupported x_all type: {type(x_all)}")

    # 如果是图像且为 NHWC，则需要转成 NCHW；如果你保存时已经是 NCHW，这里就不要 permute
    if is_image and inputs.ndim == 4 and inputs.shape[-1] in (1, 3):
        # 假设当前是 (N, H, W, C)，转成 (N, C, H, W)
        inputs = inputs.permute(0, 3, 1, 2)

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        if apply_softmax:
            outputs = torch.softmax(outputs, dim=1)

    return outputs.cpu().numpy()


from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch  # 别忘了导入

def accquire_prob(model, model_name, basedir, ptype, dataloader, device=None):
    print(f"Accquiring Prob")
    # 确保 basedir 是 Path 对象
    basedir = Path(basedir)
    basedir.mkdir(parents=True, exist_ok=True)

    # 1) 准备 basedir / predict_prob 目录
    prob_dir = basedir / "predict_prob"
    prob_dir.mkdir(parents=True, exist_ok=True)

    predicting_file_path = prob_dir / 'predict_probability_vector.npy'

    # 2) 样本总数，用于 file_list
    num_samples = len(dataloader.dataset)

    # 3) 原始模型（PyTorch）
    origin_model = model
    if device is None:
        device = next(origin_model.parameters()).device

    # 4) 计算 / 加载原始测试集的预测概率 ori_prob
    if not predicting_file_path.exists():
        ori_prob = predict_proba_on_dataloader(origin_model, dataloader,
                                               device=device, apply_softmax=True)
        np.save(predicting_file_path, ori_prob)
    else:
        ori_prob = np.load(predicting_file_path, mmap_mode="r")  # mmap 可选

    # 5) 原始预测标签（按样本顺序）
    file_name = prob_dir / f'image_perturbation_{ptype}'
    dict_path = file_name.with_suffix(".dict")  # image_perturbation_xxx.dict

    # 6) 概率缓存路径（mutants 的 prob）
    spath = prob_dir / f"{model_name}_{ptype}_prob"

    # ==== 在这里做“是否已预测完成”的检查 ====
    if spath.exists():
        existing_prob_files = list(spath.glob("*.npy"))
        if len(existing_prob_files) >= num_samples and dict_path.exists():
            print(f"[SKIP] {model_name}-{ptype}: "
                  f"found {len(existing_prob_files)} prob files and dict, skip predicting.")
            return
    # ==========================================

    # 如果没完全生成，则继续后续流程
    result_recording_file = open(str(file_name) + '.txt', 'w', encoding='utf-8')

    origin_model_temp_result = ori_prob
    origin_model_result = np.argmax(origin_model_temp_result, axis=1)
    print('origin_prediction:', origin_model_result)
    result_recording_file.write(str(origin_model_result) + "\n")

    kill_num_dict = {}

    # 7) 变异样本存放路径 & 文件列表
    perturbate_image_path = basedir / "mutated_input" / ptype

    file_list = [perturbate_image_path / f"{i}.npy" for i in range(num_samples)]

    # 保证 prob 子目录存在（可能上面 exists 为 False）
    spath.mkdir(parents=True, exist_ok=True)

    # 8) 遍历每个原始样本对应的 mutants 文件，做 PyTorch 预测 & 统计 kill_num
    image_id = 0

    for file in tqdm(file_list, desc=f"Predicting ({perturbate_image_path})"):
        x_all = np.load(file)
        is_image = True  # 图像任务

        # 用 PyTorch 模型预测 mutants 的概率向量
        temp_result = predict_proba_on_array(origin_model, x_all,
                                             device=device,
                                             is_image=is_image,
                                             apply_softmax=True)

        # 保存每个原始样本对应 mutants 的 prob
        np.save(spath / f"{image_id}.npy", temp_result)

        # 计算 kill_num
        result = np.argmax(temp_result, axis=1)
        kill_num = 0
        for r in result:
            if r != origin_model_result[image_id]:
                kill_num += 1

        kill_num_dict[image_id] = kill_num
        result_recording_file.write('image_id:' + str(image_id) + '\n')
        result_recording_file.write('kill_num:' + str(kill_num) + '\n')

        image_id += 1

    # 9) 按 kill_num 排序并保存字典
    d2 = sorted(kill_num_dict.items(), key=lambda x: x[1], reverse=True)
    kill_num_dict = {score: letter for score, letter in d2}  # 保留你原来的结构

    with open(dict_path, 'wb') as dictfile:
        pickle.dump(kill_num_dict, dictfile)

    result_recording_file.close()
