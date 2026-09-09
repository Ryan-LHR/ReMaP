# import keras
# import datetime
# import tensorflow as tf
import numpy as np
# from keras.applications import vgg19,resnet50
import math
import pickle
import sys
import os
# from datautils import get_data,get_model,data_proprecessing
# from keras.applications.vgg19 import preprocess_input
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from pathlib import Path

from baselines.confidence.prima.utils.accquire_prob import predict_proba_on_dataloader


def save_dict(filename,dictionary):
    dictfile = open(filename + '.dict', 'wb')
    pickle.dump(dictionary, dictfile)
    dictfile.close()


def load_dict(filename):
    dictfile = open(filename + '.dict', 'rb')
    a = pickle.load(dictfile)
    dictfile.close()
    return a

def core_prioritization_unit_new(
    model,
    dataloader,
    basedir,
    ptype,
    model_name,
    device=None,
):
    """
        使用传入的 PyTorch model 和 dataloader，统计每个样本在当前“变异模型”下的 kill 次数，
        并更新 file_name 对应的 kill_num_dict。

        参数：
            model      :  当前这一个“变异后的”PyTorch 模型
            dataloader :  对应测试集的 DataLoader（与 accquire_prob 时一致）
            file_name  :  kill_num_dict 的存储前缀（不含后缀），例如 ".../model_perturbation_ResNet20_GF"
            basedir    :  PRIMA 的根目录（与 accquire_prob 中的 basedir 相同），内部会使用 basedir / "predict_prob"
            ptype      :  变异类型字符串，例如 "GF" / "NEB" / "NAI" / "WS"
            file_id    :  当前变异模型的编号（用于中间结果命名），默认为 0
            device     :  PyTorch 设备，None 时自动从 model.parameters() 推断
        """

    # from utils.prob_utils import predict_proba_on_dataloader  # 按你实际放置位置调整
    # from utils.io_utils import load_dict, save_dict  # 同样按你的实际路径调整

    # 1) 统一路径为 Path
    basedir = Path(basedir)
    prob_dir = basedir / "predict_prob"
    prob_dir.mkdir(parents=True, exist_ok=True)

    # 2) 加载原始模型的预测结果（由 accquire_prob 事先生成）
    ori_prob_path = prob_dir / "predict_probability_vector.npy"
    if not ori_prob_path.exists():
        raise FileNotFoundError(
            f"[ERROR] {ori_prob_path} 不存在，请先对原始模型运行 accquire_prob()。"
        )
    ori_prob = np.load(ori_prob_path)
    origin_model_result = np.argmax(ori_prob, axis=1)  # shape: (num_samples,)
    num_samples = origin_model_result.shape[0]

    # 3) 加载 kill_num_dict
    model_dir = Path(basedir) / "mutated_model"
    file_name = 'model_perturbation_' + str(ptype)
    file_name = os.path.join(model_dir, file_name)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    kill_num_dict = {i: 0 for i in range(int(float(num_samples)))}
    save_dict(dictionary=kill_num_dict,filename=file_name)
    kill_num_dict = load_dict(file_name)  # 保持原接口，file_name 可以是 str 或 Path

    # 4) 打开日志文件
    log_path = Path(str(file_name) + ".txt")
    result_recording_file = open(log_path, "a", encoding="utf-8")

    # 5) 用当前“变异后的模型”对 X_test（即 dataloader）做一次完整预测
    if device is None:
        device = next(model.parameters()).device

    temp_result = predict_proba_on_dataloader(
        model,
        dataloader,
        device=device,
        apply_softmax=True,
    )  # shape: (num_samples, num_classes)

    if temp_result.shape[0] != num_samples:
        raise ValueError(
            f"[ERROR] temp_result 样本数 {temp_result.shape[0]} "
            f"与 ori_prob 样本数 {num_samples} 不一致，请检查 dataloader 是否对应同一测试集。"
        )

    # 6) 保存这一份变异模型的概率输出（可选，中间结果）
    temp_save_dir = prob_dir / f"model_temp_result_{ptype}"
    temp_save_dir.mkdir(parents=True, exist_ok=True)
    np.save(temp_save_dir / f"{ptype}.npy", temp_result)

    # 7) 统计 kill_num：与原始模型预测不同的样本 +1
    result = np.argmax(temp_result, axis=1)  # 当前变异模型的预测标签
    wrong_predict = []

    for idx in range(num_samples):
        if result[idx] != origin_model_result[idx]:
            kill_num_dict[idx] = kill_num_dict.get(idx, 0) + 1
            wrong_predict.append(idx)

    # 8) 写日志
    result_recording_file.write(f"ptype:{ptype}\n")
    result_recording_file.write(f"diff_num:{len(wrong_predict)}\n")
    result_recording_file.write(f"different_pred:{wrong_predict}\n")

    result_recording_file.close()

    # 9) 回写 kill_num_dict
    save_dict(dictionary=kill_num_dict, filename=file_name)

    print(
        f"[INFO] core_prioritization_unit done: ptype={ptype}, file_id={ptype}, "
        f"diff_num={len(wrong_predict)}"
    )