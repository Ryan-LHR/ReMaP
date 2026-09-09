import os
import json
import ast

import torch
import torch.nn as nn


def get_label_to_1k():
    """得到映射LABEL_TO_1K"""
    # (1) 读取IN-1k clsidx to labels
    txt_file = "imagenet1000_clsidx_to_labels.txt"
    with open(txt_file, "r", encoding="utf-8") as f:
        # 用 ast.literal_eval 解析成 Python 字典
        content = f.read()
        label_1k_dict = ast.literal_eval(content)

    # (2) 读取IN-100 code to labels
    json_file = "Labels.json"
    with open(json_file, "r", encoding="utf-8") as f:
        label_100 = json.load(f)
    # 按照编码()进行排序
    label_100_sorted = sorted(label_100.items(), key=lambda x: x[0])
    label_100_dict = {i: v for i, (k, v) in enumerate(label_100_sorted)}

    # (3) 寻找映射关系
    label_1k_inverted = {v: k for k, v in label_1k_dict.items()}

    LABEL_TO_1K = {}
    # 遍历 label_100_dict 的 key i (0~99)
    for i, label_str_100 in label_100_dict.items():
        # 在 label_1k_inverted 中找相同的字符串
        if label_str_100 in label_1k_inverted:
            j = label_1k_inverted[label_str_100]
            LABEL_TO_1K[i] = j
        else:
            # 如果没找到, 表示 label_100_dict[i] 字符串不在 label_1k_dict 值中
            # 你可以选择报错或跳过
            print(f"Warning: label '{label_str_100}' not found in label_1k_dict.")
            # LABEL_TO_1K[i] = None  # or skip
    print(f'{LABEL_TO_1K}')
    return LABEL_TO_1K

# get_label_to_1k()


def convert_to_1k(target):
    """
    将imagenet-100数据集上读取得到的labels转换为imagenet-1k上的labels，
    即能够在更换分类头之前，评估1k模型在100数据集上的性能
    """
    from utils.load_data.image.imagenet.LABEL_TO_1K import LABEL_TO_1K
    # 复制一份以免修改原 target
    new_target = target.clone()

    for i in range(len(new_target)):
        old_label = int(new_target[i].item())  # 取得整型值
        if old_label in LABEL_TO_1K:
            new_target[i] = LABEL_TO_1K[old_label]
        else:
            # 如果 old_label 不在字典中，你可选择报错或继续
            # e.g. new_target[i] remains as old_label or set to -1
            raise ValueError("Wrong")

    return new_target


class HFImgClsWrapper(nn.Module):
    """
    Wrapper for image classification tasks (for HF vision models)
    - Accepts either:
        1) a torch.Tensor pixel_values with shape [B, 3, H, W], or
        2) a dict-like batch containing 'pixel_values' (and optional keys)
    - Returns softmax probabilities [B, C], so you can directly do:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, inputs):
        # Case 1: inputs is a tensor -> treat as pixel_values
        if torch.is_tensor(inputs):
            out = self.hf_model(pixel_values=inputs)
            return torch.softmax(out.logits, dim=-1)

        # Case 2: inputs is a dict-like batch -> pass through
        if isinstance(inputs, dict):
            # Prefer explicit pixel_values if present, otherwise let HF handle keys
            out = self.hf_model(**inputs) if "pixel_values" in inputs else self.hf_model(**inputs)
            return torch.softmax(out.logits, dim=-1)

        raise TypeError(
            f"HFImgClsWrapper.forward expects a Tensor [B,,3H,W] or a dict batch, got: {type(inputs)}"
        )
