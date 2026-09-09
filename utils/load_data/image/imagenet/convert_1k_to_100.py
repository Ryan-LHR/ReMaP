import torch
import torch.nn as nn
from typing import Optional, Tuple, List


def _find_classifier_linear(model: nn.Module, _prefix: str = "model"):
    """
    返回 (parent_module, attr_name, linear_layer, full_path)
    full_path 例如 "model.hf_model.cls_classifier"
    """
    # ---- 第一步：在当前 model 上检查常见分类头字段 ----
    candidates = ["head", "fc", "classifier", "last_linear",
                  "cls_head", "cls_classifier"]
    for name in candidates:
        if hasattr(model, name):
            mod = getattr(model, name)
            full_path = f"{_prefix}.{name}"
            if isinstance(mod, nn.Linear):
                return model, name, mod, full_path
            if isinstance(mod, (nn.Sequential, nn.ModuleList)):
                for i in range(len(mod) - 1, -1, -1):
                    if isinstance(mod[i], nn.Linear):
                        return mod, str(i), mod[i], f"{full_path}[{i}]"

    # ---- 第二步：unwrap 常见 wrapper，递归查找 ----
    for attr in ["hf_model", "model", "backbone", "net"]:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if isinstance(inner, nn.Module) and inner is not model:
                res = _find_classifier_linear(inner, _prefix=f"{_prefix}.{attr}")
                if res is not None:
                    return res

    # ---- 第三步：兜底 ----
    last = None
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                path = f"{_prefix}.{parent_name}.{child_name}" if parent_name else f"{_prefix}.{child_name}"
                last = (parent, child_name, child, path)

    return last


def _resolve_head_by_path(model: nn.Module, dotted_path: str):
    """
    根据点分路径（如 "hf_model.cls_classifier"）定位到
    (parent_module, attr_name, linear_layer)
    """
    parts = dotted_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr_name = parts[-1]
    linear = getattr(parent, attr_name)
    return parent, attr_name, linear


def _replace_head_1k_to_100(parent, attr_name, head, idx_1k, full_path=""):
    """将单个 Linear head 从 1k 切成 100 并替换回 parent。"""
    if not isinstance(head, nn.Linear):
        raise RuntimeError(f"Found classifier is not nn.Linear: {type(head)} at {full_path}")

    in_dim = head.in_features
    print(f"  替换分类头: {full_path}  shape: {head.weight.shape} -> [{100}, {in_dim}]")

    with torch.no_grad():
        new_w = head.weight.detach().clone()[idx_1k, :]
        new_b = head.bias.detach().clone()[idx_1k] if head.bias is not None else None

    new_head = nn.Linear(in_dim, 100, bias=(new_b is not None))
    with torch.no_grad():
        new_head.weight.copy_(new_w)
        if new_b is not None:
            new_head.bias.copy_(new_b)

    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and attr_name.isdigit():
        parent[int(attr_name)] = new_head
    else:
        setattr(parent, attr_name, new_head)


def get_classifier_heads(model_name: str) -> list:
    """
    根据模型名称返回需要替换的分类头路径列表。
    返回 None 表示走自动查找逻辑。
    """
    HEAD_MAP = {
        "IM100Test-deit_base_patch16_224": [
            "hf_model.classifier",
        ],
    }
    return HEAD_MAP.get(model_name, None)


def convert_model_to_im100(
    model: nn.Module,
    label_to_1k: dict = None,
    strict_1000: bool = True,
    model_name: str = None,        # 新增：外部传入模型名
    head_paths: list = None,       # 新增：直接指定路径列表
) -> nn.Module:
    """
    将 ImageNet-1k 分类头替换为 ImageNet-100 分类头。

    优先级：head_paths > model_name 查表 > 自动查找
    """
    if label_to_1k is None:
        from .LABEL_TO_1K import LABEL_TO_1K
        label_to_1k = LABEL_TO_1K
    if len(label_to_1k) != 100:
        raise ValueError(f"label_to_1k should contain 100 entries, got {len(label_to_1k)}")

    keys = sorted(label_to_1k.keys())
    if keys != list(range(100)):
        raise ValueError(f"label_to_1k keys should be 0..99")

    idx_1k = torch.tensor([label_to_1k[i] for i in range(100)], dtype=torch.long)

    # ---- 确定要替换的分类头 ----
    if head_paths is None and model_name is not None:
        head_paths = get_classifier_heads(model_name)

    if head_paths is not None:
        # 外部指定路径
        for path in head_paths:
            parent, attr_name, head = _resolve_head_by_path(model, path)
            if strict_1000 and head.out_features != 1000:
                raise ValueError(f"{path}: expected out_features=1000, got {head.out_features}")
            _replace_head_1k_to_100(parent, attr_name, head, idx_1k, full_path=path)
    else:
        # 自动查找
        found = _find_classifier_linear(model)
        if found is None:
            raise RuntimeError("Cannot find a classifier nn.Linear layer in the given model.")
        parent, attr_name, head, full_path = found
        if strict_1000 and head.out_features != 1000:
            raise ValueError(f"Expected out_features=1000, got {head.out_features}")
        _replace_head_1k_to_100(parent, attr_name, head, idx_1k, full_path=full_path)



    # ---- 更新 meta 字段 ----
    if hasattr(model, "num_classes"):
        try:
            model.num_classes = 100
        except Exception:
            pass

    if hasattr(model, "config") and getattr(model, "config") is not None:
        cfg = model.config
        if hasattr(cfg, "num_labels"):
            cfg.num_labels = 100
        if hasattr(cfg, "id2label"):
            cfg.id2label = {i: f"im100_{i}" for i in range(100)}
        if hasattr(cfg, "label2id"):
            cfg.label2id = {f"im100_{i}": i for i in range(100)}

    return model