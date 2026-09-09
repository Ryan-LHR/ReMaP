import random
import os

import torch
import torch.nn as nn
from timm.data import create_dataset

from utils.datasets import CustomTensorDataset

try:
    import torchattacks
except Exception:
    None

DATASET_ATTACK_CFG = {
    "mnist": {
        "default_eps": 0.3,
        "default_alpha": 0.01,
        "default_steps": 40,

        # CW
        "cw_c": 1.0,
        "cw_kappa": 0.0,
        "cw_steps": 200,
        "cw_lr": 0.01,

        # DeepFool
        "deepfool_steps": 50,
        "deepfool_overshoot": 0.02,
    },
    "fashion_mnist": {
        "default_eps": 0.3,
        "default_alpha": 0.01,
        "default_steps": 40,

        # CW
        "cw_c": 1.0,
        "cw_kappa": 0.0,
        "cw_steps": 200,
        "cw_lr": 0.01,

        # DeepFool
        "deepfool_steps": 50,
        "deepfool_overshoot": 0.02,
    },
    "cifar_10": {
        "default_eps": 8 / 255,
        "default_alpha": 2 / 255,
        "default_steps": 10,

        "cw_c": 1.0,
        "cw_kappa": 0.0,
        "cw_steps": 300,
        "cw_lr": 0.01,

        "deepfool_steps": 50,
        "deepfool_overshoot": 0.02,
    },
    "svhn": {
        "default_eps": 8 / 255,
        "default_alpha": 2 / 255,
        "default_steps": 10,

        "cw_c": 1.0,
        "cw_kappa": 0.0,
        "cw_steps": 300,
        "cw_lr": 0.01,

        "deepfool_steps": 50,
        "deepfool_overshoot": 0.02,
    },
    "imagenet_100": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "default_eps": 4 / 255,
        "default_alpha": 1 / 255,
        "default_steps": 10,

        "cw_c": 1.0,
        "cw_kappa": 0.0,
        "cw_steps": 200,
        "cw_lr": 0.005,

        "deepfool_steps": 50,
        "deepfool_overshoot": 0.02,
    },
}

def _normalize_batch(x, mean, std):
    """
    Normalize a batch tensor by channel-wise mean and std.
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean_t) / std_t


def _denormalize_batch(x, mean, std):
    """
    Denormalize a batch tensor back to [0, 1] domain.
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return x * std_t + mean_t

class ModelWrapper(nn.Module):
    """
    Wrap a model to accept pixel-domain inputs and normalize internally.
    """
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std
        print(f"Wrap original model for normalization")

    def forward(self, x):
        "normalize inputs from [0,1] to target range"
        x = _normalize_batch(x, self.mean, self.std)
        return self.model(x)


def _build_attacker(model, dataset_name, type):
    """
    Build attacker instance.
    """
    cfg = DATASET_ATTACK_CFG[dataset_name]
    eps = cfg["default_eps"]
    alpha = cfg["default_alpha"]
    steps = cfg["default_steps"]

    c = cfg.get("cw_c", 1.0)
    kappa = cfg.get("cw_kappa", 0.0)
    cw_steps = cfg.get("cw_steps", 1000)
    lr = cfg.get("cw_lr", 0.01)

    df_steps = cfg.get("deepfool_steps", 50)
    overshoot = cfg.get("deepfool_overshoot", 0.02)

    if type == "fgsm":
        attacker = torchattacks.FGSM(model, eps=eps)
    elif type == "bim":
        attacker = torchattacks.BIM(model, eps=eps, alpha=alpha, steps=steps)
    elif type == "pgd":
        # attacker = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
        attacker = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=False)
    elif type == "cw":
        attacker = torchattacks.CW(model, c=c, kappa=kappa, steps=cw_steps, lr=lr)
    elif type == "deepfool":
        attacker = torchattacks.DeepFool(model, steps=df_steps, overshoot=overshoot)
    else:
        raise ValueError("Attack type not found!")
    return attacker


class AttackWrapper():
    def __init__(self, attacker, mean, std):
        super().__init__()
        self.attacker = attacker
        self.mean = mean
        self.std = std
        print(f"Wrap original attacker for denormalization")

    def __call__(self, inputs, labels):
        # denormalize to [0,1]
        inputs = _denormalize_batch(inputs, self.mean, self.std)
        inputs_adv = self.attacker(inputs, labels)
        # normalize to target range from [0,1]
        inputs_adv = _normalize_batch(inputs_adv, self.mean, self.std)
        return inputs_adv

class CompositeAttacker():
    """
    Attacker used in generate adversarial samples.
    Randomly selects one type.
    """

    def __init__(self, attack_types, mean, std, dataset_name, wrapped_model, seed):
        print(f"Current attack types contain: {attack_types}")
        self.attack_types = list(attack_types)
        self.rng = random.Random(seed)

        self._deck = []
        self._cursor = 0

        self.attacker_dict = {}
        for atk_type in attack_types:
            attacker = _build_attacker(wrapped_model, dataset_name, atk_type)
            if mean is not None and std is not None:
                attacker = AttackWrapper(attacker, mean, std)
            self.attacker_dict[atk_type] = attacker

    def _reshuffle_deck(self):
        self._deck = self.attack_types[:]
        self.rng.shuffle(self._deck)
        self._cursor = 0

    def __call__(self, inputs, labels):
        # atk_type = self.rng.choice(self.attack_types)
        if self._cursor >= len(self._deck):
            self._reshuffle_deck()
        atk_type = self._deck[self._cursor]
        self._cursor += 1
        print(f"\ncurrent attack type is {atk_type}")

        attacker = self.attacker_dict[atk_type]
        inputs_adv = attacker(inputs, labels)
        return inputs_adv


def save_adversarial_tensor_dataset(dataset_name, model_name, inputs_tensor, labels_tensor, path_to_data=None):
    """
    
    """
    adv_tensor_path = os.path.join(path_to_data, f"{dataset_name}/adv", f"{model_name}_adv_cand_set_tensor.pt")
    save_dir = os.path.dirname(adv_tensor_path)
    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir, exist_ok=True)
    torch.save((inputs_tensor.cpu(), labels_tensor.cpu()), adv_tensor_path)
    print(f"===> Saving processed Tensors to: "
          f"\n   {adv_tensor_path}")


def load_adversarial_tensor_dataset(dataset_name, model_name, path_to_data):
    """

    """
    adv_tensor_path = os.path.join(path_to_data, f"{dataset_name}/adv",
                                   f"{model_name}_adv_cand_set_tensor.pt")
    print(f"===> Loading processed Tensors from: "
          f"\n   {adv_tensor_path}")
    inputs_t, labels_t = torch.load(adv_tensor_path)
    dataset = CustomTensorDataset(inputs_t, labels_t)

    return dataset
