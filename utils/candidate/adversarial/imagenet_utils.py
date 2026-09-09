import os
import torch
from timm.data import create_dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms

class ImageFolderWithPath(ImageFolder):
    """ImageFolder that returns (image, label, path)."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path


def collate_with_path(batch):
    """Return images, labels, paths."""
    imgs, labels, paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(labels, dtype=torch.long),
        list(paths),
    )


def save_adv_batch_imagenet_style(
    inputs_adv: torch.Tensor,   # (B, C, H, W)
    paths: list,                # len=B, original file paths
    out_root: str,              # e.g., .../adversarial/imagenet_100/<model>/<attack>/val
    skip_if_exists: bool = True,
):
    os.makedirs(out_root, exist_ok=True)

    # 保存前确保在可保存范围（你 wrapper 已处理 domain，这里仅做保险）
    inputs_adv = inputs_adv.detach().clamp(0.0, 1.0).cpu()

    print(f"===> Saving images to: "
          f"\n   {out_root}")

    for i, src_path in enumerate(paths):
        class_folder = os.path.basename(os.path.dirname(src_path))  # n01440764
        filename = os.path.basename(src_path)                       # xxx.JPEG

        class_dir = os.path.join(out_root, class_folder)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, filename)
        if skip_if_exists and os.path.exists(save_path):
            continue

        # torchvision 保存（自动根据扩展名选择格式）
        save_image(inputs_adv[i], save_path)

import numpy as np
from PIL import Image

def create_temp_dataloader(
    path_to_data: str,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    resize_size: int = 256,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Build a temporary dataloader for ImageNet-style folder dataset (val only),
    returning (inputs, labels, paths). Inputs are normalized.
    """
    val_root = os.path.join(path_to_data, dataset_name, "raw", "val")
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f"val_root not found: {val_root}")

    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),  # <-- 这里执行 Normalize
    ])

    dataset = ImageFolderWithPath(root=val_root, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_with_path,
        drop_last=False,
    )

    return dataloader

def save_adv_batch_imagenet_style_pil(
    inputs_adv: torch.Tensor,   # (B, C, H, W), assumed in [0,1]
    paths: list,                # len=B, original file paths
    out_root: str,              # e.g., .../adversarial/imagenet_100/<model>/<attack>/val
    skip_if_exists: bool = True,
):
    os.makedirs(out_root, exist_ok=True)

    # 只做保险性 clamp，不做任何 normalize / denormalize
    inputs_adv = inputs_adv.detach().clamp(0.0, 1.0).cpu()

    print(f"===> Saving images to:\n   {out_root}")

    for i, src_path in enumerate(paths):
        class_folder = os.path.basename(os.path.dirname(src_path))  # e.g., n01440764
        filename = os.path.basename(src_path)                       # e.g., xxx.JPEG

        class_dir = os.path.join(out_root, class_folder)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, filename)

        img = inputs_adv[i].permute(1, 2, 0).numpy()
        img = (img * 255.0).round().astype(np.uint8)

        if isinstance(img, Image.Image):
            print(f"ERROR!!")
            img = np.array(img)

        pil_img = Image.fromarray(img, mode="RGB")
        pil_img.save(
            save_path,
            format="JPEG",
            quality=70,
            subsampling=0,
            optimize=True,
            progressive=False,
        )


