import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.remap.teacher_extractor.extractor import _set_hf_env, _load_or_cache

# Frozen image FM teachers: name -> dict(timm_id, img_size override or None for the model's native size).
# Features only, never fine-tuned. Add a line here to try a new teacher (the framework caches per name).
IMAGE_TEACHER_MODELS = {
    'dinov2_vitb14': dict(timm_id='vit_base_patch14_dinov2.lvd142m',
                          revision='4685c99dabffe5affac90bd99dbffd25801ae58d', img_size=224),
    'dinov2_vitl14': dict(timm_id='vit_large_patch14_dinov2.lvd142m',
                          revision='4741e1cafbf45415e77074bb0cb42dba76c8684a', img_size=224),
    'clip_vitl14': dict(timm_id='vit_large_patch14_clip_224.laion2b',
                        revision='ce143c10d3592fce8cfabbee2b990656c2aaead6', img_size=None),
    'siglip_so400m': dict(timm_id='vit_so400m_patch14_siglip_224.webli',
                          revision='f98276b381081f06e4a3d64b62f25202ebff0978', img_size=None),
    'sup_vitl16': dict(timm_id='vit_large_patch16_224.augreg_in21k_ft_in1k',
                       revision='0930ab3308b84cb2ae091a4a80703c459412a4c7', img_size=None),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRENORM_DATASETS = {'imagenet_100'}  # loader already yields ImageNet-normalized tensors

_IMAGE_TEACHER_CACHE = {}


def load_image_teacher(teacher_name, data_dir, device):
    """
    Load a frozen timm image FM teacher;
    return (model, data_cfg) with its own input size/mean/std.
    Cached per name.
    """
    if teacher_name in _IMAGE_TEACHER_CACHE:
        return _IMAGE_TEACHER_CACHE[teacher_name]

    _set_hf_env(data_dir)
    import timm
    from timm.data import resolve_model_data_config

    spec = IMAGE_TEACHER_MODELS[teacher_name]
    hf_hub_id = f"timm/{spec['timm_id']}@{spec['revision']}"
    kwargs = dict(pretrained=True, num_classes=0, pretrained_cfg_overlay={'hf_hub_id': hf_hub_id})
    if spec.get('img_size'):
        kwargs['img_size'] = spec['img_size']
    model = timm.create_model(spec['timm_id'], **kwargs)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = resolve_model_data_config(model)
    size = spec['img_size'] if spec.get('img_size') else cfg['input_size'][-1]
    data_cfg = {'size': size, 'mean': cfg['mean'], 'std': cfg['std']}
    print(f"Teacher {teacher_name} ({spec['timm_id']}): size={size}, mean={cfg['mean']}, std={cfg['std']}")
    _IMAGE_TEACHER_CACHE[teacher_name] = (model, data_cfg)
    return model, data_cfg


def _to_unit_rgb(x, dataset, device):
    """
    Bring a student-format batch to [0,1] 3-channel RGB.
    """
    x = x.float()
    if dataset in PRENORM_DATASETS:  # ImageNet-normalized -> de-normalize back to [0,1]
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x = (x * std + mean).clamp(0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def _teacher_preprocess(x, dataset, data_cfg, device):
    """
    Map a [0,1]-RGB batch to a teacher's own input space (its native size + mean/std).
    """
    x = _to_unit_rgb(x, dataset, device)
    x = F.interpolate(x, size=data_cfg['size'], mode='bicubic', align_corners=False)
    mean = torch.tensor(data_cfg['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(data_cfg['std'], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_image_teacher_features(loader, teacher_name, dataset, data_dir, device, micro_bs=32):
    """
    Extract frozen image-teacher features for every sample in a student-format loader, preserving order.

    The teacher forward is micro-batched (micro_bs) so a large student batch does not OOM a high-res
    teacher (e.g. SigLIP@384). Features are identical to the full-batch path (same model, same order).
    """
    model, data_cfg = load_image_teacher(teacher_name, data_dir, device)
    feats = []
    for batch in tqdm(loader, total=len(loader), desc=f"Extracting teacher [{teacher_name}]"):
        raw = batch[0].to(device)
        for i in range(0, raw.shape[0], micro_bs):
            chunk = _teacher_preprocess(raw[i:i + micro_bs], dataset, data_cfg, device)
            feats.append(model(chunk).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def load_image_teacher_features(args, label, teacher_name, loader, device):
    """
    Load cached image-teacher features or extract and cache them, keyed by teacher+dataset (student-independent).
    """
    return _load_or_cache(args, label, teacher_name,
                          lambda: extract_image_teacher_features(loader, teacher_name, args.dataset, args.data_dir, device))
