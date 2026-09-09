import os
from pathlib import Path

import torch


def _set_hf_env(data_dir=None):
    """
    Point HF Hub at the mirror before any download. HF_HOME (on-disk teacher cache) only when data_dir is given.
    """
    if data_dir is not None:
        os.environ.setdefault('HF_HOME', os.path.abspath(os.path.join(data_dir, 'models', 'teacher', 'hf')))
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')  # mirror doesn't proxy HF Xet backend -> force classic LFS download
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')  # default 10s is too short for the mirror


def _load_or_cache(args, label, teacher_name, extract_fn):
    """Return cached teacher features for (teacher, dataset, label), else run extract_fn(), cache, return.

    Cache is keyed by teacher+dataset (student-independent): temp/feature_teacher/<teacher>/<dataset>/<label>.pt.
    """
    save_dir = Path(args.data_dir)/"temp"/"feature_teacher"/teacher_name/args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir/f'{label}.pt'
    if save_path.exists():
        print(f"Teacher feature is loaded from: {save_path}")
        return torch.load(save_path)
    feats = extract_fn()
    torch.save(feats, save_path, pickle_protocol=5)
    print(f"Teacher feature is saved to: {save_path}")
    return feats


def load_teacher_features(args, label, teacher_name, loader, device):
    """
    Route a teacher name to its modality extractor.
    """
    from utils.remap.teacher_extractor.image import IMAGE_TEACHER_MODELS, load_image_teacher_features
    from utils.remap.teacher_extractor.pointcloud import POINT_TEACHER_MODELS, load_point_teacher_features
    from utils.remap.teacher_extractor.audio import AUDIO_TEACHER_MODELS, load_audio_teacher_features
    if teacher_name in IMAGE_TEACHER_MODELS:
        return load_image_teacher_features(args, label, teacher_name, loader, device)
    if teacher_name in POINT_TEACHER_MODELS:
        return load_point_teacher_features(args, label, teacher_name, loader, device)
    if teacher_name in AUDIO_TEACHER_MODELS:
        return load_audio_teacher_features(args, label, teacher_name, loader, device)
    known = list(IMAGE_TEACHER_MODELS) + list(POINT_TEACHER_MODELS) + list(AUDIO_TEACHER_MODELS)
    raise ValueError(f"Unknown teacher {teacher_name}. Known: {known}")
