"""Frozen 3D point-cloud FM teachers for ReMaP (Uni3D). Features only, never fine-tuned.

Design mirrors extractor.py / text.py: cache per teacher+dataset, route from
load_teacher_features. The Uni3D encoder code is VENDORED (not a full clone): drop Uni3D's two model files
into utils/models/fm/uni3d/ (point_encoder.py, uni3d.py) — see the import in _load_uni3d. pointnet2_ops
(CUDA FPS) is avoided via the pure-torch farthest_point_sample below (monkeypatched into the vendored module).

Server prep:
  1. weights:  huggingface BAAI/Uni3D -> modelzoo/uni3d-b/model.pt  (download via hf-mirror)
  2. vendor:   utils/models/fm/uni3d/{point_encoder.py, uni3d.py, __init__.py}  (from baaivision/Uni3D)
"""
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from utils.remap.teacher_extractor.extractor import _set_hf_env, _load_or_cache

# Frozen point-cloud teachers. Uni3D-B config from baaivision/Uni3D scripts/inference.sh.
POINT_TEACHER_MODELS = {
    'uni3d': dict(
        hf_repo='BAAI/Uni3D', ckpt='modelzoo/uni3d-b/model.pt',
        revision='3d8233b76aa350d72f6213ecd2123c2026b42355',
        pc_model='eva02_base_patch14_448', pc_feat_dim=768, pc_encoder_dim=512,
        embed_dim=1024, num_group=512, group_size=64, npoints=10000,
    ),
}

_POINT_TEACHER_CACHE = {}


def _normalize_to_unit_ball(pc):
    """Center at centroid + scale into the unit ball (Uni3D's expected normalization). pc [B,N,3]."""
    pc = pc - pc.mean(dim=1, keepdim=True)
    scale = pc.norm(dim=2).max(dim=1)[0].clamp(min=1e-6).view(-1, 1, 1)
    return pc / scale


def _to_xyzrgb(xyz, rgb_const=0.4):
    """xyz [B,N,3] -> [B,N,6] with constant gray RGB (Uni3D expects xyz+rgb; ModelNet40 has no color)."""
    rgb = torch.full_like(xyz, rgb_const)
    return torch.cat([xyz, rgb], dim=2)


def _load_uni3d(spec, data_dir, device):
    """Build the vendored Uni3D model, load the HF checkpoint, freeze. Returns the model (has .encode_pc)."""
    key = ('uni3d', spec['ckpt'])
    if key in _POINT_TEACHER_CACHE:
        return _POINT_TEACHER_CACHE[key]

    _set_hf_env(data_dir)

    try:
        from utils.models.fm.uni3d.uni3d import create_uni3d
    except ImportError as e:
        raise ImportError(
            "Uni3D code not vendored. Put baaivision/Uni3D's models/{point_encoder,uni3d}.py "
            "into utils/models/fm/uni3d/ (+ an empty __init__.py). Original error: " + str(e))

    args = SimpleNamespace(
        pc_model=spec['pc_model'], pretrained_pc='', drop_path_rate=0.0,
        pc_feat_dim=spec['pc_feat_dim'], pc_encoder_dim=spec['pc_encoder_dim'],
        embed_dim=spec['embed_dim'], group_size=spec['group_size'], num_group=spec['num_group'],
        patch_dropout=0.0,
    )
    model = create_uni3d(args)

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=spec['hf_repo'], filename=spec['ckpt'],
                                revision=spec['revision'])
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd.get('module', sd.get('state_dict', sd))
    sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Uni3D loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    _POINT_TEACHER_CACHE[key] = model
    return model


@torch.no_grad()
def extract_point_teacher_features(loader, teacher_name, data_dir, device, micro_bs=16):
    """Extract frozen Uni3D pc embeddings for every point cloud in the loader, preserving order."""
    spec = POINT_TEACHER_MODELS[teacher_name]
    model = _load_uni3d(spec, data_dir, device)
    feats = []
    for batch in tqdm(loader, total=len(loader), desc=f"Extracting point teacher [{teacher_name}]"):
        raw = batch[0].to(device).float()  # [B, N, 3]
        for i in range(0, raw.shape[0], micro_bs):
            xyz = _normalize_to_unit_ball(raw[i:i + micro_bs])
            pc = _to_xyzrgb(xyz)  # [b, N, 6]
            emb = model.encode_pc(pc)
            feats.append(emb.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def load_point_teacher_features(args, label, teacher_name, loader, device):
    """Load cached point-cloud teacher features or extract and cache them (keyed by teacher+dataset)."""
    return _load_or_cache(args, label, teacher_name,
                          lambda: extract_point_teacher_features(loader, teacher_name, args.data_dir, device))
