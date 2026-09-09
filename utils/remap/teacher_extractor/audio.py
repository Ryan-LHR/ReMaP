"""Frozen CLAP audio teacher for ReMaP (language-audio aligned -> independent reference geometry).

Mirrors pointcloud / text: cache per teacher+dataset, route from load_teacher_features. CLAP
(laion/clap-htsat-unfused) via transformers ClapModel.get_audio_features on 48 kHz mono waveforms (the rate
the loader serves). Features only, never fine-tuned.
"""
import numpy as np
import torch
from tqdm import tqdm

from utils.remap.teacher_extractor.extractor import _set_hf_env, _load_or_cache

# Frozen audio teachers. CLAP = the audio analog of CLIP (contrastive language-audio), so it is to an
# environmental-sound student what Uni3D/CLIP are to 3D/image students: a genuinely independent geometry.
AUDIO_TEACHER_MODELS = {
    'clap': dict(hf_repo='laion/clap-htsat-unfused',
                 revision='8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a', sr=48000),
}
_AUDIO_TEACHER_CACHE = {}


def _load_clap(spec, device):
    key = ('clap', spec['hf_repo'])
    if key in _AUDIO_TEACHER_CACHE:
        return _AUDIO_TEACHER_CACHE[key]
    _set_hf_env()
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(spec['hf_repo'], revision=spec['revision']).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    processor = ClapProcessor.from_pretrained(spec['hf_repo'], revision=spec['revision'])
    _AUDIO_TEACHER_CACHE[key] = (model, processor)
    return model, processor


@torch.no_grad()
def extract_audio_teacher_features(loader, teacher_name, device, micro_bs=16):
    """Extract frozen CLAP audio embeddings for every clip in the loader, preserving order."""
    spec = AUDIO_TEACHER_MODELS[teacher_name]
    model, processor = _load_clap(spec, device)
    feats = []
    for batch in tqdm(loader, total=len(loader), desc=f"Extracting audio teacher [{teacher_name}]"):
        wav = batch[0].float()  # [B, T] @ 48 kHz
        for i in range(0, wav.shape[0], micro_bs):
            chunk = wav[i:i + micro_bs]
            inputs = processor(audios=[w.cpu().numpy() for w in chunk], sampling_rate=spec['sr'],
                               return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = model.get_audio_features(**inputs)
            feats.append(emb.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def load_audio_teacher_features(args, label, teacher_name, loader, device):
    """Load cached CLAP features or extract and cache them (keyed by teacher+dataset)."""
    return _load_or_cache(args, label, teacher_name,
                          lambda: extract_audio_teacher_features(loader, teacher_name, device))
