"""Audio-modality dataset loader for ReMaP (CLAP teacher; HF spectrogram student).

Returns (train_set, test_set) Datasets of (waveform[ESC50_LEN] @ 48 kHz, label). 48 kHz is CLAP-native
(teacher fidelity first); students downsample internally. Split = fold 1 test / folds 3-5 reference (fold 2 dropped), exactly matching the public checkpoint
Adam-ousse/ast-esc50-finetuned-fold1 (train folds 3-5, val fold 2, fold 1 held out): the candidate is
genuinely unseen AND the kNN reference is exactly the folds the student was trained on -- the test fold MUST
equal the checkpoint's held-out fold or errors leak into training. Per-item lazy
wav decode via librosa (load + resample + mono in one call; no torchaudio). librosa imported lazily so
non-audio runs never require it. shuffle=False downstream so teacher features align with the student.
"""
import os
import csv

import torch
from torch.utils.data import Dataset

ESC50_SR = 48000  # CLAP-native sample rate
ESC50_SECONDS = 5
ESC50_LEN = ESC50_SR * ESC50_SECONDS  # fixed clip length in samples
# held-out fold = candidate set; MUST equal the checkpoint's held-out fold (Adam-ousse/ast-esc50-finetuned-fold1
# holds out fold 1, trains on folds 3-5, val fold 2) or the candidate leaks into the student's training data
ESC50_TEST_FOLD = '1'
# the checkpoint's validation fold -- excluded from the kNN reference so the reference is EXACTLY the folds the
# student was trained on (3-5), per the card's documented protocol
ESC50_VAL_FOLD = '2'

class _ESC50Dataset(Dataset):
    def __init__(self, rows, audio_dir):
        import librosa  # lazy: keep utils.load_data import working without librosa installed
        self._librosa = librosa
        self.rows = rows  # list of (filename, target)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        filename, target = self.rows[i]
        wav, _ = self._librosa.load(os.path.join(self.audio_dir, filename), sr=ESC50_SR, mono=True)  # [T] @ 48k
        wav = torch.from_numpy(wav)
        if wav.numel() < ESC50_LEN:
            wav = torch.nn.functional.pad(wav, (0, ESC50_LEN - wav.numel()))
        else:
            wav = wav[:ESC50_LEN]
        return wav, target


def load_audio_data(dataset_name, path_to_data):
    """Load an audio dataset as (train_set, test_set) of (waveform, label)."""
    if dataset_name == 'esc50':
        root = os.path.join(path_to_data, 'esc50')
        audio_dir = os.path.join(root, 'audio')
        meta_path = os.path.join(root, 'meta', 'esc50.csv')
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"ESC-50 meta not found at {meta_path} (expected karoldvl/ESC-50 layout).")
        train_rows, test_rows = [], []
        with open(meta_path, newline='') as f:
            for r in csv.DictReader(f):
                row = (r['filename'], int(r['target']))
                if r['fold'] == ESC50_TEST_FOLD:
                    test_rows.append(row)
                elif r['fold'] != ESC50_VAL_FOLD:  # drop the checkpoint's val fold from the kNN reference
                    train_rows.append(row)
        return _ESC50Dataset(train_rows, audio_dir), _ESC50Dataset(test_rows, audio_dir)

    raise ValueError(f"Unknown audio dataset: {dataset_name}")
