"""Standalone probe: detect a PUBLIC ESC-50 checkpoint's UNSEEN fold + verify/repair label alignment.

Pure-public students have undocumented splits AND often a class-id order != ESC-50's `target` order (raw
accuracy then ~0.02 = 1/50 on every fold even on trained folds). This runs the public HF audio-classifier
on each ESC-50 fold and prints BOTH raw and NAME-ALIGNED per-fold accuracy. Alignment = model output id ->
ESC-50 target, matched by category name via model.config.id2label. After alignment, trained folds jump to
~0.99 and the held-out fold is clearly lower (= the clean candidate set). Degenerate cases this catches:
num_labels != 50 (wrong head, e.g. a 527-class AudioSet head -> unusable); name-match count << 50 (labels
are LABEL_x / different names -> can't align by name); all folds high even after alignment -> trained on all.

  HF_ENDPOINT=https://hf-mirror.com python utils/models/audio/probe_esc50_fold.py \
      --repo cj94/hubert-esc50-finetuned-v2 --data_dir ../data/datasets/esc50

Needs: librosa (+ soundfile), transformers, torch.
"""
import os
import re
import csv
import argparse

import torch
import librosa
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


def _norm(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).strip().lower()).strip('_')  # canonicalize a class name


def load_rows(data_dir):
    rows, cat2target = [], {}
    with open(os.path.join(data_dir, 'meta', 'esc50.csv'), newline='') as f:
        for r in csv.DictReader(f):
            rows.append((r['filename'], int(r['fold']), int(r['target'])))
            cat2target[_norm(r['category'])] = int(r['target'])  # ESC-50 fixed name -> target id
    return rows, cat2target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)  # HF repo id (via mirror) or a local snapshot dir
    ap.add_argument('--data_dir', default='../data/datasets/esc50')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    fe = AutoFeatureExtractor.from_pretrained(args.repo)
    model = AutoModelForAudioClassification.from_pretrained(args.repo).eval().to(device)
    sr = fe.sampling_rate
    rows, cat2target = load_rows(args.data_dir)
    audio_dir = os.path.join(args.data_dir, 'audio')

    id2label = model.config.id2label
    print(f"model num_labels = {model.config.num_labels}; feature-extractor sr = {sr}")
    print("id2label sample:", {k: id2label[k] for k in list(id2label)[:5]})

    # Build model-output-id -> ESC-50 target via category NAME (repairs a permuted/relabeled head)
    id2target = {}
    for mid, name in id2label.items():
        t = cat2target.get(_norm(name))
        if t is not None:
            id2target[int(mid)] = t
    print(f"name-matched labels: {len(id2target)}/{model.config.num_labels}  (50 = perfect ESC-50 alignment)")

    raw = {f: [0, 0] for f in range(1, 6)}
    rmp = {f: [0, 0] for f in range(1, 6)}
    with torch.no_grad():
        for filename, fold, target in rows:
            wav, _ = librosa.load(os.path.join(audio_dir, filename), sr=sr, mono=True)
            inputs = fe(wav, sampling_rate=sr, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pred = model(**inputs).logits.argmax(-1).item()
            raw[fold][1] += 1
            raw[fold][0] += int(pred == target)
            rmp[fold][1] += 1
            rmp[fold][0] += int(id2target.get(pred, -1) == target)

    print("\nfold   raw_acc   aligned_acc   errors/total(aligned)")
    for f in range(1, 6):
        cr, t = raw[f]
        cm, _ = rmp[f]
        print(f"  {f}    {cr / t:.4f}    {cm / t:.4f}      {t - cm}/{t}")
    accs = {f: rmp[f][0] / rmp[f][1] for f in rmp}
    clean = min(accs, key=accs.get)
    print(f"\n(aligned) likely UNSEEN clean fold = {clean}  (lowest acc {accs[clean]:.4f})")
    print("num_labels!=50 -> wrong head (unusable); name-matched<<50 -> can't align by name; "
          "aligned_acc high on ALL folds -> trained on everything")


if __name__ == '__main__':
    main()
