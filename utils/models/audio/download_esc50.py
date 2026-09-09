"""Materialize ESC-50 as local wav files + meta/esc50.csv from the HF dataset (reliable in CN via mirror).

The karolpiczak/ESC-50 GitHub zip is large and flaky through CN proxies; this pulls the SAME data from the
ashraq/esc50 HF dataset (run with HF_ENDPOINT=https://hf-mirror.com) and writes the standard layout that
probe_esc50_fold.py / utils.load_data.audio expect:
    {out}/audio/<filename>.wav    {out}/meta/esc50.csv  (filename, fold, target, category, esc10, src_file, take)

  HF_ENDPOINT=https://hf-mirror.com python utils/models/audio/download_esc50.py --out ../data/datasets/esc50

Needs: datasets, soundfile.
"""
import os
import csv
import argparse

import soundfile as sf
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='../data/datasets/esc50')
    args = ap.parse_args()

    audio_dir = os.path.join(args.out, 'audio')
    meta_dir = os.path.join(args.out, 'meta')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    ds = load_dataset('ashraq/esc50', split='train')  # via HF_ENDPOINT mirror
    cols = ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
    with open(os.path.join(meta_dir, 'esc50.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for ex in ds:
            audio = ex['audio']
            sf.write(os.path.join(audio_dir, ex['filename']), audio['array'], audio['sampling_rate'])
            w.writerow({c: ex.get(c, '') for c in cols})
    print(f"Wrote {len(ds)} wavs to {audio_dir} and meta to {meta_dir}/esc50.csv")


if __name__ == '__main__':
    main()
