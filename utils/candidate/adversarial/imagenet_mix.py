import os
import random
import shutil
from pathlib import Path
from typing import Optional, Literal


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPEG", ".JPG"}


def _list_images(folder: Path):
    if not folder.exists():
        return []
    files = []
    for p in folder.iterdir():
        if p.is_file() and (p.suffix.lower() in {e.lower() for e in IMG_EXTS}):
            files.append(p.name)
    return sorted(files)


def _safe_place_file(src: Path, dst: Path, mode: str = "copy"):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _build_mixed_imagenet_style(
    val_root: Path,
    adv_root: Path,
    out_root: Path,
    *,
    per_class: int = 50,
    seed: int = 42,
    copy_mode: Literal["copy", "symlink", "hardlink"] = "copy",
    verbose: bool = True,
):
    """
    Build a mixed dataset with 5:5 split per class from val_root and adv_root.

    Key constraints:
    - Class folders preserved (e.g., n01440764).
    - File names preserved.
    - For each class, choose (per_class/2) from val and (per_class/2) from adv.
    - The chosen filenames are disjoint between val and adv (mutually exclusive).
    - Sampling is done from the intersection of filenames existing in both val and adv,
      so "same-ID" correspondence is respected.
    """
    if per_class <= 0:
        raise ValueError("per_class must be > 0")
    if per_class % 2 != 0:
        raise ValueError("per_class must be even for 5:5 split")

    rng = random.Random(seed)
    out_root.mkdir(parents=True, exist_ok=True)

    val_classes = {p.name for p in val_root.iterdir() if p.is_dir()}
    adv_classes = {p.name for p in adv_root.iterdir() if p.is_dir()}
    classes = sorted(val_classes & adv_classes)

    if not classes:
        raise RuntimeError("No common class folders found between val_root and adv_root.")

    k_val = per_class // 2
    k_adv = per_class - k_val

    if verbose:
        print(f"[MIX] common classes: {len(classes)}")
        print(f"[MIX] val_root: {val_root}")
        print(f"[MIX] adv_root: {adv_root}")
        print(f"[MIX] out_root: {out_root}")
        print(f"[MIX] per_class={per_class}, seed={seed}, copy_mode={copy_mode}")

    for cls in classes:
        v_dir = val_root / cls
        a_dir = adv_root / cls
        o_dir = out_root / cls

        v_files = set(_list_images(v_dir))
        a_files = set(_list_images(a_dir))
        common = sorted(v_files & a_files)

        if len(common) < 2:
            if verbose:
                print(f"[SKIP] {cls}: common={len(common)} (too small)")
            continue

        # If not enough, downscale but keep 5:5 and disjointness
        cur_k_val, cur_k_adv = k_val, k_adv
        if len(common) < per_class:
            max_pairs = len(common) // 2
            cur_k_val = min(k_val, max_pairs)
            cur_k_adv = min(k_adv, max_pairs)
            # Ensure we still have disjoint sets
            cur_k_adv = min(cur_k_adv, len(common) - cur_k_val)

            if cur_k_val == 0 or cur_k_adv == 0:
                if verbose:
                    print(f"[SKIP] {cls}: common={len(common)} cannot form 5:5")
                continue

            if verbose:
                print(
                    f"[WARN] {cls}: common={len(common)} < per_class={per_class}. "
                    f"Downscale to val={cur_k_val}, adv={cur_k_adv}."
                )

        pool = common[:]
        rng.shuffle(pool)

        chosen_val = set(pool[:cur_k_val])
        remaining = pool[cur_k_val:]
        chosen_adv = set(remaining[:cur_k_adv])

        # Double-check disjointness
        if not chosen_val.isdisjoint(chosen_adv):
            raise RuntimeError(f"Unexpected overlap in {cls} (should never happen).")

        # Place files
        for fn in chosen_val:
            _safe_place_file(v_dir / fn, o_dir / fn, mode=copy_mode)

        for fn in chosen_adv:
            _safe_place_file(a_dir / fn, o_dir / fn, mode=copy_mode)

        if verbose:
            print(f"[OK] {cls}: val={len(chosen_val)}, adv={len(chosen_adv)}, total={len(chosen_val)+len(chosen_adv)}")


def make_mixed_im100_dataset(
    val_root: str,
    adv_root: str,
    out_root: str,
    seed: int,
    *,
    per_class: int = 50,
    copy_mode: Literal["copy", "symlink", "hardlink"] = "copy",
    verbose: bool = True,
):
    """
    One-shot wrapper (as requested):
    Just pass three paths + seed.

    Args:
        val_root: ImageNet100 val root (class folders).
        adv_root: adversarial val root (class folders).
        out_root: output root for mixed dataset.
        seed: random seed to make the split reproducible.
        per_class: total images per class in mixed set (default 50 -> 25+25).
        copy_mode: 'copy' (default), or 'symlink', or 'hardlink'.
        verbose: print progress logs.
    """
    _build_mixed_imagenet_style(
        Path(val_root),
        Path(adv_root),
        Path(out_root),
        per_class=per_class,
        seed=seed,
        copy_mode=copy_mode,
        verbose=verbose,
    )
