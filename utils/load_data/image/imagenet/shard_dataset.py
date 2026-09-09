from pathlib import Path
import torch
from tqdm import tqdm


def materialize_loader_to_shards(
    dataloader,
    save_dir,
    shard_size=4096,
    save_dtype=torch.float16,
):
    """
    Materialize dataloader outputs (images, targets) into shard .pt files.

    images are assumed to have been fully transformed (resize/crop/normalize).
    We convert images to save_dtype (e.g., float16) before saving.

    Files created under save_dir:
      - shard_00000.pt, shard_00001.pt, ...
      - index.pt: list[(shard_id, offset)]
      - meta.pt: basic metadata
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_path = save_dir / "meta.pt"
    index_path = save_dir / "index.pt"

    cur_imgs = []
    cur_tgts = []
    shard_id = 0
    index = []

    def flush():
        nonlocal shard_id, cur_imgs, cur_tgts, index
        if not cur_imgs:
            return

        imgs = torch.cat(cur_imgs, dim=0)        # [N, C, H, W]
        tgts = torch.cat(cur_tgts, dim=0).long() # [N]

        shard_path = save_dir / f"shard_{shard_id:05d}.pt"
        torch.save({"images": imgs, "targets": tgts}, shard_path)

        n = imgs.size(0)
        for i in range(n):
            index.append((shard_id, i))

        shard_id += 1
        cur_imgs, cur_tgts = [], []

    total = 0
    for batch in tqdm(dataloader, desc="Materializing shards"):
        # Most timm loaders yield (images, targets)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        else:
            raise TypeError(f"Unexpected batch type from dataloader: {type(batch)}")

        images = images.detach()
        targets = targets.detach()

        # Move to CPU before saving (avoid serializing CUDA tensors)
        if images.is_cuda:
            images = images.cpu()
        if targets.is_cuda:
            targets = targets.cpu()

        # Convert dtype ONCE here (your requirement)
        if save_dtype is not None:
            images = images.to(save_dtype)

        cur_imgs.append(images)
        cur_tgts.append(targets)
        total += images.size(0)

        cur_count = sum(x.size(0) for x in cur_imgs)
        if cur_count >= shard_size:
            flush()

    flush()

    torch.save(index, index_path)
    torch.save(
        {
            "total": len(index),
            "shard_size": shard_size,
            "save_dtype": str(save_dtype),
        },
        meta_path
    )

    print(f"[OK] Materialized {len(index)} samples into {shard_id} shards at: {save_dir}")
    print(f"     meta: {meta_path}")
    print(f"     index: {index_path}")

