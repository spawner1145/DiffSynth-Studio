import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
import imagesize
from PIL import Image
from tqdm import tqdm
import glob

from diffsynth.core.data import UnifiedDataset, ImageTextPairDataset
from diffsynth.core.data.unified_dataset import BucketManager

"""分桶索引构建脚本（适用于超大数据集预处理）

用法概览：

1) 为 UnifiedDataset 的 metadata（json/jsonl/csv）构建预分桶索引：

    python tools/build_bucket_index.py unified \
        --base_path /root/autodl-tmp/DiffSynth-Studio/edit/images \
        --metadata_path /root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl \
        --output /root/autodl-tmp/DiffSynth-Studio/edit/prebucket_index.jsonl \
        --bucket_data_key image \
        --max_bucket_reso 1024 \
        --min_bucket_reso 256 \
        --bucket_reso_steps 64 \
        --bucket_base_reso 256 256

    生成的 prebucket_index.jsonl 每行格式类似：
        {"data_id": 0, "bucket": [1024, 576]}
    其中 data_id 为 metadata 中的行号（0-based）。

    在训练脚本中使用：

        dataset = UnifiedDataset(
            base_path=..., metadata_path=..., ...,
            enable_bucket=True,
            bucket_index_path="/root/autodl-tmp/DiffSynth-Studio/edit/prebucket_index.jsonl",
        )

2) 为 ImageTextPairDataset 的目录构建预分桶索引：

    python tools/build_bucket_index.py pairs \
        --data_dir /root/autodl-tmp/DiffSynth-Studio/edit/images \
        --output /root/autodl-tmp/DiffSynth-Studio/edit/prebucket_pairs_index.jsonl \
        --max_bucket_reso 1024 \
        --min_bucket_reso 256 \
        --bucket_reso_steps 64 \
        --bucket_base_reso 256 256

    生成的 prebucket_pairs_index.jsonl 每行格式：
        {"data_id": 0, "bucket": [1024, 576]}
    其中 data_id 为 ImageTextPairDataset._scan_pairs() 得到的 pairs 列表下标（按文件名排序）。

    在训练脚本中使用：

        dataset = ImageTextPairDataset(
            data_dir=..., ...,
            enable_bucket=True,
            bucket_index_path="/root/autodl-tmp/DiffSynth-Studio/edit/prebucket_pairs_index.jsonl",
        )

注意：
    - 脚本内部使用 BucketManager，与 DiffSynth 训练代码的分桶逻辑保持一致；
    - max_bucket_reso / min_bucket_reso / bucket_reso_steps / bucket_base_reso
      等参数需与训练脚本保持一致，否则训练时的桶尺寸与预处理结果会不匹配；
    - 适合上千万级样本的预处理，将分桶这一步前移到离线阶段，训练启动时只需读取索引文件。
"""


def build_bucket_manager(max_bucket_reso, min_bucket_reso, bucket_reso_steps, bucket_base_reso=None, bucket_no_upscale=False):
    """Utility to build a BucketManager consistent with training settings."""
    if isinstance(bucket_base_reso, (tuple, list)) and len(bucket_base_reso) == 2:
        max_reso = (int(bucket_base_reso[0]), int(bucket_base_reso[1]))
    elif isinstance(max_bucket_reso, (tuple, list)) and len(max_bucket_reso) == 2:
        max_reso = (int(max_bucket_reso[0]), int(max_bucket_reso[1]))
    else:
        base_side = int(max_bucket_reso)
        max_reso = (base_side, base_side)

    if isinstance(max_bucket_reso, (tuple, list)) and len(max_bucket_reso) == 2:
        max_bucket_reso_limit = int(max(max_bucket_reso))
    else:
        max_bucket_reso_limit = int(max_bucket_reso)

    min_bucket_reso = int(min_bucket_reso)
    bucket_reso_steps = int(bucket_reso_steps)

    if min_bucket_reso % bucket_reso_steps != 0:
        min_bucket_reso = max(bucket_reso_steps, min_bucket_reso - min_bucket_reso % bucket_reso_steps)
    if max_bucket_reso_limit % bucket_reso_steps != 0:
        max_bucket_reso_limit = max_bucket_reso_limit + bucket_reso_steps - max_bucket_reso_limit % bucket_reso_steps
    if max_bucket_reso_limit < bucket_reso_steps:
        max_bucket_reso_limit = bucket_reso_steps

    if min(max_reso) < min_bucket_reso:
        raise ValueError("min_bucket_reso must be <= min(resolution)")
    if max(max_reso) > max_bucket_reso_limit:
        raise ValueError("max_bucket_reso must be >= max(resolution)")

    bm = BucketManager(
        no_upscale=bucket_no_upscale,
        max_reso=max_reso,
        min_size=min_bucket_reso,
        max_size=max_bucket_reso_limit,
        reso_steps=bucket_reso_steps,
    )
    if not bucket_no_upscale:
        bm.make_buckets()
    return bm


def iter_unified_metadata(metadata_path):
    """Stream metadata (json / jsonl / csv) and yield (index, record).

    This mirrors UnifiedDataset.load_metadata, but avoids loading everything
    into memory so that we can build a bucket index for very large datasets.
    """
    if metadata_path.endswith(".jsonl"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield idx, json.loads(line)
    elif metadata_path.endswith(".json"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for idx, item in enumerate(data):
            yield idx, item
    else:
        import pandas as pd
        df = pd.read_csv(metadata_path)
        for idx in range(len(df)):
            yield idx, df.iloc[idx].to_dict()


def resolve_path(base_path, p):
    if os.path.isabs(p):
        return p
    return os.path.join(base_path, p)


def probe_image_size(path):
    width, height = imagesize.get(path)
    if width > 0 and height > 0:
        return width, height
    with Image.open(path) as img:
        return img.size


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# 全局缓存 BucketManager，避免在子进程中重复初始化
_worker_bm_cache = None

def get_cached_bm(bucket_args):
    global _worker_bm_cache
    if _worker_bm_cache is None:
        _worker_bm_cache = build_bucket_manager(**bucket_args)
    return _worker_bm_cache


def _probe_unified_bucket_task(task):
    data_id, path, bucket_args = task
    if path is None or not os.path.exists(path):
        return None
    try:
        w, h = probe_image_size(path)
        bm = get_cached_bm(bucket_args)
        reso, _, _ = bm.select_bucket(w, h)
        return {"data_id": int(data_id), "bucket": [int(reso[0]), int(reso[1])]}
    except Exception:
        return None


def _probe_pairs_bucket_task(task):
    data_id, image_path, bucket_args = task
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        w, h = probe_image_size(image_path)
        bm = get_cached_bm(bucket_args)
        reso, _, _ = bm.select_bucket(w, h)
        return {"data_id": int(data_id), "bucket": [int(reso[0]), int(reso[1])]}
    except Exception:
        return None


def _iterate_results(tasks, worker_fn, num_workers, total=None, desc=None):
    if int(num_workers) <= 1:
        for task in tqdm(tasks, total=total, desc=desc):
            yield worker_fn(task)
        return

    # 对于 700w+ 级别的数据，必须使用较大的 chunksize 减少 IPC 开销
    chunksize = 2000 if total and total > 100000 else 100
    with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        for result in tqdm(executor.map(worker_fn, tasks, chunksize=chunksize), total=total, desc=desc):
            yield result


def build_bucket_index_for_unified(base_path, metadata_path, output_path,
                                   bucket_data_key="image",
                                   max_bucket_reso=1024,
                                   min_bucket_reso=256,
                                   bucket_reso_steps=64,
                                   bucket_base_reso=None,
                                   bucket_no_upscale=False,
                                   num_workers=1):
    """Precompute bucket assignments for a UnifiedDataset-style metadata file.

    The output is a jsonl file where each line is:
        {"data_id": <int>, "bucket": [width, height]}
    """
    bucket_args = {
        "max_bucket_reso": max_bucket_reso,
        "min_bucket_reso": min_bucket_reso,
        "bucket_reso_steps": bucket_reso_steps,
        "bucket_base_reso": bucket_base_reso,
        "bucket_no_upscale": bucket_no_upscale,
    }
    build_bucket_manager(**bucket_args)

    ensure_parent_dir(output_path)
    valid_tasks = []
    skipped = 0

    for data_id, record in tqdm(iter_unified_metadata(metadata_path), desc="Scanning Unified metadata"):
            if bucket_data_key not in record:
                skipped += 1
                continue
            value = record[bucket_data_key]
            path = None
            if isinstance(value, str):
                path = value
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                path = value[0]
            if path is None:
                skipped += 1
                continue

            path = resolve_path(base_path, path)
            valid_tasks.append((data_id, path, bucket_args))

    total = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for result in _iterate_results(
            valid_tasks,
            _probe_unified_bucket_task,
            num_workers=num_workers,
            total=len(valid_tasks),
            desc="Computing buckets for UnifiedDataset",
        ):
            if result is None:
                skipped += 1
                continue
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            total += 1

    print(f"Unified bucket index written to {output_path}: {total} items, {skipped} skipped.")


def build_bucket_index_for_image_text_pair(data_dir, output_path,
                                           max_bucket_reso=1024,
                                           min_bucket_reso=256,
                                           bucket_reso_steps=64,
                                           bucket_base_reso=None,
                                           bucket_no_upscale=False,
                                           num_workers=1,
                                           recursive=False):
    """Precompute bucket assignments for an ImageTextPairDataset directory.

    This mirrors ImageTextPairDataset._scan_pairs + _setup_buckets, but only
    computes (data_id -> bucket_reso) and writes to jsonl.
    """
    if recursive:
        print(f"Recursively scanning {data_dir} for pairs...")
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
        image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if os.path.splitext(file)[1] in image_exts:
                    img_path = os.path.join(root, file)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(txt_path):
                        image_paths.append(img_path)
        image_paths.sort()
        tasks = [(data_id, img_path, {
            "max_bucket_reso": max_bucket_reso,
            "min_bucket_reso": min_bucket_reso,
            "bucket_reso_steps": bucket_reso_steps,
            "bucket_base_reso": bucket_base_reso,
            "bucket_no_upscale": bucket_no_upscale,
        }) for data_id, img_path in enumerate(image_paths)]
        print(f"Found {len(tasks)} pairs.")
    else:
        # Reuse the existing dataset's scanning logic for (image, text) pairs.
        ds = ImageTextPairDataset(
            data_dir=data_dir,
            enable_bucket=False,  # we only need pairs, not internal bucketing
        )

        bucket_args = {
            "max_bucket_reso": max_bucket_reso,
            "min_bucket_reso": min_bucket_reso,
            "bucket_reso_steps": bucket_reso_steps,
            "bucket_base_reso": bucket_base_reso,
            "bucket_no_upscale": bucket_no_upscale,
        }
        build_bucket_manager(**bucket_args)

        ensure_parent_dir(output_path)
        tasks = [(data_id, pair["image"], bucket_args) for data_id, pair in enumerate(ds.pairs)]

    total = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for result in _iterate_results(
            tasks,
            _probe_pairs_bucket_task,
            num_workers=num_workers,
            total=len(tasks),
            desc="Computing buckets for ImageTextPairDataset",
        ):
            if result is None:
                skipped += 1
                continue
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            total += 1

    print(f"ImageTextPair bucket index written to {output_path}: {total} items, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser(description="Precompute bucket index jsonl for UnifiedDataset or ImageTextPairDataset.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Unified metadata mode
    p_unified = subparsers.add_parser("unified", help="Build bucket index for UnifiedDataset metadata")
    p_unified.add_argument("--base_path", type=str, required=True, help="Base path for images in metadata")
    p_unified.add_argument("--metadata_path", type=str, required=True, help="Path to metadata file (json/jsonl/csv)")
    p_unified.add_argument("--output", type=str, required=True, help="Output jsonl index path")
    p_unified.add_argument("--bucket_data_key", type=str, default="image", help="Key in metadata that contains image path")

    # Shared bucket params
    for p in (p_unified,):
        p.add_argument("--max_bucket_reso", type=int, default=1024)
        p.add_argument("--min_bucket_reso", type=int, default=256)
        p.add_argument("--bucket_reso_steps", type=int, default=64)
        p.add_argument("--bucket_base_reso", type=int, nargs=2, default=None, help="Base resolution as two ints: H W")
        p.add_argument("--bucket_no_upscale", action="store_true")
        p.add_argument("--num_workers", type=int, default=1, help="Number of CPU worker processes for probing image sizes")

    # ImageTextPair mode
    p_pairs = subparsers.add_parser("pairs", help="Build bucket index for ImageTextPairDataset")
    p_pairs.add_argument("--data_dir", type=str, required=True, help="Directory containing image/txt pairs")
    p_pairs.add_argument("--output", type=str, required=True, help="Output jsonl index path")
    p_pairs.add_argument("--recursive", action="store_true", help="Whether to search for image/txt pairs recursively")
    for p in (p_pairs,):
        p.add_argument("--max_bucket_reso", type=int, default=1024)
        p.add_argument("--min_bucket_reso", type=int, default=256)
        p.add_argument("--bucket_reso_steps", type=int, default=64)
        p.add_argument("--bucket_base_reso", type=int, nargs=2, default=None, help="Base resolution as two ints: H W")
        p.add_argument("--bucket_no_upscale", action="store_true")
        p.add_argument("--num_workers", type=int, default=1, help="Number of CPU worker processes for probing image sizes")

    args = parser.parse_args()

    if args.mode == "unified":
        base_reso = tuple(args.bucket_base_reso) if args.bucket_base_reso is not None else None
        build_bucket_index_for_unified(
            base_path=args.base_path,
            metadata_path=args.metadata_path,
            output_path=args.output,
            bucket_data_key=args.bucket_data_key,
            max_bucket_reso=args.max_bucket_reso,
            min_bucket_reso=args.min_bucket_reso,
            bucket_reso_steps=args.bucket_reso_steps,
            bucket_base_reso=base_reso,
            bucket_no_upscale=args.bucket_no_upscale,
            num_workers=args.num_workers,
        )
    elif args.mode == "pairs":
        base_reso = tuple(args.bucket_base_reso) if args.bucket_base_reso is not None else None
        build_bucket_index_for_image_text_pair(
            data_dir=args.data_dir,
            output_path=args.output,
            max_bucket_reso=args.max_bucket_reso,
            min_bucket_reso=args.min_bucket_reso,
            bucket_reso_steps=args.bucket_reso_steps,
            bucket_base_reso=base_reso,
            bucket_no_upscale=args.bucket_no_upscale,
            num_workers=args.num_workers,
            recursive=args.recursive,
        )


if __name__ == "__main__":
    main()
