import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


"""递归扫描图片目录并计算逐通道 mean/std。

设计目标：
    - 支持递归扫描子目录；
    - 流式聚合，不把所有图片像素一次性加载进内存；
    - 支持多进程并行，参考 build_bucket_index.py 的 worker 设计；
    - 默认输出 RGB 统计量，同时给出 [0,1] 与 [-1,1] 两种数值域结果。

示例：
    python tools/compute_image_mean_std.py ^
        --input_dir D:/data/images ^
        --output D:/data/image_mean_std.json ^
        --num_workers 8

    python tools/compute_image_mean_std.py ^
        --input_dir D:/data/images ^
        --mode RGBA ^
        --ext .png .webp ^
        --num_workers 4
"""


DEFAULT_IMAGE_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
)


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def normalize_exts(exts):
    normalized = []
    for ext in exts:
        value = str(ext).strip()
        if not value:
            continue
        if not value.startswith("."):
            value = "." + value
        normalized.append(value.lower())
    return tuple(sorted(set(normalized)))


def iter_image_paths(input_dir: str, recursive: bool, exts):
    exts = set(exts)
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file_name in files:
                if os.path.splitext(file_name)[1].lower() in exts:
                    yield os.path.join(root, file_name)
        return

    for file_name in os.listdir(input_dir):
        path = os.path.join(input_dir, file_name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(file_name)[1].lower() in exts:
            yield path


def _compute_image_stats(task):
    path, mode = task
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode != mode:
                image = image.convert(mode)
            array = np.asarray(image, dtype=np.float32) / 255.0

        if array.ndim == 2:
            array = array[..., None]

        flat = array.reshape(-1, array.shape[-1])
        channel_sum = flat.sum(axis=0, dtype=np.float64)
        channel_sum_sq = np.square(flat, dtype=np.float32).sum(axis=0, dtype=np.float64)
        pixel_count = int(flat.shape[0])
        return {
            "path": path,
            "ok": True,
            "sum": channel_sum.tolist(),
            "sum_sq": channel_sum_sq.tolist(),
            "pixel_count": pixel_count,
        }
    except Exception as e:
        return {
            "path": path,
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


def _iterate_results(tasks, worker_fn, num_workers, total=None, desc=None):
    if int(num_workers) <= 1:
        for task in tqdm(tasks, total=total, desc=desc):
            yield worker_fn(task)
        return

    chunksize = 64 if total and total > 10000 else 16
    with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
        for result in tqdm(executor.map(worker_fn, tasks, chunksize=chunksize), total=total, desc=desc):
            yield result


def compute_image_mean_std(
    input_dir: str,
    *,
    mode: str = "RGB",
    recursive: bool = True,
    exts=DEFAULT_IMAGE_EXTS,
    num_workers: int = 1,
):
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"input_dir not found: {input_dir}")

    mode = str(mode).upper()
    if mode not in ("RGB", "RGBA", "L"):
        raise ValueError("mode must be one of: RGB / RGBA / L")

    paths = sorted(iter_image_paths(input_dir, recursive=recursive, exts=normalize_exts(exts)))
    if len(paths) == 0:
        raise ValueError(f"No images found under {input_dir}")

    channel_count = len(mode)
    total_sum = np.zeros((channel_count,), dtype=np.float64)
    total_sum_sq = np.zeros((channel_count,), dtype=np.float64)
    total_pixels = 0
    num_success = 0
    failures = []

    tasks = ((path, mode) for path in paths)
    for result in _iterate_results(
        tasks,
        _compute_image_stats,
        num_workers=num_workers,
        total=len(paths),
        desc="Computing image mean/std",
    ):
        if not result["ok"]:
            failures.append({"path": result["path"], "error": result["error"]})
            continue
        total_sum += np.asarray(result["sum"], dtype=np.float64)
        total_sum_sq += np.asarray(result["sum_sq"], dtype=np.float64)
        total_pixels += int(result["pixel_count"])
        num_success += 1

    if total_pixels <= 0:
        raise RuntimeError("No valid pixels were processed.")

    mean_01 = total_sum / float(total_pixels)
    var_01 = total_sum_sq / float(total_pixels) - np.square(mean_01)
    var_01 = np.maximum(var_01, 0.0)
    std_01 = np.sqrt(var_01)

    mean_11 = mean_01 * 2.0 - 1.0
    std_11 = std_01 * 2.0

    return {
        "input_dir": os.path.abspath(input_dir),
        "mode": mode,
        "recursive": bool(recursive),
        "extensions": list(normalize_exts(exts)),
        "num_workers": int(num_workers),
        "num_images_found": len(paths),
        "num_images_processed": int(num_success),
        "num_images_failed": len(failures),
        "total_pixels_per_channel": int(total_pixels),
        "mean_01": [float(v) for v in mean_01.tolist()],
        "std_01": [float(v) for v in std_01.tolist()],
        "mean_11": [float(v) for v in mean_11.tolist()],
        "std_11": [float(v) for v in std_11.tolist()],
        "failed_files": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute per-channel image mean/std for a directory tree.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing images.")
    parser.add_argument("--output", type=str, default=None, help="Optional output json path.")
    parser.add_argument("--mode", type=str, default="RGB", help="Image convert mode: RGB / RGBA / L")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--no_recursive", action="store_true", help="Disable recursive scanning.")
    parser.add_argument(
        "--ext",
        nargs="*",
        default=list(DEFAULT_IMAGE_EXTS),
        help="Image extensions to include, e.g. --ext .jpg .png .webp",
    )
    parser.add_argument(
        "--print_failed_limit",
        type=int,
        default=20,
        help="How many failed files to print to stdout at the end.",
    )
    args = parser.parse_args()

    result = compute_image_mean_std(
        args.input_dir,
        mode=args.mode,
        recursive=not args.no_recursive,
        exts=args.ext,
        num_workers=args.num_workers,
    )

    if args.output:
        ensure_parent_dir(args.output)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(
        {
            "input_dir": result["input_dir"],
            "mode": result["mode"],
            "num_images_found": result["num_images_found"],
            "num_images_processed": result["num_images_processed"],
            "num_images_failed": result["num_images_failed"],
            "total_pixels_per_channel": result["total_pixels_per_channel"],
            "mean_01": result["mean_01"],
            "std_01": result["std_01"],
            "mean_11": result["mean_11"],
            "std_11": result["std_11"],
            "output": os.path.abspath(args.output) if args.output else None,
        },
        ensure_ascii=False,
        indent=2,
    ))

    if result["num_images_failed"] > 0:
        print("\nFailed files:")
        for item in result["failed_files"][: max(0, int(args.print_failed_limit))]:
            print(f"- {item['path']}: {item['error']}")


if __name__ == "__main__":
    main()
