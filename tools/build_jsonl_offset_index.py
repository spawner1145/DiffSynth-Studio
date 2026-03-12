import os
import argparse

"""jsonl 行偏移索引构建脚本（配合 UnifiedDataset.jsonl_index_path 使用）

用途：
    为超大 jsonl metadata 文件（如 metadata_merged.jsonl）构建一份 "行号 -> 文件偏移量" 的索引，
    使得 UnifiedDataset 可以在不把整份 jsonl 读入内存的情况下，按 data_id 随机读取某一行。

基本用法：

    python tools/build_jsonl_offset_index.py \
        --metadata_path /root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl \
        --output /root/autodl-tmp/DiffSynth-Studio/edit/metadata_merged.jsonl.offsets

    生成的 metadata_merged.jsonl.offsets 是一个纯文本文件：
        - 每行一个整数，表示对应样本所在行在 jsonl 文件中的字节偏移量（file offset）；
        - 只记录非空行，空行会被跳过；
        - 第 0 行 offset 对应 data_id=0，第 1 行 offset 对应 data_id=1，以此类推。

    在训练脚本中配合 UnifiedDataset 使用示例：

        dataset = UnifiedDataset(
            base_path=..., metadata_path=".../metadata_merged.jsonl", ...,
            enable_bucket=True,
            bucket_index_path=".../prebucket_index.jsonl",             # 预分桶索引（可选）
            jsonl_index_path=".../metadata_merged.jsonl.offsets",       # 本脚本生成的行偏移索引
        )

    这样：
        - 分桶信息从 prebucket_index.jsonl 读取，不再在训练时逐图 Image.open；
        - metadata 内容通过行偏移索引按需读取单行，而非整份 jsonl 常驻内存，适合千万级样本。
"""


def build_jsonl_offset_index(metadata_path: str, output_path: str):
    """为 jsonl metadata 文件构建行偏移索引（data_id -> file offset）。

    输出索引文件格式：
        每行一个整数，表示 jsonl 文件中某个 *非空行* 的字节偏移量（从文件开头算起）。

    - offsets[0] = 第一条有效样本所在行的文件 offset
    - offsets[1] = 第二条有效样本所在行的文件 offset
    - ...

    仅非空行会被写入索引，空行 / 纯空白行会被跳过；
    因此 data_id 对应的是“有效行”的 0-based 下标，这与 UnifiedDataset 在
    使用 jsonl_index_path 时的行为保持一致。
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    offsets = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                # Skip empty lines; they won't correspond to any data_id.
                continue
            offsets.append(offset)

    with open(output_path, "w", encoding="utf-8") as fout:
        for off in offsets:
            fout.write(str(off) + "\n")

    print(
        f"Built jsonl offset index for {metadata_path} -> {output_path}\n"
        f"  total valid (non-empty) lines indexed: {len(offsets)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build line-offset index file for a jsonl metadata file."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to the jsonl metadata file (e.g. metadata_merged.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the offset index file (one integer offset per line).",
    )
    args = parser.parse_args()

    build_jsonl_offset_index(args.metadata_path, args.output)


if __name__ == "__main__":
    main()
