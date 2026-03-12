from .operators import LoadImage, ImageCropAndResize, ToAbsolutePath
from .unified_dataset import BucketManager
import torch, os, math, random
from PIL import Image
import json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


class ImageTextPairDataset(torch.utils.data.Dataset):
    """Dataset for image/text pairs where each image has a corresponding .txt file.

    Expected directory structure:
        data_dir/
        ├── 001.jpg
        ├── 001.txt
        ├── 002.png
        ├── 002.txt
        └── ...

    The .txt file contains the text prompt for the corresponding image.
    """

    def __init__(
        self,
        data_dir,
        max_pixels=1920 * 1080,
        height=None,
        width=None,
        height_division_factor=16,
        width_division_factor=16,
        repeat=1,
        max_data_items=None,
        # Bucket parameters
        enable_bucket=False,
        bucket_no_upscale=False,
        min_bucket_reso=256,
        max_bucket_reso=1024,
        bucket_reso_steps=64,
        bucket_base_reso=None,
        bucket_index_path=None,
    ):
        self.data_dir = data_dir
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.repeat = repeat
        self.max_data_items = max_data_items

        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.bucket_reso_steps = bucket_reso_steps
        self.bucket_base_reso = bucket_base_reso
        self.bucket_index_path = bucket_index_path

        self.bucket_manager = None
        self.bucket_reso_by_data_id = {}
        self.bucket_to_data_ids = {}
        self.bucket_batch_indices = []
        self.bucket_enabled_batching = False
        self.load_from_cache = False
        self.batch_size = 1
        self.seed = 0
        self.current_epoch = 0
        self.current_step = 0

        self.pairs = []
        self._scan_pairs()
        self._setup_buckets()
        self._rebuild_bucket_batches()

    def _load_bucket_index(self, index_path, max_reso):
        """Load precomputed bucket resolutions from jsonl for ImageTextPairDataset.

        Expected jsonl format (one of following keys is accepted):
            {"data_id": 0, "bucket": [1024, 576]}
            {"idx": 0, "reso": [1024, 576]}

        Here data_id / idx is the 0-based index in self.pairs after _scan_pairs().
        """
        self.bucket_reso_by_data_id = {}
        bucket_counts = {}

        if index_path is None or not os.path.exists(index_path):
            raise ValueError(f"Bucket index file not found: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                data_id = item.get("data_id", item.get("idx", None))
                reso = item.get("bucket", item.get("reso", None))
                if data_id is None or reso is None or not isinstance(reso, (list, tuple)) or len(reso) != 2:
                    continue

                w, h = int(reso[0]), int(reso[1])
                if w < self.min_bucket_reso or h < self.min_bucket_reso:
                    continue
                if max(w, h) > self.max_bucket_reso:
                    continue
                if w % self.bucket_reso_steps != 0 or h % self.bucket_reso_steps != 0:
                    continue

                did = int(data_id)
                if did < 0 or did >= len(self.pairs):
                    continue

                reso_tuple = (w, h)
                self.bucket_reso_by_data_id[did] = reso_tuple
                bucket_counts[reso_tuple] = bucket_counts.get(reso_tuple, 0) + 1

        if len(self.bucket_reso_by_data_id) == 0:
            raise ValueError("Bucket index file is provided but no valid items were loaded.")

        print(
            f"Bucket enabled with precomputed index: {len(bucket_counts)} buckets, "
            f"{len(self.bucket_reso_by_data_id)} items."
        )

    def _scan_pairs(self):
        """Scan data_dir for image/txt pairs."""
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"data_dir does not exist: {self.data_dir}")

        image_stems = {}
        for fname in os.listdir(self.data_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                stem = os.path.splitext(fname)[0]
                image_stems[stem] = fname

        for stem in sorted(image_stems.keys()):
            txt_path = os.path.join(self.data_dir, stem + ".txt")
            if os.path.isfile(txt_path):
                self.pairs.append({
                    "image": os.path.join(self.data_dir, image_stems[stem]),
                    "text": txt_path,
                })

        if len(self.pairs) == 0:
            raise ValueError(
                f"No image/txt pairs found in {self.data_dir}. "
                "Expected matching files like 001.jpg + 001.txt."
            )
        print(f"ImageTextPairDataset: found {len(self.pairs)} pairs in {self.data_dir}")

    def _setup_buckets(self):
        if not self.enable_bucket:
            return

        self.bucket_reso_steps = int(self.bucket_reso_steps)
        self.min_bucket_reso = int(self.min_bucket_reso)

        if isinstance(self.bucket_base_reso, (tuple, list)) and len(self.bucket_base_reso) == 2:
            max_reso = (int(self.bucket_base_reso[0]), int(self.bucket_base_reso[1]))
        elif isinstance(self.max_bucket_reso, (tuple, list)) and len(self.max_bucket_reso) == 2:
            max_reso = (int(self.max_bucket_reso[0]), int(self.max_bucket_reso[1]))
        else:
            base_side = int(self.max_bucket_reso)
            max_reso = (base_side, base_side)

        if isinstance(self.max_bucket_reso, (tuple, list)):
            max_bucket_limit = int(max(self.max_bucket_reso))
        else:
            max_bucket_limit = int(self.max_bucket_reso)

        # Align to reso_steps
        if self.min_bucket_reso % self.bucket_reso_steps != 0:
            self.min_bucket_reso = max(
                self.bucket_reso_steps,
                self.min_bucket_reso - self.min_bucket_reso % self.bucket_reso_steps,
            )
        if max_bucket_limit % self.bucket_reso_steps != 0:
            max_bucket_limit += self.bucket_reso_steps - max_bucket_limit % self.bucket_reso_steps
        max_bucket_limit = max(self.bucket_reso_steps, max_bucket_limit)
        self.max_bucket_reso = max_bucket_limit

        self.bucket_manager = BucketManager(
            no_upscale=self.bucket_no_upscale,
            max_reso=max_reso,
            min_size=self.min_bucket_reso,
            max_size=max_bucket_limit,
            reso_steps=self.bucket_reso_steps,
        )
        if not self.bucket_no_upscale:
            self.bucket_manager.make_buckets()

        # If precomputed bucket index is provided, use it and skip scanning images.
        if self.bucket_index_path is not None:
            self._load_bucket_index(self.bucket_index_path, max_reso)
            return

        bucket_counts = {}
        skipped = 0
        for data_id, pair in enumerate(self.pairs):
            try:
                with Image.open(pair["image"]) as img:
                    w, h = img.size
            except Exception:
                skipped += 1
                continue
            reso, _, _ = self.bucket_manager.select_bucket(w, h)
            self.bucket_reso_by_data_id[data_id] = reso
            bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        if len(self.bucket_reso_by_data_id) == 0:
            raise ValueError("Bucket enabled but no valid images could be read.")

        print(f"Bucket enabled: {len(bucket_counts)} buckets, {skipped} skipped.")
        for (w, h), count in sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  ({w}, {h}) => {count}")

    def _rebuild_bucket_batches(self):
        self.bucket_to_data_ids = {}
        self.bucket_batch_indices = []
        self.bucket_enabled_batching = False

        if not self.enable_bucket or self.batch_size <= 1:
            return
        if len(self.bucket_reso_by_data_id) == 0:
            return

        bucket_to_data_ids = {}
        for data_id, reso in self.bucket_reso_by_data_id.items():
            bucket_to_data_ids.setdefault(reso, []).append(data_id)

        for reso in sorted(bucket_to_data_ids.keys()):
            data_ids = list(bucket_to_data_ids[reso])
            if self.repeat > 1:
                data_ids = data_ids * int(self.repeat)
            self.bucket_to_data_ids[reso] = data_ids
            batch_count = math.ceil(len(data_ids) / self.batch_size)
            for batch_idx in range(batch_count):
                self.bucket_batch_indices.append((reso, batch_idx))

        if self.max_data_items is not None and self.max_data_items > 0:
            max_batches = math.ceil(int(self.max_data_items) / self.batch_size)
            self.bucket_batch_indices = self.bucket_batch_indices[:max_batches]

        self.bucket_enabled_batching = len(self.bucket_batch_indices) > 0
        self._shuffle_buckets()

    def _shuffle_buckets(self):
        if not self.bucket_enabled_batching:
            return
        rng = random.Random(self.seed + self.current_epoch)
        for reso in self.bucket_to_data_ids:
            rng.shuffle(self.bucket_to_data_ids[reso])
        rng.shuffle(self.bucket_batch_indices)

    def set_seed(self, seed):
        self.seed = int(seed)

    def set_batch_size(self, batch_size):
        batch_size = max(1, int(batch_size))
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self._rebuild_bucket_batches()

    def set_current_epoch(self, epoch):
        epoch = int(epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            self._shuffle_buckets()

    def set_current_step(self, step):
        self.current_step = int(step)

    def _load_text(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _load_image(self, image_path, target_reso=None):
        image = Image.open(image_path).convert("RGB")
        if target_reso is not None:
            target_w, target_h = target_reso
            op = ImageCropAndResize(
                height=target_h, width=target_w,
                max_pixels=None,
                height_division_factor=1, width_division_factor=1,
            )
        else:
            op = ImageCropAndResize(
                height=self.height, width=self.width,
                max_pixels=self.max_pixels,
                height_division_factor=self.height_division_factor,
                width_division_factor=self.width_division_factor,
            )
        return op(image)

    def _process_single(self, data_id, force_reso=None):
        pair = self.pairs[data_id]
        reso = force_reso
        if reso is None and self.enable_bucket:
            reso = self.bucket_reso_by_data_id.get(data_id)
        return {
            "image": self._load_image(pair["image"], target_reso=reso),
            "prompt": self._load_text(pair["text"]),
        }

    def __getitem__(self, idx):
        if self.bucket_enabled_batching:
            bucket_reso, batch_idx = self.bucket_batch_indices[idx % len(self.bucket_batch_indices)]
            source_ids = self.bucket_to_data_ids[bucket_reso]
            start = batch_idx * self.batch_size
            batch_ids = source_ids[start:start + self.batch_size]
            return [self._process_single(did, force_reso=bucket_reso) for did in batch_ids]
        else:
            source_id = idx % len(self.pairs)
            return self._process_single(source_id)

    def __len__(self):
        if self.bucket_enabled_batching:
            return len(self.bucket_batch_indices)
        if self.max_data_items is not None:
            return int(self.max_data_items)
        return len(self.pairs) * self.repeat
