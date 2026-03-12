from .operators import *
import torch, json, pandas, os, math
import random
from PIL import Image


class BucketManager:
    # Adapted from sd-scripts BucketManager for dataset-side resolution bucketing.
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps):
        self.no_upscale = no_upscale
        self.max_reso = max_reso
        self.max_area = None if max_reso is None else max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps
        self.resos = []
        self.reso_to_id = {}

        self.predefined_resos = []
        self.predefined_resos_set = set()
        self.predefined_aspect_ratios = []

    def set_predefined_resos(self, resos):
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = [w / h for w, h in resos]

    def make_buckets(self):
        if self.max_reso is None:
            self.set_predefined_resos([])
            return
        max_width, max_height = self.max_reso
        max_area = max_width * max_height
        min_size = self.min_size
        max_size = self.max_size
        divisible = self.reso_steps

        resos = set()
        width = int(math.sqrt(max_area) // divisible) * divisible
        resos.add((width, width))

        width = min_size
        while width <= max_size:
            height = min(max_size, int((max_area // width) // divisible) * divisible)
            if height >= min_size:
                resos.add((width, height))
                resos.add((height, width))
            width += divisible
        resos = sorted(list(resos))
        self.set_predefined_resos(resos)

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            self.reso_to_id[reso] = len(self.resos)
            self.resos.append(reso)

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            reso = (image_width, image_height)
            if reso not in self.predefined_resos_set:
                ar_errors = [abs(ar - aspect_ratio) for ar in self.predefined_aspect_ratios]
                predefined_bucket_id = ar_errors.index(min(ar_errors))
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width
            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
        else:
            if self.max_area is not None and image_width * image_height > self.max_area:
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width

                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                b_height_in_wr = max(self.reso_steps, b_height_in_wr)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                b_width_in_hr = max(self.reso_steps, b_width_in_hr)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
            else:
                resized_size = (image_width, image_height)

            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            bucket_width = max(self.reso_steps, bucket_width)
            bucket_height = max(self.reso_steps, bucket_height)
            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)
        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        enable_bucket=False,
        bucket_no_upscale=False,
        min_bucket_reso=256,
        max_bucket_reso=1024,
        bucket_reso_steps=64,
        bucket_data_key=None,
        bucket_base_reso=None,
        bucket_index_path=None,
        jsonl_index_path=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale
        self.min_bucket_reso = min_bucket_reso
        self.max_bucket_reso = max_bucket_reso
        self.bucket_reso_steps = bucket_reso_steps
        self.bucket_data_key = bucket_data_key
        self.bucket_base_reso = bucket_base_reso
        self.bucket_index_path = bucket_index_path
        self.jsonl_index_path = jsonl_index_path
        self.metadata_file_path = None
        self.jsonl_offsets = []  # list[int], data_id -> file offset
        self.bucket_manager = None
        self.bucket_reso_by_data_id = {}
        self.bucket_info = {}
        self.bucket_to_data_ids = {}
        self.bucket_batch_indices = []
        self.bucket_enabled_batching = False
        self.batch_size = 1
        self.seed = 0
        self.current_epoch = 0
        self.current_step = 0
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        self.setup_buckets_if_needed()
        self.rebuild_bucket_batches_if_needed()

    def _load_bucket_index(self, index_path, max_reso):
        """Load precomputed bucket resolutions from a jsonl index file.

        Expected jsonl format (one of following keys is accepted):
            {"data_id": 0, "bucket": [1024, 576]}
            {"idx": 0, "reso": [1024, 576]}

        - data_id / idx: 0-based index in self.data (metadata list)
        - bucket / reso: [width, height] already snapped to bucket_reso_steps

        This function fills self.bucket_reso_by_data_id and self.bucket_info,
        then UnifiedDataset will reuse the existing batch building logic.
        """
        bucket_counts = {}
        self.bucket_reso_by_data_id = {}

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

                # Basic sanity checks to keep consistency with current bucket constraints.
                if w < self.min_bucket_reso or h < self.min_bucket_reso:
                    continue
                if max(w, h) > self.max_bucket_reso:
                    continue
                if w % self.bucket_reso_steps != 0 or h % self.bucket_reso_steps != 0:
                    continue

                reso_tuple = (w, h)
                did = int(data_id)
                if did < 0 or did >= len(self.data):
                    continue

                self.bucket_reso_by_data_id[did] = reso_tuple
                bucket_counts[reso_tuple] = bucket_counts.get(reso_tuple, 0) + 1

        if len(self.bucket_reso_by_data_id) == 0:
            raise ValueError("Bucket index file is provided but no valid items were loaded.")

        self.bucket_info = {
            "bucket_data_key": self.bucket_data_key,
            "bucket_no_upscale": self.bucket_no_upscale,
            "min_bucket_reso": self.min_bucket_reso,
            "max_bucket_reso": self.max_bucket_reso,
            "max_reso": [max_reso[0], max_reso[1]] if max_reso is not None else None,
            "bucket_reso_steps": self.bucket_reso_steps,
            "num_buckets": len(bucket_counts),
            "num_bucketed_items": len(self.bucket_reso_by_data_id),
            "num_skipped_missing_key": 0,
            "num_skipped_invalid_value": 0,
            "num_skipped_missing_file": 0,
            "num_skipped_unreadable": 0,
            "bucket_counts": {f"{w}x{h}": count for (w, h), count in sorted(bucket_counts.items())},
        }
        print(
            f"Bucket enabled with precomputed index: "
            f"{self.bucket_info['num_buckets']} buckets, "
            f"{self.bucket_info['num_bucketed_items']} items."
        )
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            self.metadata_file_path = metadata_path
            if self.jsonl_index_path is not None and os.path.exists(self.jsonl_index_path):
                offsets = []
                with open(self.jsonl_index_path, "r", encoding="utf-8") as idx_f:
                    for line in idx_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            offsets.append(int(line))
                        except Exception:
                            continue
                self.jsonl_offsets = offsets
                self.data = []  # metadata records will be read lazily
                print(f"UnifiedDataset: loaded {len(self.jsonl_offsets)} jsonl offsets from index {self.jsonl_index_path}")
            else:
                # Fallback: load all metadata into memory (legacy behavior)
                metadata = []
                with open(metadata_path, 'r', encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        metadata.append(json.loads(line))
                self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def get_record_by_id(self, data_id: int):
        """Return a single metadata record by id.

        - If jsonl_index_path is provided and jsonl_offsets is populated, we read
          the corresponding line from the jsonl file lazily.
        - Otherwise, we fall back to self.data[data_id].
        """
        if self.metadata_file_path is not None and self.jsonl_offsets:
            # Streaming mode for jsonl metadata.
            if data_id < 0 or data_id >= len(self.jsonl_offsets):
                raise IndexError("data_id out of range for jsonl metadata")
            offset = self.jsonl_offsets[data_id]
            with open(self.metadata_file_path, "r", encoding="utf-8") as f:
                f.seek(offset)
                line = f.readline()
            return json.loads(line.strip())

        # Legacy in-memory behavior
        return self.data[data_id]

    def resolve_data_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_path, path)

    def fetch_bucket_data_key(self):
        if self.bucket_data_key is not None:
            return self.bucket_data_key
        if len(self.data_file_keys) > 0:
            return self.data_file_keys[0]
        return None

    def fetch_image_size_from_data_value(self, value):
        path = None
        if isinstance(value, str):
            path = value
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
            path = value[0]
        if path is None:
            return None
        path = self.resolve_data_path(path)
        if not os.path.exists(path):
            return None
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception:
            return None
        return width, height

    def setup_buckets_if_needed(self):
        if not self.enable_bucket:
            return
        if self.load_from_cache:
            print("Bucket is enabled but metadata_path is None. Disabling bucket for cached .pth data.")
            self.enable_bucket = False
            return

        bucket_data_key = self.fetch_bucket_data_key()
        if bucket_data_key is None:
            print("Bucket is enabled but no data_file_keys are provided. Disabling bucket.")
            self.enable_bucket = False
            return

        self.bucket_data_key = bucket_data_key
        self.bucket_reso_steps = int(self.bucket_reso_steps)
        if self.bucket_reso_steps <= 0:
            raise ValueError("bucket_reso_steps must be a positive integer.")

        self.min_bucket_reso = int(self.min_bucket_reso)
        if isinstance(self.bucket_base_reso, (tuple, list)) and len(self.bucket_base_reso) == 2:
            max_reso = (int(self.bucket_base_reso[0]), int(self.bucket_base_reso[1]))
        elif isinstance(self.max_bucket_reso, (tuple, list)) and len(self.max_bucket_reso) == 2:
            max_reso = (int(self.max_bucket_reso[0]), int(self.max_bucket_reso[1]))
        else:
            base_side = int(self.max_bucket_reso)
            max_reso = (base_side, base_side)

        if isinstance(self.max_bucket_reso, (tuple, list)) and len(self.max_bucket_reso) == 2:
            max_bucket_reso_limit = int(max(self.max_bucket_reso))
        else:
            max_bucket_reso_limit = int(self.max_bucket_reso)

        if self.min_bucket_reso % self.bucket_reso_steps != 0:
            adjusted_min_bucket_reso = self.min_bucket_reso - self.min_bucket_reso % self.bucket_reso_steps
            adjusted_min_bucket_reso = max(self.bucket_reso_steps, adjusted_min_bucket_reso)
            print(f"min_bucket_reso is adjusted to be multiple of bucket_reso_steps: {self.min_bucket_reso} -> {adjusted_min_bucket_reso}")
            self.min_bucket_reso = adjusted_min_bucket_reso
        if max_bucket_reso_limit % self.bucket_reso_steps != 0:
            adjusted_max_bucket_reso = max_bucket_reso_limit + self.bucket_reso_steps - max_bucket_reso_limit % self.bucket_reso_steps
            print(f"max_bucket_reso is adjusted to be multiple of bucket_reso_steps: {max_bucket_reso_limit} -> {adjusted_max_bucket_reso}")
            max_bucket_reso_limit = adjusted_max_bucket_reso
        if max_bucket_reso_limit < self.bucket_reso_steps:
            max_bucket_reso_limit = self.bucket_reso_steps
        self.max_bucket_reso = max_bucket_reso_limit

        if self.max_bucket_reso < self.min_bucket_reso:
            raise ValueError("max_bucket_reso must be greater than or equal to min_bucket_reso.")

        if min(max_reso) < self.min_bucket_reso:
            raise ValueError("min_bucket_reso must be equal or less than min(resolution).")
        if max(max_reso) > self.max_bucket_reso:
            raise ValueError("max_bucket_reso must be equal or greater than max(resolution).")
        max_bucket_size = int(self.max_bucket_reso)
        self.bucket_manager = BucketManager(
            no_upscale=self.bucket_no_upscale,
            max_reso=max_reso,
            min_size=self.min_bucket_reso,
            max_size=max_bucket_size,
            reso_steps=self.bucket_reso_steps,
        )
        if not self.bucket_no_upscale:
            self.bucket_manager.make_buckets()

        # If a precomputed bucket index is provided, use it directly and
        # skip scanning images from disk.
        if self.bucket_index_path is not None:
            self._load_bucket_index(self.bucket_index_path, max_reso)
            return

        bucket_counts = {}
        skipped_missing_key = 0
        skipped_invalid_value = 0
        skipped_missing_file = 0
        skipped_unreadable = 0

        # Iterate over all items regardless of whether metadata is kept in memory
        # or accessed lazily via jsonl index.
        num_items = len(self.jsonl_offsets) if (self.metadata_file_path is not None and self.jsonl_offsets) else len(self.data)
        for data_id in range(num_items):
            data = self.get_record_by_id(data_id)
            if bucket_data_key not in data:
                skipped_missing_key += 1
                continue

            value = data[bucket_data_key]
            path = None
            if isinstance(value, str):
                path = value
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                path = value[0]

            if path is None:
                skipped_invalid_value += 1
                continue

            path = self.resolve_data_path(path)
            if not os.path.exists(path):
                skipped_missing_file += 1
                continue

            try:
                with Image.open(path) as image:
                    image_width, image_height = image.size
            except Exception:
                skipped_unreadable += 1
                continue

            reso, _, _ = self.bucket_manager.select_bucket(image_width, image_height)
            self.bucket_reso_by_data_id[data_id] = reso
            bucket_counts[reso] = bucket_counts.get(reso, 0) + 1

        if len(self.bucket_reso_by_data_id) == 0:
            raise ValueError(
                "Bucket is enabled but no valid items are available for bucketing. "
                + f"missing_key={skipped_missing_key}, invalid_value={skipped_invalid_value}, "
                + f"missing_file={skipped_missing_file}, unreadable={skipped_unreadable}."
            )

        self.bucket_info = {
            "bucket_data_key": self.bucket_data_key,
            "bucket_no_upscale": self.bucket_no_upscale,
            "min_bucket_reso": self.min_bucket_reso,
            "max_bucket_reso": self.max_bucket_reso,
            "max_reso": [max_reso[0], max_reso[1]],
            "bucket_reso_steps": self.bucket_reso_steps,
            "num_buckets": len(bucket_counts),
            "num_bucketed_items": len(self.bucket_reso_by_data_id),
            "num_skipped_missing_key": skipped_missing_key,
            "num_skipped_invalid_value": skipped_invalid_value,
            "num_skipped_missing_file": skipped_missing_file,
            "num_skipped_unreadable": skipped_unreadable,
            "bucket_counts": {f"{w}x{h}": count for (w, h), count in sorted(bucket_counts.items())},
        }
        print(f"Bucket enabled: {self.bucket_info['num_buckets']} buckets on key '{self.bucket_data_key}'.")
        skipped_total = skipped_missing_key + skipped_invalid_value + skipped_missing_file + skipped_unreadable
        if skipped_total > 0:
            print(
                "Bucket skipped items: "
                + f"missing_key={skipped_missing_key}, invalid_value={skipped_invalid_value}, "
                + f"missing_file={skipped_missing_file}, unreadable={skipped_unreadable}"
            )
        if len(bucket_counts) > 0:
            sorted_bucket_items = sorted(bucket_counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
            for (w, h), count in sorted_bucket_items:
                print(f"({w},{h})==>{count}")

    def set_seed(self, seed):
        self.seed = int(seed)

    def set_batch_size(self, batch_size):
        batch_size = max(1, int(batch_size))
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.rebuild_bucket_batches_if_needed()

    def set_current_epoch(self, epoch):
        epoch = int(epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            self.shuffle_buckets()

    def set_current_step(self, step):
        self.current_step = int(step)

    def rebuild_bucket_batches_if_needed(self):
        self.bucket_to_data_ids = {}
        self.bucket_batch_indices = []
        self.bucket_enabled_batching = False

        if (not self.enable_bucket) or self.load_from_cache:
            return
        if self.batch_size <= 1:
            return
        if len(self.bucket_reso_by_data_id) == 0:
            return

        bucket_to_data_ids = {}
        for data_id, reso in self.bucket_reso_by_data_id.items():
            if reso not in bucket_to_data_ids:
                bucket_to_data_ids[reso] = []
            bucket_to_data_ids[reso].append(data_id)

        self.bucket_to_data_ids = {}
        self.bucket_batch_indices = []
        for reso in sorted(bucket_to_data_ids.keys()):
            data_ids = list(bucket_to_data_ids[reso])
            if self.repeat > 1:
                data_ids = data_ids * int(self.repeat)
            self.bucket_to_data_ids[reso] = data_ids
            batch_count = int(math.ceil(len(data_ids) / self.batch_size))
            for batch_index in range(batch_count):
                self.bucket_batch_indices.append((reso, batch_index))

        if self.max_data_items is not None and self.max_data_items > 0:
            max_batches = int(math.ceil(int(self.max_data_items) / self.batch_size))
            self.bucket_batch_indices = self.bucket_batch_indices[:max_batches]

        self.bucket_enabled_batching = len(self.bucket_batch_indices) > 0
        self.shuffle_buckets()

    def shuffle_buckets(self):
        if not self.bucket_enabled_batching:
            return
        random.seed(self.seed + self.current_epoch)
        for reso in self.bucket_to_data_ids:
            random.shuffle(self.bucket_to_data_ids[reso])
        random.shuffle(self.bucket_batch_indices)

    def apply_bucket_image_operator(self, value, reso):
        target_width, target_height = reso
        operator = ImageCropAndResize(height=target_height, width=target_width, max_pixels=None, height_division_factor=1, width_division_factor=1)

        if isinstance(value, str):
            image = LoadImage()(ToAbsolutePath(self.base_path)(value))
            return operator(image)
        elif isinstance(value, list):
            return [operator(LoadImage()(ToAbsolutePath(self.base_path)(item))) for item in value]
        return self.main_data_operator(value)

    def process_single_data(self, source_data_id, force_reso=None):
        data = self.data[source_data_id].copy()
        for key in self.data_file_keys:
            if key in data:
                if key in self.special_operator_map:
                    data[key] = self.special_operator_map[key](data[key])
                elif self.enable_bucket and key == self.bucket_data_key:
                    reso = force_reso if force_reso is not None else self.bucket_reso_by_data_id.get(source_data_id)
                    if reso is not None:
                        data[key] = self.apply_bucket_image_operator(data[key], reso)
                    else:
                        data[key] = self.main_data_operator(data[key])
                elif key in self.data_file_keys:
                    data[key] = self.main_data_operator(data[key])
        return data

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        elif self.bucket_enabled_batching:
            bucket_reso, batch_index = self.bucket_batch_indices[data_id % len(self.bucket_batch_indices)]
            source_data_ids = self.bucket_to_data_ids[bucket_reso]
            image_index = batch_index * self.batch_size
            batch_data_ids = source_data_ids[image_index : image_index + self.batch_size]
            data = [self.process_single_data(source_data_id, force_reso=bucket_reso) for source_data_id in batch_data_ids]
        else:
            source_data_id = data_id % len(self.data)
            data = self.process_single_data(source_data_id)
        return data

    def __len__(self):
        if self.load_from_cache:
            if self.max_data_items is not None:
                return int(self.max_data_items)
            return len(self.cached_data) * self.repeat
        elif self.bucket_enabled_batching:
            return len(self.bucket_batch_indices)
        else:
            if self.max_data_items is not None:
                return int(self.max_data_items)
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
