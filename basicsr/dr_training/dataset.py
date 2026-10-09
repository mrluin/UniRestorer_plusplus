import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DenseCaptionCSVDataset(Dataset):
    def __init__(
        self, metadata_csv, *, image_root, crop_size=None,
        cutmix_probability, cutmix_alpha, caption_id_mapping=None,
    ):
        self.image_root = Path(image_root)
        with Path(metadata_csv).open(encoding="utf-8-sig", newline="") as stream:
            raw_rows = [(row["image_path"], int(row["caption_id"]))
                        for row in csv.DictReader(stream)]
        self.caption_id_mapping = ({int(alias): int(canonical)
                                    for alias, canonical in caption_id_mapping.items()}
                                   if caption_id_mapping is not None else {})
        self.original_caption_ids = tuple(caption for _, caption in raw_rows)
        # Alias captions must become the same class before donor eligibility.
        self.rows = [(path, self.caption_id_mapping.get(caption, caption))
                     for path, caption in raw_rows]
        self.caption_ids = tuple(sorted({caption for _, caption in self.rows}))
        self._indices_by_caption = {}
        for index, (_, caption) in enumerate(self.rows):
            self._indices_by_caption.setdefault(caption, []).append(index)
        self.crop_size = ((crop_size, crop_size) if isinstance(crop_size, int)
                          else tuple(crop_size) if crop_size is not None else None)
        self.cutmix_probability = float(cutmix_probability)
        self.cutmix_alpha = float(cutmix_alpha)
        self._beta = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha)

    def __len__(self):
        return len(self.rows)

    def _load(self, index):
        relative_path, caption = self.rows[index]
        path = self.image_root / relative_path
        with Image.open(path) as source:
            array = np.array(source.convert("RGB"), dtype=np.float32) / 255.
        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        top = left = 0
        if self.crop_size is not None:
            height, width = self.crop_size
            top = int(torch.randint(image.shape[1] - height + 1, ()).item())
            left = int(torch.randint(image.shape[2] - width + 1, ()).item())
            image = image[:, top:top + height, left:left + width].contiguous()
        return image, caption, str(path), torch.tensor((top, left), dtype=torch.long)

    def _different_caption_index(self, caption):
        # Uniform over eligible samples, without a potentially unbounded retry.
        count = len(self.rows) - len(self._indices_by_caption[caption])
        if count == 0:
            raise ValueError("CutMix requires a source with a different caption_id")
        choice = int(torch.randint(count, ()).item())
        for other_caption, indices in self._indices_by_caption.items():
            if other_caption != caption:
                if choice < len(indices):
                    return indices[choice]
                choice -= len(indices)

    def __getitem__(self, index):
        image, caption, path, origin = self._load(index)
        height, width = image.shape[-2:]
        labels = torch.full((height, width), caption, dtype=torch.long)
        source_index, source_caption, source_path, source_origin = index, caption, path, origin
        bbox = torch.zeros(4, dtype=torch.long)
        sampled_lambda = actual_lambda = 1.
        mixed = False
        if self.cutmix_probability and torch.rand(()).item() < self.cutmix_probability:
            source_index = self._different_caption_index(caption)
            source, source_caption, source_path, source_origin = self._load(source_index)
            if source.shape != image.shape:
                raise ValueError("CutMix requires equal image grids; supply crop_size")
            sampled_lambda = float(self._beta.sample().item())
            ratio = math.sqrt(1. - sampled_lambda)
            cut_width, cut_height = int(width * ratio), int(height * ratio)
            center_x = int(torch.randint(width, ()).item())
            center_y = int(torch.randint(height, ()).item())
            x0, x1 = max(center_x - cut_width // 2, 0), min(center_x + cut_width // 2, width)
            y0, y1 = max(center_y - cut_height // 2, 0), min(center_y + cut_height // 2, height)
            bbox = torch.tensor((x0, y0, x1, y1), dtype=torch.long)
            image[:, y0:y1, x0:x1] = source[:, y0:y1, x0:x1]
            labels[y0:y1, x0:x1] = source_caption
            area = (x1 - x0) * (y1 - y0)
            actual_lambda = 1. - area / (height * width)
            mixed = area > 0
        return {
            "image": image, "caption_id_map": labels,
            "caption_id": caption, "source_caption_id": source_caption,
            "original_caption_id": self.original_caption_ids[index],
            "source_original_caption_id": self.original_caption_ids[source_index],
            "source_index": source_index, "cutmix_bbox": bbox,
            "image_path": path, "source_image_path": source_path,
            "crop_origin": origin, "source_crop_origin": source_origin,
            "sampled_lambda": sampled_lambda, "cutmix_lambda": actual_lambda,
            "mixed": mixed,
        }
