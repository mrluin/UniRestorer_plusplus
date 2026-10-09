import csv
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(records):
    return [{key: row[key] for key in ("sample_id", "lq_path", "gt_path")} for row in records]


def binding_sha256(records):
    payload = json.dumps(binding(records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_pairs(options, base_dir):
    with (base_dir / options["pair_csv"]).open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in records]
    if not records or any(not sample.strip() for sample in ids) or len(set(ids)) != len(ids):
        raise ValueError("Pair CSV requires nonempty unique sample_id values")
    records.sort(key=lambda row: row["sample_id"])
    lq_root, gt_root = base_dir / options["lq_root"], base_dir / options["gt_root"]
    for index, row in enumerate(records):
        row["_index"] = index
        row["_lq_file"], row["_gt_file"] = lq_root / row["lq_path"], gt_root / row["gt_path"]
        with Image.open(row["_lq_file"]) as lq, Image.open(row["_gt_file"]) as gt:
            if lq.size != gt.size:
                raise ValueError("LQ/GT grids differ for sample_id " + row["sample_id"])
            row["_size"] = (lq.height, lq.width)
    return records


class PairedCSVDataset(Dataset):
    def __init__(self, records, *, seed, crop_size=None):
        self.records = records
        self.seed, self.epoch = seed, 0
        self.crop_size = ((crop_size, crop_size) if isinstance(crop_size, int)
                          else tuple(crop_size) if crop_size is not None else None)
        if self.crop_size is not None:
            for row in records:
                if any(size < crop for size, crop in zip(row["_size"], self.crop_size)):
                    raise ValueError("crop_size exceeds the image grid for " + row["sample_id"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        images = []
        for key in ("_lq_file", "_gt_file"):
            with Image.open(row[key]) as source:
                pixels = np.array(source.convert("RGB"), dtype=np.float32) / 255.
            images.append(torch.from_numpy(pixels).permute(2, 0, 1).contiguous())
        if self.crop_size is not None:
            height, width = self.crop_size
            # Crop draws depend on sample/epoch, so resuming skips no RNG history.
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + row["_index"])
            top = int(torch.randint(images[0].shape[1] - height + 1, (), generator=generator))
            left = int(torch.randint(images[0].shape[2] - width + 1, (), generator=generator))
            images = [image[:, top:top + height, left:left + width].contiguous() for image in images]
        return {"lq": images[0], "gt": images[1], "sample_id": row["sample_id"]}


class MatchedHierarchyPairedDataset(PairedCSVDataset):
    """Paired RGB crops with all-level group maps in hierarchy sample order."""

    def __init__(self, records, labels, *, seed, crop_size=None):
        super().__init__(records, seed=seed, crop_size=crop_size)
        labels = torch.as_tensor(labels)
        if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Hierarchy labels must have an integer dtype")
        if labels.ndim != 2 or labels.shape[0] != len(records) or labels.shape[1] == 0:
            raise ValueError("Hierarchy labels must be N-by-L with one row per record and L > 0")
        self.labels = labels.long()
        sizes = self.labels.new_tensor([2 ** level for level in range(self.labels.shape[1])])
        if (torch.any(self.labels < 0) or torch.any(self.labels >= sizes) or
                torch.any(self.labels[:, 0] != 0) or
                torch.any(self.labels[:, 1:] // 2 != self.labels[:, :-1])):
            raise ValueError("Hierarchy labels must follow root 0, level ranges, and binary parent ancestry")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.records[index]
        height, width = sample["lq"].shape[-2:]
        top = left = 0
        if self.crop_size is not None:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003 + row["_index"])
            top = int(torch.randint(row["_size"][0] - height + 1, (), generator=generator))
            left = int(torch.randint(row["_size"][1] - width + 1, (), generator=generator))
        sample.update({
            "group_maps": self.labels[index, :, None, None].expand(-1, height, width).clone(),
            "hierarchy_row": row["_index"],
            "crop_origin": (top, left),
            "crop_size": (height, width),
            "original_size": row["_size"],
        })
        return sample


def paired_cutmix(recipient, donor, bbox):
    """Copy one half-open (x0, y0, x1, y1) box across RGB pairs and group maps."""
    keys = ("lq", "gt", "group_maps")
    if any(not isinstance(sample[key], torch.Tensor) or sample[key].ndim != 3
           for sample in (recipient, donor) for key in keys):
        raise ValueError("CutMix requires single-sample CHW tensors")
    if any(sample["lq"].shape != sample["gt"].shape or sample["lq"].shape[0] != 3 or
           sample["group_maps"].shape[0] == 0 for sample in (recipient, donor)):
        raise ValueError("CutMix requires paired RGB images and at least one group-map level")
    if any(recipient[key].shape != donor[key].shape for key in keys):
        raise ValueError("CutMix requires matching paired image and group-map grids")
    height, width = recipient["lq"].shape[-2:]
    if any(sample[key].shape[-2:] != (height, width) for sample in (recipient, donor) for key in keys):
        raise ValueError("CutMix images and all-level group maps must use the same grid")
    if len(bbox) != 4 or any(isinstance(value, bool) or not isinstance(value, Integral) for value in bbox):
        raise ValueError("CutMix bbox must contain four integers, excluding bool")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 <= x1 <= width and 0 <= y0 <= y1 <= height):
        raise ValueError("CutMix bbox must lie within the paired image grid")
    mixed = dict(recipient)
    for key in keys:
        mixed[key] = recipient[key].clone()
        mixed[key][:, y0:y1, x0:x1] = donor[key][:, y0:y1, x0:x1]
    provenance = ("sample_id", "hierarchy_row", "crop_origin", "crop_size", "original_size")
    mixed["cutmix"] = {
        "bbox": tuple(bbox),
        "recipient": {key: recipient[key] for key in provenance},
        "donor": {key: donor[key] for key in provenance},
    }
    return mixed


def read_hierarchy(directory, records):
    hierarchy = json.loads((directory / "hierarchy.json").read_text(encoding="utf-8"))
    if hierarchy.get("completed") is not True or hierarchy["p"] != 512:
        raise ValueError("A completed p=512 hierarchy artifact is required")
    depth, levels = hierarchy["depth"], hierarchy["L"]
    sizes = [2 ** level for level in range(depth + 1)]
    parents = [[]] + [[group // 2 for group in range(size)] for size in sizes[1:]]
    leaves = 2 ** depth
    ancestors = [[leaf // 2 ** (depth - level) for leaf in range(leaves)] for level in range(levels)]
    if (levels != depth + 1 or hierarchy["level_sizes"] != sizes or
            hierarchy["parents_by_level"] != parents or hierarchy["leaf_to_group"] != ancestors or
            hierarchy["expert_offsets"] != [0] + [2 ** level - 2 for level in range(1, levels)]):
        raise ValueError("Hierarchy mapping must match the complete binary tree")
    with (directory / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    ids = [row["sample_id"] for row in samples]
    if (len(samples) != hierarchy["N"] or len(set(ids)) != len(ids) or
            any(not sample.strip() for sample in ids) or
            any(int(row["row"]) != index for index, row in enumerate(samples))):
        raise ValueError("Hierarchy samples require unique sample_id and contiguous row mapping")
    by_id = {row["sample_id"]: row for row in records}
    if set(by_id) != set(ids):
        raise ValueError("Pair CSV and hierarchy sample_id sets must match exactly")
    ordered = []
    for index, sample in enumerate(samples):
        row = dict(by_id[sample["sample_id"]])
        if row["lq_path"] != sample["relative_path"]:
            raise ValueError("lq_path differs from hierarchy relative_path for " + row["sample_id"])
        row["_index"] = index
        ordered.append(row)
    labels = np.load(directory / "labels.npy", allow_pickle=False)
    if (labels.shape != (len(samples), levels) or not np.issubdtype(labels.dtype, np.integer) or
            np.any(labels[:, 0] != 0) or np.any(labels[:, 1:] // 2 != labels[:, :-1])):
        raise ValueError("Hierarchy labels disagree with sample rows or parent ancestry")
    nodes = hierarchy["nodes"]
    if len(nodes) != 2 ** levels - 1:
        raise ValueError("Hierarchy must contain every binary heap node")
    counts = []
    for level, size in enumerate(sizes):
        if np.any(labels[:, level] < 0) or np.any(labels[:, level] >= size):
            raise ValueError("Hierarchy group labels are outside their level")
        counts.append(np.bincount(labels[:, level], minlength=size))
        for group in range(size):
            heap = 2 ** level - 1 + group
            node = nodes[heap]
            children = [] if level == depth else [2 * heap + 1, 2 * heap + 2]
            if (node["heap_id"] != heap or node["level"] != level or node["group"] != group or
                    node["expert_id"] != heap - 1 or node["parent"] != (heap - 1) // 2 or
                    node["children"] != children or node["center_row"] != heap or
                    node["sample_count"] != int(counts[level][group]) or counts[level][group] == 0):
                raise ValueError("Hierarchy node mapping/counts disagree at heap " + str(heap))
    hashes = {name: file_sha256(directory / name) for name in
              ("hierarchy.json", "samples.csv", "labels.npy", "centers.npy", "raw_centers.npy", "source_metadata.json")}
    return hierarchy, labels, ordered, hashes
