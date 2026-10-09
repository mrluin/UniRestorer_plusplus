import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from basicsr.models.archs.unirestorerpp.dense_prior import DenseResNet18Encoder
from basicsr.dr_training.text import file_sha256, load_local_checkpoint


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT; use an existing 512-D D checkpoint.
CONFIG = {
    "checkpoint": "checkpoints/densedrD.pt",
    "checkpoint_key": "dr_encoder",
    "checkpoint_prefix": "",
    "image_root": "data/inputs",
    "recursive": True,
    "image_list": None,  # Optional file of image_root-relative paths, in desired order.
    "output_dir": "outputs/dr_features",
    "save_dense": True,
    "save_normalized": True,  # Save raw GAP and its L2-normalized representation.
    "cpu_threads": 1,
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SAMPLE_FIELDS = ["row", "sample_id", "relative_path", "image_h", "image_w",
                 "dense_h", "dense_w", "dense_path"]


def extract_dr(options: dict, base_dir=Path(".")) -> dict:
    """Write a new feature directory; relative options resolve against base_dir."""
    base_dir = Path(base_dir).resolve()
    image_root = (base_dir / options["image_root"]).resolve()
    output_dir = (base_dir / options["output_dir"]).resolve()
    if not options.get("checkpoint"):
        raise ValueError("checkpoint is required; random D weights are not extracted")
    checkpoint = (base_dir / options["checkpoint"]).resolve()
    checkpoint_key = options.get("checkpoint_key")
    checkpoint_prefix = options.get("checkpoint_prefix", "")
    if options.get("mean") is not None or options.get("std") is not None:
        raise NotImplementedError("D uses RGB/255; extra mean/std preprocessing is unsupported")

    if options.get("image_list"):
        image_list = (base_dir / options["image_list"]).resolve()
        names = [line.strip() for line in image_list.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        images = [(image_root / name).resolve() for name in names]
        sample_order = "image_list"
    else:
        candidates = image_root.rglob("*") if options.get("recursive", True) else image_root.glob("*")
        images = sorted((path.resolve() for path in candidates
                         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                        key=lambda path: path.relative_to(image_root).as_posix())
        sample_order = "relative_path_lexicographic"
    relative_paths = [image.relative_to(image_root).as_posix() for image in images]
    if not images:
        raise FileNotFoundError("no images found under image_root")
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("duplicate image paths would produce duplicate sample_id values")

    state = load_local_checkpoint(checkpoint, checkpoint_key, checkpoint_prefix)
    checkpoint_sha = file_sha256(checkpoint)
    encoder = DenseResNet18Encoder(p=512, output_stride=8).cpu().eval()
    encoder.load_state_dict(state, strict=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    save_dense = options.get("save_dense", True)
    save_normalized = options.get("save_normalized", True)
    if save_dense:
        (output_dir / "dense").mkdir()
    features = np.lib.format.open_memmap(output_dir / "features.npy", mode="w+",
                                         dtype=np.float32, shape=(len(images), 512))
    normalized = None
    try:
        if save_normalized:
            normalized = np.lib.format.open_memmap(output_dir / "features_l2.npy", mode="w+",
                                                  dtype=np.float32, shape=(len(images), 512))
        with (output_dir / "samples.csv").open("w", encoding="utf-8", newline="") as stream, torch.no_grad():
            writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
            writer.writeheader()
            for row, (image_path, relative_path) in enumerate(zip(images, relative_paths)):
                with Image.open(image_path) as image:
                    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                image_h, image_w = pixels.shape[:2]
                image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
                # Process original sizes separately: padding would change GAP(E).
                dense = encoder(image_tensor.unsqueeze(0))[0]
                vector = dense.mean(dim=(-2, -1))
                if not torch.isfinite(vector).all():
                    raise ValueError(f"non-finite GAP vector for {relative_path}")
                features[row] = vector.numpy()
                if save_normalized:
                    norm = vector.norm(p=2)
                    if not torch.isfinite(norm) or norm == 0:
                        raise ValueError(f"cannot L2-normalize GAP vector for {relative_path}: norm={norm.item()}")
                    normalized[row] = (vector / norm).numpy()
                dense_path = "dense/{:08d}.npy".format(row) if save_dense else ""
                if save_dense:
                    np.save(output_dir / dense_path, dense.numpy(), allow_pickle=False)
                writer.writerow({"row": row, "sample_id": relative_path, "relative_path": relative_path,
                                 "image_h": image_h, "image_w": image_w,
                                 "dense_h": dense.shape[-2], "dense_w": dense.shape[-1],
                                 "dense_path": dense_path})
        features.flush()
        if normalized is not None:
            normalized.flush()
    finally:
        features._mmap.close()
        if normalized is not None:
            normalized._mmap.close()
    model_source = Path(inspect.getfile(DenseResNet18Encoder))
    metadata = {
        "schema_version": 1, "completed": True, "num_samples": len(images),
        "feature_shape": [len(images), 512], "feature_dtype": "float32",
        "p": 512, "output_stride": 8, "aggregation": "GAP",
        "save_dense": save_dense, "save_normalized": save_normalized, "sample_order": sample_order,
        "feature_spaces": {"raw": {"file": "features.npy", "normalization": "none"},
                           "l2": ({"file": "features_l2.npy", "normalization": "L2 after GAP",
                                   "zero_vector_policy": "error"} if save_normalized else None)},
        "preprocessing": {"color": "RGB", "input_range": [0.0, 1.0],
                          "resize": None, "padding": None, "mean": None, "std": None},
        "image_root": str(options["image_root"]),
        "image_list": str(options["image_list"]) if options.get("image_list") else None,
        "model": {"class": "DenseResNet18Encoder", "source": "basicsr/models/archs/unirestorerpp/dense_prior.py",
                  "sha256": file_sha256(model_source), "output": "E = D(image), before phi_mu/phi_rawvar"},
        "checkpoint": {"path": str(options["checkpoint"]), "key": checkpoint_key,
                       "prefix": checkpoint_prefix, "sha256": checkpoint_sha,
                       "training_status": "not_inferred"},
        "device": "cpu", "cpu_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
    }
    # This completion record is written only after all feature/dense/CSV rows succeed.
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output_dir": str(output_dir), "num_samples": len(images),
            "feature_shape": [len(images), 512], "save_dense": save_dense,
            "normalized_features": "features_l2.npy" if save_normalized else None,
            "features": "features.npy", "samples": "samples.csv", "metadata": "metadata.json"}


def main():
    parser = argparse.ArgumentParser(description="Extract D features using CONFIG at the top of this file.")
    parser.parse_args()
    torch.set_num_threads(int(CONFIG.get("cpu_threads", 1)))
    print(json.dumps(extract_dr(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
