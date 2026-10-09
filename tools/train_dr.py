import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from basicsr.dr_training import DenseCaptionCSVDataset, unique_caption_pixel_ce
from basicsr.dr_training.text import file_sha256, load_local_checkpoint
from basicsr.models.archs.unirestorerpp.dense_prior import DenseResNet18Encoder


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT. Fill the None run settings before training.
CONFIG = {
    "output_dir": "outputs/dr_training",
    "resume": None,
    "text": {
        "mode": "precomputed",
        "freeze": True,
        "features": "outputs/caption_bank/caption_bank.npy",
        "captions": "outputs/caption_bank/captions.csv",
        "class_mapping": "outputs/caption_bank/class_mapping.csv",
        "source_metadata": "outputs/caption_bank/metadata.json",
        "source": None,
    },
    "data": {
        "metadata_csv": "data/dr_samples.csv",
        "image_root": "data/inputs",
        "crop_size": None,
        "cutmix_probability": None,
        "cutmix_alpha": None,
        "label_alignment": "nearest",
    },
    "model": {
        "initialization": None,  # random or checkpoint
        "checkpoint": None,
        "checkpoint_key": None,
        "checkpoint_prefix": "",
        "bn_mode": None,  # train or frozen_stats
    },
    "loss": {
        "type": "caption_ce",
        "temperature": None,
        "chunk_size": None,
    },
    "optimizer": {"type": None, "lr": None, "weight_decay": 0.0},
    "scheduler": {"type": None},  # none or StepLR; StepLR also needs step_size and gamma.
    "run": {
        "epochs": None,
        "max_steps": None,
        "batch_size": None,
        "shuffle": None,
        "seed": None,
        "cpu_threads": 1,
    },
}


def train_dr(options: dict, base_dir=Path(".")) -> dict:
    """Train D on CPU and save a new checkpoint directory."""
    recipe = json.loads(json.dumps(options))
    base_dir = Path(base_dir).resolve()
    text, data, model = options["text"], options["data"], options["model"]
    loss_options, run = options["loss"], options["run"]
    if text.get("mode", "precomputed") != "precomputed" or text.get("freeze", True) is not True:
        raise ValueError("text.mode must be 'precomputed' and text.freeze must be true")
    source_metadata_path = (base_dir / text["source_metadata"]
                            if text.get("source_metadata") else None)
    source_metadata = (json.loads(source_metadata_path.read_text(encoding="utf-8"))
                       if source_metadata_path else None)
    text_source = text.get("source")
    if source_metadata is not None:
        if (source_metadata.get("completed") is not True or source_metadata.get("p") != 512
                or source_metadata.get("normalization") != "none"):
            raise ValueError("a completed raw512 caption-bank artifact is required")
        if text_source is None:
            text_source = source_metadata["encoder"]
    if text_source is None:
        raise ValueError("text.source or generated text.source_metadata is required")
    loss_type = loss_options.get("type", "caption_ce")
    if loss_type != "caption_ce":
        raise ValueError("loss.type must be 'caption_ce': caption-class CE is the approved supervision")
    temperature = float(loss_options["temperature"])
    if data.get("label_alignment", "nearest") != "nearest":
        raise ValueError("data.label_alignment must be 'nearest'")
    if data.get("mean") is not None or data.get("std") is not None:
        raise NotImplementedError("D uses RGB/255; extra mean/std preprocessing is unsupported")
    initialization = model["initialization"]
    if initialization not in ("random", "checkpoint"):
        raise ValueError("model.initialization must be 'random' or 'checkpoint'")
    bn_mode = model["bn_mode"]
    if bn_mode not in ("train", "frozen_stats"):
        raise ValueError("model.bn_mode must be 'train' or 'frozen_stats'")
    epochs, seed = int(run["epochs"]), int(run["seed"])
    max_steps = run.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    torch.set_num_threads(int(run["cpu_threads"]))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    bank_path, captions_path = base_dir / text["features"], base_dir / text["captions"]
    bank_array = np.load(bank_path, allow_pickle=False)
    if bank_array.ndim != 2 or bank_array.shape[1] != 512:
        raise ValueError("precomputed text features must be C x 512; no dimension bridge is used")
    caption_bank = torch.from_numpy(np.asarray(bank_array, dtype=np.float32))
    bank_norms = caption_bank.norm(dim=1)
    if not torch.isfinite(bank_norms).all() or (bank_norms == 0).any():
        raise ValueError("caption vectors must be finite and nonzero for the contrastive loss")
    with captions_path.open(encoding="utf-8-sig", newline="") as stream:
        captions = [{"caption_id": int(row["caption_id"]), "caption": row["caption"]}
                    for row in csv.DictReader(stream)]
    id_to_row = {row["caption_id"]: index for index, row in enumerate(captions)}
    if len(captions) != len(caption_bank) or len(id_to_row) != len(captions):
        raise ValueError("caption CSV rows must uniquely bind every caption-bank row")
    mapping_path = base_dir / text["class_mapping"] if text.get("class_mapping") else None
    if source_metadata is not None:
        if mapping_path is None:
            mapping_path = source_metadata_path.parent / source_metadata["files"]["class_mapping"]
        for key, path in (("features", bank_path), ("captions", captions_path),
                          ("class_mapping", mapping_path)):
            if file_sha256(path) != source_metadata["file_sha256"][key]:
                raise ValueError(f"{key} does not match the prepared caption-bank metadata")
    class_mapping, caption_id_mapping = [], {}
    if mapping_path is not None:
        with mapping_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                alias, canonical, bank_row = (int(row[key]) for key in
                                               ("input_caption_id", "caption_id", "bank_row"))
                if id_to_row.get(canonical) != bank_row or row["caption"] != captions[bank_row]["caption"]:
                    raise ValueError("caption aliases must bind to the canonical caption-bank row")
                if alias in caption_id_mapping and caption_id_mapping[alias] != canonical:
                    raise ValueError("one input caption ID cannot refer to two classes")
                if alias in id_to_row and alias != canonical:
                    raise ValueError("a canonical caption ID cannot be an alias of a different class")
                caption_id_mapping[alias] = canonical
                class_mapping.append(dict(input_caption_id=alias, caption_id=canonical, bank_row=bank_row))
    dataset = DenseCaptionCSVDataset(
        base_dir / data["metadata_csv"], image_root=base_dir / data["image_root"],
        crop_size=data.get("crop_size"),
        cutmix_probability=data["cutmix_probability"], cutmix_alpha=data["cutmix_alpha"],
        caption_id_mapping=caption_id_mapping or None,
    )
    if not len(dataset) or set(dataset.caption_ids) - id_to_row.keys():
        raise ValueError("every training sample must bind to the caption CSV")

    encoder = DenseResNet18Encoder(p=512, output_stride=8).cpu()
    resume_path = base_dir / options["resume"] if options.get("resume") else None
    resumed = load_local_checkpoint(resume_path) if resume_path else None
    initialization_record = {"type": initialization}
    if resume_path is not None and initialization != "checkpoint":
        raise ValueError("resume requires model.initialization='checkpoint'")
    if initialization == "checkpoint":
        if resume_path is not None:
            checkpoint, checkpoint_name, key = resume_path, options["resume"], "dr_encoder"
            state = resumed[key]
        else:
            checkpoint, checkpoint_name = base_dir / model["checkpoint"], model["checkpoint"]
            key, prefix = model.get("checkpoint_key"), model.get("checkpoint_prefix", "")
            state = load_local_checkpoint(checkpoint, key, prefix)
        encoder.load_state_dict(state, strict=True)
        initialization_record.update(path=str(checkpoint_name), sha256=file_sha256(checkpoint), key=key)
        if resume_path is None:
            initialization_record["prefix"] = prefix

    optimizer_options = dict(options["optimizer"])
    optimizer_type = optimizer_options.pop("type")
    optimizer = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[optimizer_type](
        encoder.parameters(), lr=optimizer_options.pop("lr"), **optimizer_options)
    scheduler_options = dict(options["scheduler"])
    scheduler_type = scheduler_options.pop("type")
    scheduler = None if scheduler_type == "none" else {"StepLR": torch.optim.lr_scheduler.StepLR}[scheduler_type](
        optimizer, step_size=scheduler_options.pop("step_size"), gamma=scheduler_options.pop("gamma"),
        **scheduler_options)
    epoch, batch_in_epoch, step = 0, 0, 0
    if resumed is not None:
        if resumed["recipe"]["optimizer"]["type"] != optimizer_type:
            raise ValueError("resume optimizer type must match the checkpoint")
        if resumed["recipe"]["scheduler"]["type"] != scheduler_type:
            raise ValueError("resume scheduler type must match the checkpoint")
        optimizer.load_state_dict(resumed["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resumed["scheduler"])
        epoch, batch_in_epoch, step = (int(resumed["state"][key]) for key in ("epoch", "batch_in_epoch", "step"))
    if epoch >= epochs or (max_steps is not None and step >= max_steps):
        raise ValueError("the requested training limit must allow at least one new step")
    encoder.train()
    if bn_mode == "frozen_stats":
        for module in encoder.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()  # Running statistics frozen; affine parameters remain trainable.

    output_dir = (base_dir / options["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    first_step = step
    fields = ["step", "epoch", "batch", "loss", "grad_norm", "positions", "mixed_images"]
    loss_kwargs = {"temperature": temperature, "chunk_size": loss_options.get("chunk_size")}
    with (output_dir / "metrics.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while epoch < epochs and (max_steps is None or step < max_steps):
            generator = torch.Generator().manual_seed(seed + epoch)
            if run["shuffle"] is None:
                raise ValueError("shuffle must be configured")
            loader = DataLoader(dataset, batch_size=int(run["batch_size"]),
                                shuffle=run["shuffle"], num_workers=0, generator=generator)
            if batch_in_epoch >= len(loader):
                raise ValueError("resume batch position is incompatible with this data/batch_size")
            for batch_index, batch in enumerate(loader):
                if batch_index < batch_in_epoch:
                    continue
                optimizer.zero_grad(set_to_none=True)
                dense = encoder(batch["image"])
                # Remap integer IDs before resizing to avoid float rounding of large IDs.
                labels = torch.empty_like(batch["caption_id_map"])
                for caption_id in batch["caption_id_map"].unique().tolist():
                    labels[batch["caption_id_map"] == caption_id] = id_to_row[caption_id]
                labels = F.interpolate(labels[:, None].float(), size=dense.shape[-2:],
                                       mode="nearest")[:, 0].long()
                loss = unique_caption_pixel_ce(dense, caption_bank, labels, **loss_kwargs)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss; no completed training artifact is saved")
                loss.backward()
                grads = [parameter.grad for parameter in encoder.parameters() if parameter.grad is not None]
                grad_norm = float(sum(gradient.square().sum() for gradient in grads).sqrt())
                if not np.isfinite(grad_norm):
                    raise RuntimeError("non-finite gradient norm; no completed training artifact is saved")
                optimizer.step()
                step += 1
                batch_in_epoch = batch_index + 1
                last_loss = float(loss.detach())
                writer.writerow({"step": step, "epoch": epoch, "batch": batch_index,
                                 "loss": last_loss, "grad_norm": grad_norm,
                                 "positions": labels.numel(), "mixed_images": int(batch["mixed"].sum())})
                if max_steps is not None and step >= max_steps:
                    break
            if batch_in_epoch == len(loader):
                epoch += 1
                batch_in_epoch = 0
                if scheduler is not None:
                    scheduler.step()

    state = {"epoch": epoch, "batch_in_epoch": batch_in_epoch, "step": step}
    artifact = {"dr_encoder": encoder.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "state": state, "recipe": recipe, "rng_state_restored": False}
    temporary = output_dir / "checkpoint.pt.tmp"
    torch.save(artifact, temporary)
    os.replace(temporary, output_dir / "checkpoint.pt")
    metadata = {
        "schema_version": 1, "completed": True, "device": "cpu", "p": 512, "output_stride": 8,
        "model": {"class": "DenseResNet18Encoder", "output": "raw E = D(image)",
                  "source_sha256": file_sha256(inspect.getfile(DenseResNet18Encoder))},
        "preprocessing": {"color": "RGB", "input_range": [0.0, 1.0], "resize": None,
                          "crop_size": data.get("crop_size"), "mean": None, "std": None},
        "text": {"mode": "precomputed", "frozen": True, "dimension": 512, "source": text_source,
                 "bank_sha256": file_sha256(bank_path), "captions_sha256": file_sha256(captions_path),
                 "caption_rows": captions, "text_encoder_implemented": True, "dimension_bridge": None,
                 "encoder_execution_in_this_training_run": False,
                 "source_metadata": source_metadata,
                 "source_metadata_sha256": file_sha256(source_metadata_path) if source_metadata_path else None,
                 "class_mapping": class_mapping,
                 "class_mapping_sha256": file_sha256(mapping_path) if mapping_path else None,
                 "alias_mapping_applied_before_cutmix": bool(caption_id_mapping)},
        "contrastive": {"loss": loss_type, "normalization": "L2 on contrastive vectors only",
                        "temperature": temperature, "temperature_trainable": False,
                        "label_alignment": "nearest", "pixel_sampling": None},
        "bn_mode": bn_mode, "bn_affine_trainable": True, "initialization": initialization_record,
        "optimizer": {"type": optimizer_type,
                      "effective_param_groups": [{key: value for key, value in group.items() if key != "params"}
                                                 for group in optimizer.param_groups]},
        "scheduler": {"type": scheduler_type, "effective_state": scheduler.state_dict() if scheduler else None},
        "recipe": recipe, "paper_local_supervision_approved": True,
        "training_quality_inferred": False,
        "resume": {"path": str(options["resume"]) if resume_path else None,
                   "rng_state_restored": False, "epoch_permutation_seed": "seed + epoch",
                   "optimizer_scheduler_precedence": "checkpoint state overrides requested recipe" if resume_path else None},
        "state": state, "steps_this_run": step - first_step, "last_loss": last_loss,
        "checkpoint": "checkpoint.pt", "checkpoint_sha256": file_sha256(output_dir / "checkpoint.pt"),
        "torch_version": str(torch.__version__),
    }
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output_dir": str(output_dir), "checkpoint": "checkpoint.pt", "metadata": "metadata.json",
            "step": step, "steps_this_run": step - first_step, "last_loss": last_loss}


def main():
    parser = argparse.ArgumentParser(description="Train D using CONFIG at the top of this file.")
    parser.parse_args()
    print(json.dumps(train_dr(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
