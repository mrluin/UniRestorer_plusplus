from copy import deepcopy
import csv
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from basicsr.models.archs.unirestorerpp.lora import LoRAConv2dResidual
from basicsr.models.archs.unirestorerpp.degradation_space import DegradationSpace
from basicsr.models.archs.unirestorerpp.network import UniRestorerPP
from .data import (MatchedHierarchyPairedDataset, binding, binding_sha256,
                   file_sha256, paired_cutmix, read_hierarchy, read_pairs)
from .pipeline import (_check_resume, _effective, _load, _metadata,
                       _optimization, _save, _settings, _state)
from .routing_losses import joint_restoration_loss


STAGES = ("der_initialization", "joint_finetuning")


def configure_stage_training(model, stage):
    """Freeze D/base and select the explicitly approved stage parameters.

    Joint base operations retain input gradients; setting requires_grad=False
    is sufficient. Keeping all expert/base copies in eval also fixes buffers.
    """
    if stage not in STAGES:
        raise ValueError("Unknown routing stage: " + str(stage))
    model.eval().requires_grad_(False)
    prior = model.degradation_prior
    prior.set_freeze_dr(True)
    for head in (prior.phi_mu, prior.phi_rawvar):
        head.requires_grad_(True).train()
    for _, block in model.routed_blocks():
        for branch in (block.router.der_prior, block.router.der_feature):
            branch.requires_grad_(True).train()
        if stage == "joint_finetuning":
            # Enable the already approved training noise only in the routers.
            block.router.train()
            for branch in (block.router.ger_prior, block.router.ger_feature):
                branch.requires_grad_(True)
    if stage == "joint_finetuning":
        for module in model.modules():
            if isinstance(module, LoRAConv2dResidual):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _trainable_names(model):
    return [name for name, value in model.named_parameters() if value.requires_grad]


def frozen_state_sha256(model):
    """Hash fixed parameters and all buffers using unique owner names."""
    values = [(name, value) for name, value in model.named_parameters() if not value.requires_grad]
    values.extend(model.named_buffers())
    digest = hashlib.sha256()
    for name, value in sorted(values):
        tensor = value.detach().cpu().contiguous()
        digest.update((name + str(tuple(tensor.shape)) + str(tensor.dtype)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _inactive_variance_sha256(model):
    digest = hashlib.sha256()
    for name, value in model.degradation_prior.phi_rawvar.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update((name + str(tuple(tensor.shape)) + str(tensor.dtype)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_routing_stage_checkpoint(path, *, expected_stage=None, hierarchy=None, artifact_sha256=None):
    """Load a local full-state assembly/stage without replacing shared owners."""
    payload = _load(path)
    if payload.get("completed") is not True or payload.get("artifact_type") not in (
            "assembled_cluster_lora",) + STAGES:
        raise ValueError("A completed assembled restoration or routing-stage checkpoint is required")
    if expected_stage is not None and payload["artifact_type"] != expected_stage:
        raise ValueError("Checkpoint has a different stage identity")
    if hierarchy is not None and payload.get("hierarchy") != hierarchy:
        raise ValueError("Checkpoint has a different hierarchy")
    if artifact_sha256 is not None and payload.get("artifact_sha256") != artifact_sha256:
        raise ValueError("Checkpoint has different hierarchy artifact hashes")
    tree = payload["hierarchy"]
    config = deepcopy(payload["network_config"])
    if config["tree_depth"] != tree["depth"] or config.get("p") != 512:
        raise ValueError("Network configuration differs from the p=512 hierarchy")
    if any(config.get(key) != value for key, value in payload["base_config"].items()):
        raise ValueError("Network and base configurations differ")
    dr_sha = tree.get("source_metadata", {}).get("checkpoint", {}).get("sha256")
    if not isinstance(dr_sha, str) or not dr_sha or dr_sha != payload.get("dr_checkpoint_sha256"):
        raise ValueError("D checkpoint SHA differs from the hierarchy extraction checkpoint")
    leaf_base = 2 ** tree["depth"] - 1
    leaf_heaps = set(range(leaf_base, 2 ** (tree["depth"] + 1) - 1)) - {0}
    if not leaf_heaps.issubset(set(payload["trained_heap_ids"])):
        raise ValueError("DER initialization requires all finest leaf LoRA checkpoints")
    config["freeze_dr"] = True
    model = UniRestorerPP(**config).cpu().eval()
    model.load_state_dict(payload["params"], strict=True)
    model.degradation_prior.set_freeze_dr(True)
    return model, payload


def der_ce_loss(aux):
    """Each block's valid-pixel CE mean, then an equal mean over blocks."""
    if not aux["der_logits"]:
        raise ValueError("DER initialization requires at least one restoration block")
    losses = []
    for key, logits in aux["der_logits"].items():
        labels, valid = aux["label_maps"][key], aux["valid_maps"][key]
        if labels.shape != valid.shape or labels.shape != (logits.shape[0],) + logits.shape[-2:]:
            raise ValueError("DER logits, labels and validity grids differ")
        if not bool(valid.any()):
            raise ValueError("DER block has no valid supervised locations")
        losses.append(F.cross_entropy(logits, labels, reduction="none")[valid].mean())
    return torch.stack(losses).mean()


def _cutmix_settings(options, records, settings):
    config = deepcopy(options.get("cutmix", {"enabled": False}))
    config.setdefault("enabled", False)
    config.setdefault("box", None)
    config.setdefault("donor_shift", None)
    if not isinstance(config["enabled"], bool):
        raise ValueError("cutmix.enabled must be a bool")
    if config["enabled"]:
        size = settings["run"]["batch_size"]
        shift, box = config["donor_shift"], config["box"]
        if (size < 2 or len(records) % size or isinstance(shift, bool) or
                not isinstance(shift, int) or shift % size == 0):
            raise ValueError("Explicit CutMix needs complete batches >=2 and a nonzero donor_shift modulo batch_size")
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(value, bool) or not isinstance(value, int) for value in box)):
            raise ValueError("Configure the explicit CutMix box (x0,y0,x1,y1)")
        crop = settings["run"]["crop_size"]
        grids = ([tuple([crop, crop]) if isinstance(crop, int) else tuple(crop)]
                 if crop is not None else [row["_size"] for row in records])
        x0, y0, x1, y1 = box
        if any(not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) for height, width in grids):
            raise ValueError("CutMix box exceeds an input grid")
        config["box"] = list(box)
    return config


def _mix_batch(batch, config):
    if not config["enabled"]:
        return batch
    count = len(batch["lq"])
    mixed = []
    for index in range(count):
        donor = (index + config["donor_shift"]) % count
        def sample(row):
            item = {key: batch[key][row] for key in ("lq", "gt", "group_maps", "sample_id")}
            item["hierarchy_row"] = int(batch["hierarchy_row"][row])
            for key in ("crop_origin", "crop_size", "original_size"):
                # default_collate transposes an integer coordinate tuple.
                item[key] = tuple(int(column[row]) for column in batch[key])
            return item
        mixed.append(paired_cutmix(sample(index), sample(donor), config["box"]))
    return {key: torch.stack([sample[key] for sample in mixed]) for key in ("lq", "gt", "group_maps")}


def _train_stage(model, dataset, parameters, optimizer, scheduler, settings, metrics_path, resume, joint_space=None):
    run = settings["run"]
    progress = dict(epoch=0, batch_in_epoch=0, step=0) if resume is None else dict(resume["progress"])
    if resume is not None:
        torch.set_rng_state(resume["torch_rng_state"])
    first_step = progress["step"]
    last_loss = resume.get("last_loss") if resume is not None else None
    last_losses = resume.get("last_losses") if resume is not None else None
    last_per_block = resume.get("last_per_block") if resume is not None else None
    with metrics_path.open("x", encoding="utf-8", newline="") as stream:
        fields = ("step", "epoch", "batch", "loss", "grad_norm", "lr")
        if joint_space is not None:
            fields += ("l1", "dg", "aux")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while progress["epoch"] < run["epochs"] and (run["max_steps"] is None or progress["step"] < run["max_steps"]):
            epoch = progress["epoch"]
            dataset.epoch = epoch
            loader = DataLoader(dataset, batch_size=run["batch_size"], shuffle=run["shuffle"], num_workers=0,
                                generator=torch.Generator().manual_seed(run["seed"] + epoch))
            for batch_index, batch in enumerate(loader):
                if batch_index < progress["batch_in_epoch"]:
                    continue
                batch = _mix_batch(batch, settings["cutmix"])
                optimizer.zero_grad(set_to_none=True)
                if joint_space is None:
                    _, aux = model.forward_der_initialization(batch["lq"], batch["group_maps"][:, -1])
                    loss = der_ce_loss(aux)
                else:
                    restored, aux = model(batch["lq"], return_aux=True)
                    loss, details = joint_restoration_loss(restored, batch["gt"], aux,
                                                           joint_space, batch["group_maps"][:, -1])
                    last_losses = {key: float(details[key].detach()) for key in ("l1", "dg", "aux")}
                    last_per_block = {key: {name: float(value.detach()) for name, value in terms.items()}
                                      for key, terms in details["per_block"].items()}
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite routing-stage loss; no completion metadata is written")
                loss.backward()
                gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
                if not gradients or any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise RuntimeError("Missing or non-finite routing-stage gradients; no completion metadata is written")
                grad_norm = torch.stack([gradient.square().sum() for gradient in gradients]).sum().sqrt()
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("Non-finite routing-stage gradient norm; no completion metadata is written")
                optimizer.step()
                progress.update(batch_in_epoch=batch_index + 1, step=progress["step"] + 1)
                last_loss = float(loss.detach())
                row = dict(step=progress["step"], epoch=epoch, batch=batch_index, loss=last_loss,
                           grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"])
                if joint_space is not None:
                    row.update(last_losses)
                writer.writerow(row)
                if run["max_steps"] is not None and progress["step"] >= run["max_steps"]:
                    break
            if progress["batch_in_epoch"] == len(loader):
                progress.update(epoch=epoch + 1, batch_in_epoch=0)
                if scheduler is not None:
                    scheduler.step()
    result = dict(progress=progress, last_loss=last_loss, steps_this_run=progress["step"] - first_step,
                  torch_rng_state=torch.get_rng_state())
    if joint_space is not None:
        result.update(last_losses=last_losses, last_per_block=last_per_block)
    return result


def train_der_initialization(options, base_dir=Path(".")):
    """True-leaf teacher trajectory, CE only; CPU/local artifacts only."""
    settings = _settings(options)
    base_dir = Path(base_dir).resolve()
    output = base_dir / options["output_dir"]
    if output.exists():
        raise FileExistsError("Training output directory already exists: " + str(output))
    torch.set_num_threads(1)
    torch.manual_seed(settings["run"]["seed"])
    records = read_pairs(options, base_dir)
    hierarchy, labels, ordered, hashes = read_hierarchy(base_dir / options["hierarchy_dir"], records)
    run = settings["run"]
    if run["crop_size"] is None and run["batch_size"] > 1 and len({row["_size"] for row in records}) > 1:
        raise ValueError("Batching unequal grids requires crop_size or batch_size=1")
    settings["cutmix"] = _cutmix_settings(options, records, settings)
    dataset = MatchedHierarchyPairedDataset(ordered, labels, seed=run["seed"], crop_size=run["crop_size"])
    source_path = base_dir / options["assembled_checkpoint"]
    model, source = load_routing_stage_checkpoint(source_path, expected_stage="assembled_cluster_lora",
                                                  hierarchy=hierarchy, artifact_sha256=hashes)
    if source["pair_binding"] != binding(records):
        raise ValueError("Assembly pair_binding differs from the supplied pair CSV")
    parameters = configure_stage_training(model, "der_initialization")
    frozen_sha = frozen_state_sha256(model)
    variance_sha = _inactive_variance_sha256(model)
    identity = dict(source_checkpoint_sha256=file_sha256(source_path), hierarchy=hierarchy,
                    artifact_sha256=hashes, pair_binding=binding(records), pair_binding_sha256=binding_sha256(records),
                    network_config=dict(source["network_config"], freeze_dr=True),
                    trainable_parameter_names=_trainable_names(model), frozen_state_sha256=frozen_sha,
                    inactive_variance_sha256=variance_sha)
    resume = None
    parent_path = source_path
    if options.get("resume"):
        parent_path = base_dir / options["resume"]
        resume = _load(parent_path, "der_initialization")
        _check_resume(resume, identity, settings, records)
        model.load_state_dict(resume["params"], strict=True)
        if frozen_state_sha256(model) != frozen_sha:
            raise ValueError("Resume changes a fixed parameter or buffer")
        if _inactive_variance_sha256(model) != variance_sha:
            raise ValueError("Resume changes the variance head that has no DER CE path")
    optimizer, scheduler, parameters = _optimization(parameters, settings, resume)
    output.mkdir(parents=True, exist_ok=False)
    result = _train_stage(model, dataset, parameters, optimizer, scheduler, settings, output / "metrics.csv", resume)
    if frozen_state_sha256(model) != frozen_sha:
        raise RuntimeError("Fixed parameters or buffers changed; no completion metadata is written")
    if _inactive_variance_sha256(model) != variance_sha:
        raise RuntimeError("Variance changed without a CE gradient; no completion metadata is written")
    stages = deepcopy(source["stage_status"])
    stages.update(der="initialized with true-leaf teacher CE", ger="untrained",
                  phi_mu="DER CE updates enabled", phi_variance="trainable; no DER CE gradient; unchanged",
                  routers="DER initialized; GER untrained", phi="mean CE updates; variance unchanged")
    inherited = {key: deepcopy(source[key]) for key in ("base_config", "base_checkpoint_sha256", "trained_heap_ids",
                 "untrained_heap_ids", "dr_checkpoint_sha256", "adapter_checkpoints")}
    payload = dict(schema_version=1, artifact_type="der_initialization", completed=True,
                   params=_state(model), **inherited, **identity, parent_checkpoint_sha256=file_sha256(parent_path),
                   recipe=settings, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                   optimizer_parameter_count=len(parameters), trainable_parameter_count=sum(p.numel() for p in parameters),
                   effective_options=_effective(optimizer, scheduler), stage_status=stages,
                   loss="valid-pixel CE mean per block; equal mean over blocks; no DG/GER loss",
                   variance_gradient_expected=False, training_quality_inferred=False, **result)
    _save(output / "checkpoint.pt", payload)
    metadata = {key: value for key, value in payload.items()
                if key not in ("params", "optimizer", "scheduler", "torch_rng_state", "schema_version", "completed")}
    metadata.update(checkpoint="checkpoint.pt", checkpoint_sha256=file_sha256(output / "checkpoint.pt"))
    _metadata(output, metadata)
    return dict(output_dir=str(output), checkpoint="checkpoint.pt", progress=result["progress"],
                last_loss=result["last_loss"], steps_this_run=result["steps_this_run"], stage_status=stages)


def train_joint_finetuning(options, base_dir=Path(".")):
    """Actual DER/GER routing, source-leaf DG and per-block level balance."""
    settings = _settings(options)
    base_dir = Path(base_dir).resolve()
    output = base_dir / options["output_dir"]
    if output.exists():
        raise FileExistsError("Training output directory already exists: " + str(output))
    torch.set_num_threads(1)
    torch.manual_seed(settings["run"]["seed"])
    records = read_pairs(options, base_dir)
    hierarchy, labels, ordered, hashes = read_hierarchy(base_dir / options["hierarchy_dir"], records)
    run = settings["run"]
    if run["crop_size"] is None and run["batch_size"] > 1 and len({row["_size"] for row in records}) > 1:
        raise ValueError("Batching unequal grids requires crop_size or batch_size=1")
    mix_options = dict(options)
    mix_options.setdefault("cutmix", dict(enabled=True, box=None, donor_shift=None))
    settings["cutmix"] = _cutmix_settings(mix_options, records, settings)
    dataset = MatchedHierarchyPairedDataset(ordered, labels, seed=run["seed"], crop_size=run["crop_size"])
    space = DegradationSpace(base_dir / options["hierarchy_dir"])
    source_path = base_dir / options["der_checkpoint"]
    model, source = load_routing_stage_checkpoint(source_path, expected_stage="der_initialization",
                                                  hierarchy=hierarchy, artifact_sha256=hashes)
    if source["pair_binding"] != binding(records):
        raise ValueError("DER checkpoint pair_binding differs from the supplied pair CSV")
    parameters = configure_stage_training(model, "joint_finetuning")
    frozen_sha = frozen_state_sha256(model)
    identity = dict(source_checkpoint_sha256=file_sha256(source_path), hierarchy=hierarchy,
                    artifact_sha256=hashes, pair_binding=binding(records), pair_binding_sha256=binding_sha256(records),
                    network_config=dict(source["network_config"], freeze_dr=True),
                    trainable_parameter_names=_trainable_names(model), frozen_state_sha256=frozen_sha)
    resume = None
    parent_path = source_path
    if options.get("resume"):
        parent_path = base_dir / options["resume"]
        resume = _load(parent_path, "joint_finetuning")
        _check_resume(resume, identity, settings, records)
        model.load_state_dict(resume["params"], strict=True)
        if frozen_state_sha256(model) != frozen_sha:
            raise ValueError("Resume changes a fixed parameter or buffer")
    optimizer, scheduler, parameters = _optimization(parameters, settings, resume)
    output.mkdir(parents=True, exist_ok=False)
    result = _train_stage(model, dataset, parameters, optimizer, scheduler, settings,
                          output / "metrics.csv", resume, joint_space=space)
    if frozen_state_sha256(model) != frozen_sha:
        raise RuntimeError("Fixed parameters or buffers changed; no completion metadata is written")
    stages = deepcopy(source["stage_status"])
    stages.update(joint="actual DER/GER routing; joint objective", der="joint updates enabled after CE initialization",
                  ger="joint level-balance/reconstruction updates enabled",
                  phi_mu="joint DG and DER updates enabled", phi_variance="joint DG and GER updates enabled",
                  routers="DER and GER jointly trained", phi="both heads jointly trained",
                  cluster_lora="joint updates from pretrained cluster experts")
    inherited = {key: deepcopy(source[key]) for key in ("base_config", "base_checkpoint_sha256", "trained_heap_ids",
                 "untrained_heap_ids", "dr_checkpoint_sha256", "adapter_checkpoints")}
    payload = dict(schema_version=1, artifact_type="joint_finetuning", completed=True,
                   params=_state(model), **inherited, **identity, parent_checkpoint_sha256=file_sha256(parent_path),
                   recipe=settings, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                   optimizer_parameter_count=len(parameters), trainable_parameter_count=sum(p.numel() for p in parameters),
                   effective_options=_effective(optimizer, scheduler), stage_status=stages,
                   loss="L1 + 0.1 * mean(block DG) + 0.01 * mean(block level load balance)",
                   DG_target="source/CutMix leaf raw centroid; original native D grid; shared across blocks",
                   load_balance="0.5 CV^2(clean-softmax importance) + 0.5 CV^2(noise-expected load); sample std",
                   variance_gradient_expected=True, training_quality_inferred=False, **result)
    _save(output / "checkpoint.pt", payload)
    metadata = {key: value for key, value in payload.items()
                if key not in ("params", "optimizer", "scheduler", "torch_rng_state", "schema_version", "completed")}
    metadata.update(checkpoint="checkpoint.pt", checkpoint_sha256=file_sha256(output / "checkpoint.pt"))
    _metadata(output, metadata)
    return dict(output_dir=str(output), checkpoint="checkpoint.pt", progress=result["progress"],
                last_loss=result["last_loss"], last_losses=result["last_losses"],
                steps_this_run=result["steps_this_run"], stage_status=stages)


__all__ = ["configure_stage_training", "der_ce_loss", "frozen_state_sha256",
           "load_routing_stage_checkpoint", "train_der_initialization", "train_joint_finetuning"]
