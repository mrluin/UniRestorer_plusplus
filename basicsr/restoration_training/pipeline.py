import csv
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from basicsr.dr_training.text import load_local_checkpoint
from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.unirestorerpp.network import UniRestorerPP
from .data import PairedCSVDataset, binding, binding_sha256, file_sha256, read_hierarchy, read_pairs


BASE_CONFIG = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                   num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                   bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False)


def _settings(options):
    required = {"run": ("seed", "epochs", "batch_size", "shuffle"),
                "optimizer": ("type", "lr"), "scheduler": ("type",)}
    missing = [section + "." + key for section, keys in required.items()
               for key in keys if options.get(section, {}).get(key) is None]
    if missing:
        raise ValueError("Configure required fields: " + ", ".join(missing))
    settings = {key: deepcopy(options[key]) for key in required}
    run = settings["run"]
    run.setdefault("crop_size", None)
    run.setdefault("max_steps", None)
    if run["epochs"] <= 0 or run["batch_size"] <= 0 or (run["max_steps"] is not None and run["max_steps"] <= 0):
        raise ValueError("run.epochs, batch_size and any max_steps must be positive")
    if settings["optimizer"]["type"] not in ("Adam", "AdamW"):
        raise ValueError("optimizer.type must be Adam or AdamW")
    scheduler = settings["scheduler"]
    if scheduler["type"] not in ("none", "StepLR"):
        raise ValueError("scheduler.type must be 'none' or 'StepLR'")
    if scheduler["type"] == "StepLR" and any(scheduler.get(key) is None for key in ("step_size", "gamma")):
        raise ValueError("Configure scheduler.step_size and scheduler.gamma")
    return settings


def _stable_recipe(settings):
    recipe = deepcopy(settings)
    recipe["run"].pop("epochs")
    recipe["run"].pop("max_steps")
    return json.loads(json.dumps(recipe))


def _load(path, artifact_type=None):
    payload = load_local_checkpoint(path)
    if artifact_type is not None and (payload.get("completed") is not True or payload.get("artifact_type") != artifact_type):
        raise ValueError("Checkpoint is not a completed " + artifact_type + " artifact")
    return payload


def _check_resume(payload, expected, settings, records):
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError("Resume checkpoint has a different " + key)
    if _stable_recipe(payload["recipe"]) != _stable_recipe(settings):
        raise ValueError("Resume training recipe differs; optimizer/lr and crop/shuffle settings must match")
    # Validate the saved RNG before any selected node trains or writes outputs.
    torch.Generator(device="cpu").set_state(payload["torch_rng_state"])
    progress, run = payload["progress"], settings["run"]
    batches = (len(records) + run["batch_size"] - 1) // run["batch_size"]
    if (progress["epoch"] >= run["epochs"] or not 0 <= progress["batch_in_epoch"] < batches or
            (run["max_steps"] is not None and progress["step"] >= run["max_steps"])):
        raise ValueError("Resume limits must allow a new step; extend epochs/max_steps and check dataset batches")


def _optimization(parameters, settings, resume=None):
    parameters = list(dict.fromkeys(parameters))
    config = deepcopy(settings["optimizer"])
    optimizer = getattr(torch.optim, config.pop("type"))(parameters, **config)
    config = deepcopy(settings["scheduler"])
    scheduler = None if config.pop("type") == "none" else torch.optim.lr_scheduler.StepLR(optimizer, **config)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resume["scheduler"])
        for parameter, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state and state[key].shape != parameter.shape:
                    raise ValueError("Resume optimizer tensor dimensions do not match target parameters")
    return optimizer, scheduler, parameters


def _effective(optimizer, scheduler):
    return {"optimizer": [{key: value for key, value in group.items() if key != "params"}
                           for group in optimizer.param_groups],
            "scheduler": scheduler.state_dict() if scheduler is not None else None}


def _base_config(options):
    config = deepcopy(BASE_CONFIG)
    config.update(options.get("model", {}))
    if config["inp_channels"] != 3 or config["out_channels"] != 3 or config["dual_pixel_task"]:
        raise ValueError("Base training supports paired RGB restoration")
    return config


def _native_forward(model, X):
    height, width = X.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        X = F.pad(X, (0, pad_w, 0, pad_h), mode=mode)
    return model(X)[..., :height, :width]


def _state(model):
    # Preserve aliases in the snapshot, so torch.save writes each shared base once.
    owners = dict(model.named_parameters(remove_duplicate=False))
    owners.update(model.named_buffers(remove_duplicate=False))
    snapshots, state = {}, {}
    for key, value in model.state_dict().items():
        owner = id(owners.get(key, value))
        if owner not in snapshots:
            snapshots[owner] = value.detach().cpu().clone()
        state[key] = snapshots[owner]
    return state


def _dataset(records, settings):
    run = settings["run"]
    if (run["crop_size"] is None and run["batch_size"] > 1 and
            len({row["_size"] for row in records}) > 1):
        raise ValueError("Batching unequal image grids requires crop_size or batch_size=1; images are not resized")
    return PairedCSVDataset(records, seed=run["seed"], crop_size=run["crop_size"])


def _save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _metadata(directory, payload):
    with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(schema_version=1, completed=True, **payload), stream, indent=2)
        stream.write("\n")


def _train(forward, dataset, parameters, optimizer, scheduler, settings, metrics_path, resume=None):
    run = settings["run"]
    progress = dict(epoch=0, batch_in_epoch=0, step=0) if resume is None else dict(resume["progress"])
    if resume is not None:
        torch.set_rng_state(resume["torch_rng_state"])
    first_step = progress["step"]
    last_loss = resume.get("last_loss") if resume is not None else None
    with metrics_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "epoch", "batch", "loss", "grad_norm", "lr"))
        writer.writeheader()
        while progress["epoch"] < run["epochs"] and (run["max_steps"] is None or progress["step"] < run["max_steps"]):
            epoch = progress["epoch"]
            dataset.epoch = epoch
            generator = torch.Generator().manual_seed(run["seed"] + epoch)
            loader = DataLoader(dataset, batch_size=run["batch_size"], shuffle=run["shuffle"],
                                num_workers=0, generator=generator)
            for batch_index, batch in enumerate(loader):
                if batch_index < progress["batch_in_epoch"]:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = F.l1_loss(forward(batch["lq"]), batch["gt"], reduction="mean")
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite L1 loss; no completion metadata is written")
                loss.backward()
                grad_norm = sum(parameter.grad.square().sum() for parameter in parameters
                                if parameter.grad is not None).sqrt()
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("Non-finite gradients; no completion metadata is written")
                optimizer.step()
                progress.update(batch_in_epoch=batch_index + 1, step=progress["step"] + 1)
                last_loss = float(loss.detach())
                writer.writerow(dict(step=progress["step"], epoch=epoch, batch=batch_index,
                                     loss=last_loss, grad_norm=float(grad_norm), lr=optimizer.param_groups[0]["lr"]))
                if run["max_steps"] is not None and progress["step"] >= run["max_steps"]:
                    break
            if progress["batch_in_epoch"] == len(loader):
                progress.update(epoch=epoch + 1, batch_in_epoch=0)
                if scheduler is not None:
                    scheduler.step()
    return dict(progress=progress, last_loss=last_loss, steps_this_run=progress["step"] - first_step,
                torch_rng_state=torch.get_rng_state())


def train_base(options, base_dir=Path(".")):
    settings = _settings(options)
    base_dir = Path(base_dir).resolve()
    output = base_dir / options["output_dir"]
    if output.exists():
        raise FileExistsError("Training output directory already exists: " + str(output))
    torch.set_num_threads(1)
    torch.manual_seed(settings["run"]["seed"])
    records = read_pairs(options, base_dir)
    dataset = _dataset(records, settings)
    config = _base_config(options)
    model = Restormer(**config).cpu().train()
    identity = dict(base_config=config, pair_binding=binding(records), pair_binding_sha256=binding_sha256(records))
    resume = None
    if options.get("resume"):
        resume = _load(base_dir / options["resume"], "base_restoration_training")
        _check_resume(resume, identity, settings, records)
        model.load_state_dict(resume["params"], strict=True)
    optimizer, scheduler, parameters = _optimization(model.parameters(), settings, resume)
    output.mkdir(parents=True, exist_ok=False)
    result = _train(lambda X: _native_forward(model, X), dataset, parameters, optimizer, scheduler,
                    settings, output / "metrics.csv", resume)
    payload = dict(schema_version=1, artifact_type="base_restoration_training", completed=True,
                   params=_state(model), **identity, recipe=settings, optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict() if scheduler is not None else None,
                   trainable_parameter_count=sum(parameter.numel() for parameter in parameters),
                   optimizer_parameter_count=len(parameters), effective_options=_effective(optimizer, scheduler), **result)
    _save(output / "checkpoint.pt", payload)
    _metadata(output, dict(artifact_type=payload["artifact_type"], checkpoint="checkpoint.pt",
                          checkpoint_sha256=file_sha256(output / "checkpoint.pt"), base_config=config,
                          pair_binding=identity["pair_binding"], pair_binding_sha256=identity["pair_binding_sha256"],
                          selected_sample_ids=[row["sample_id"] for row in records], loss="L1 mean",
                          recipe=settings, effective_options=payload["effective_options"], progress=result["progress"],
                          optimizer_parameter_count=len(parameters), trainable_parameter_count=payload["trainable_parameter_count"],
                          steps_this_run=result["steps_this_run"], training_quality_inferred=False))
    return dict(output_dir=str(output), checkpoint="checkpoint.pt", **{key: result[key] for key in
                ("progress", "last_loss", "steps_this_run")})


def _base_payload(path):
    payload = _load(path, "base_restoration_training")
    model = Restormer(**payload["base_config"])
    model.load_state_dict(payload["params"], strict=True)
    return payload


def _cluster_identity(base, base_sha, hierarchy, hashes, records, selected, heap):
    node = hierarchy["nodes"][heap]
    return dict(base_checkpoint_sha256=base_sha, base_config=base["base_config"],
                hierarchy=hierarchy, artifact_sha256=hashes, hierarchy_sha256=hashes["hierarchy.json"],
                heap_id=heap, expert_id=heap - 1, level=node["level"], group=node["group"],
                membership_sample_ids=[row["sample_id"] for row in selected], pair_binding=binding(selected),
                all_pairs_sha256=binding_sha256(records))


def _cluster_model(base, network_config, seed):
    torch.manual_seed(seed)
    model = UniRestorerPP(**base["base_config"], **network_config,
                         base_checkpoint=base, base_checkpoint_key="params").cpu()
    model.eval().requires_grad_(False)
    return model


def train_cluster_lora(options, base_dir=Path(".")):
    settings = _settings(options)
    base_dir = Path(base_dir).resolve()
    output = base_dir / options["output_dir"]
    if output.exists():
        raise FileExistsError("Training output directory already exists: " + str(output))
    torch.set_num_threads(1)
    records = read_pairs(options, base_dir)
    base_path = base_dir / options["base_checkpoint"]
    base = _base_payload(base_path)
    if base["pair_binding"] != binding(records):
        raise ValueError("Base checkpoint pair_binding differs from the supplied pair CSV")
    hierarchy, labels, ordered, hashes = read_hierarchy(base_dir / options["hierarchy_dir"], records)
    selection = options.get("cluster_heap_ids", "all")
    heaps = list(range(1, len(hierarchy["nodes"]))) if selection == "all" else list(selection)
    if not heaps or len(set(heaps)) != len(heaps) or any(heap <= 0 or heap >= len(hierarchy["nodes"]) for heap in heaps):
        raise ValueError("Select unique non-root cluster heap IDs; root has no LoRA expert")
    resume_paths = options.get("resume") or {}
    resume_paths = {int(heap): path for heap, path in resume_paths.items()}
    if set(resume_paths) - set(heaps):
        raise ValueError("Resume keys must be selected cluster heap IDs")
    rank = options.get("rank", 8)
    alpha = options.get("lora_alpha")
    network_config = dict(tree_depth=hierarchy["depth"], p=512, rank=rank,
                          lora_alpha=2 * rank if alpha is None else alpha,
                          router_hidden_channels=options.get("router_hidden_channels", 256))
    base_sha = file_sha256(base_path)
    template = _cluster_model(base, network_config, settings["run"]["seed"])
    plans = []
    # Validate every selected resume before creating any output or training a node.
    for heap in heaps:
        node = hierarchy["nodes"][heap]
        selected = [ordered[index] for index in np.flatnonzero(labels[:, node["level"]] == node["group"])]
        dataset = _dataset(selected, settings)
        identity = _cluster_identity(base, base_sha, hierarchy, hashes, records, selected, heap)
        identity["network_config"] = network_config
        resume = None
        if heap in resume_paths:
            resume = _load(base_dir / resume_paths[heap], "cluster_lora_training")
            _check_resume(resume, identity, settings, selected)
            template.load_cluster_lora_state_dict(heap, resume["lora_state"], strict=True)
        _optimization(template.cluster_lora_parameters(heap), settings, resume)
        plans.append((heap, dataset, identity, resume))
    del template
    output.mkdir(parents=True, exist_ok=False)
    checkpoints, summaries = [], []
    for heap, dataset, identity, resume in plans:
        model = _cluster_model(base, network_config, settings["run"]["seed"] + heap)
        parameters = list(model.cluster_lora_parameters(heap))
        for parameter in parameters:
            parameter.requires_grad_(True)
        if resume is not None:
            model.load_cluster_lora_state_dict(heap, resume["lora_state"], strict=True)
        optimizer, scheduler, parameters = _optimization(parameters, settings, resume)
        result = _train(lambda X: model.forward_cluster(X, heap), dataset, parameters, optimizer, scheduler,
                        settings, output / f"cluster_{heap:04d}_metrics.csv", resume)
        filename = f"cluster_{heap:04d}.pt"
        payload = dict(schema_version=1, artifact_type="cluster_lora_training", completed=True,
                       lora_state=model.cluster_lora_state_dict(heap), **identity, recipe=settings,
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                       optimizer_parameter_count=len(parameters), trainable_parameter_count=sum(p.numel() for p in parameters),
                       effective_options=_effective(optimizer, scheduler), initialization_seed=settings["run"]["seed"] + heap,
                       **result)
        _save(output / filename, payload)
        checkpoints.append(filename)
        summaries.append(dict(heap_id=heap, expert_id=heap - 1, checkpoint=filename,
                              checkpoint_sha256=file_sha256(output / filename),
                              membership_sample_ids=identity["membership_sample_ids"], progress=result["progress"],
                              steps_this_run=result["steps_this_run"], optimizer_parameter_count=len(parameters),
                              trainable_parameter_count=payload["trainable_parameter_count"],
                              effective_options=payload["effective_options"], last_loss=result["last_loss"]))
        del model, parameters, optimizer, scheduler
    _metadata(output, dict(artifact_type="cluster_lora_training", base_checkpoint_sha256=base_sha,
                          base_config=base["base_config"], network_config=network_config,
                          hierarchy=hierarchy, artifact_sha256=hashes, all_pairs_sha256=binding_sha256(records),
                          recipe=settings, nodes=summaries, loss="L1 mean", forward="explicit whole cluster expert",
                          frozen_modules_evaluated=True, dr_phi_router_execution=False, training_quality_inferred=False))
    return dict(output_dir=str(output), checkpoints=checkpoints, heap_ids=heaps, nodes=summaries)


def assemble_cluster_lora(options, base_dir=Path(".")):
    base_dir = Path(base_dir).resolve()
    output = base_dir / options["output_dir"]
    if output.exists():
        raise FileExistsError("Assembly output directory already exists: " + str(output))
    if options.get("seed") is None:
        raise ValueError("Configure assembly seed for the untrained router/phi initialization")
    torch.set_num_threads(1)
    base_path = base_dir / options["base_checkpoint"]
    base = _base_payload(base_path)
    # The base's canonical binding supplies identities; images are not read for assembly.
    records = base["pair_binding"]
    hierarchy, labels, ordered, hashes = read_hierarchy(base_dir / options["hierarchy_dir"], records)
    base_sha = file_sha256(base_path)
    paths = sorted((base_dir / options["cluster_dir"]).glob("cluster_*.pt"))
    if not paths:
        raise ValueError("No cluster LoRA checkpoints were found")
    payloads, heaps, network_config = [], set(), None
    for path in paths:
        payload = _load(path, "cluster_lora_training")
        heap = payload["heap_id"]
        if heap <= 0 or heap >= len(hierarchy["nodes"]) or heap in heaps:
            raise ValueError("Cluster artifacts contain duplicate or invalid/root heap IDs")
        node = hierarchy["nodes"][heap]
        selected = [ordered[index] for index in np.flatnonzero(labels[:, node["level"]] == node["group"])]
        expected = _cluster_identity(base, base_sha, hierarchy, hashes, records, selected, heap)
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError("Cluster checkpoint has a different " + key)
        current = payload["network_config"]
        if current["tree_depth"] != hierarchy["depth"] or current["p"] != 512:
            raise ValueError("Cluster network configuration differs from the hierarchy")
        if network_config is not None and current != network_config:
            raise ValueError("Cluster rank/alpha/network configurations differ")
        network_config = current
        heaps.add(heap)
        payloads.append((path, payload))
    expected_heaps = set(range(1, len(hierarchy["nodes"])))
    if options.get("require_complete", True) and heaps != expected_heaps:
        raise ValueError("Complete assembly requires every non-root cluster checkpoint")
    model = _cluster_model(base, network_config, options["seed"])
    for _, payload in payloads:
        model.load_cluster_lora_state_dict(payload["heap_id"], payload["lora_state"], strict=True)
    dr_path = base_dir / options["dr_checkpoint"]
    dr = _load(dr_path)
    dr_sha = file_sha256(dr_path)
    source_checkpoint = hierarchy.get("source_metadata", {}).get("checkpoint", {})
    if source_checkpoint.get("sha256") != dr_sha:
        raise ValueError("D checkpoint SHA is missing or differs from the hierarchy extraction checkpoint")
    model.degradation_prior.dr_encoder.load_state_dict(dr["dr_encoder"], strict=True)
    stages = dict(base="trained checkpoint loaded", cluster_lora="complete" if heaps == expected_heaps else "partial",
                  dr="explicit dr_encoder checkpoint loaded; training quality not inferred",
                  phi="untrained", routers="untrained")
    saved_network_config = dict(base["base_config"], **network_config)
    payload = dict(schema_version=1, artifact_type="assembled_cluster_lora", completed=True,
                   params=_state(model), base_config=base["base_config"], network_config=saved_network_config,
                   pair_binding=base["pair_binding"], base_checkpoint_sha256=base_sha,
                   hierarchy=hierarchy, artifact_sha256=hashes, trained_heap_ids=sorted(heaps),
                   untrained_heap_ids=sorted(expected_heaps - heaps), dr_checkpoint_sha256=dr_sha,
                   adapter_checkpoints=[dict(heap_id=item["heap_id"], file=path.name, sha256=file_sha256(path))
                                        for path, item in payloads], stage_status=stages,
                   initialization_seed=options["seed"], training_quality_inferred=False)
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "checkpoint.pt", payload)
    _metadata(output, {key: value for key, value in payload.items() if key not in ("params", "schema_version", "completed")})
    return dict(output_dir=str(output), checkpoint="checkpoint.pt", network_config=saved_network_config,
                trained_heap_ids=sorted(heaps), stage_status=stages)
