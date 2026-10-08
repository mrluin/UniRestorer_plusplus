"""Options for the included single-process restoration entry points."""

from pathlib import Path
import yaml


def load_runtime_options(filename, is_train):
    path = Path(filename).resolve()
    options = yaml.safe_load(path.read_text(encoding="utf-8"))
    options.setdefault("num_gpu", 0)
    if options["num_gpu"] != 0 or options.get("dist"):
        raise NotImplementedError("GPU/distributed execution is not supported by these runners")
    options.update(is_train=is_train, dist=False, rank=0, world_size=1)
    options.setdefault("scale", 1)
    options.setdefault("name", path.stem)
    options.setdefault("model_type", "SRModel")
    options.setdefault("manual_seed", 19)
    options.setdefault("val", {})
    root = path.parent

    def resolve(value):
        target = Path(value).expanduser()
        return str(target if target.is_absolute() else (root / target).resolve())

    for name, config in options.get("datasets", {}).items():
        config.setdefault("phase", name.split("_", 1)[0])
        config.setdefault("scale", options["scale"])
        for key in ("dataroot_gt", "dataroot_lq", "meta_info_file"):
            if config.get(key):
                config[key] = resolve(config[key])
    locations = options.setdefault("path", {})
    for key in ("models", "training_states", "visualization", "pretrain_network_g", "resume_state"):
        if locations.get(key):
            locations[key] = resolve(locations[key])
    checkpoint = options["network_g"].get("base_checkpoint")
    if checkpoint:
        options["network_g"]["base_checkpoint"] = resolve(checkpoint)
    return options
