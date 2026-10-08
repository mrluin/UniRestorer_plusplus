"""Generic supervised restoration runner; objectives are explicit in YAML.

This runner does not implement paper-specific clustering or supervision stages.
Run from the repository root: python -m basicsr.train --opt options/train_example.yml
"""

import argparse
from pathlib import Path
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import set_random_seed
from basicsr.utils.runtime import load_runtime_options


def train_pipeline(options):
    if options.get("num_gpu", 0) != 0 or options.get("dist"):
        raise NotImplementedError("GPU/distributed training is not supported by this runner")
    torch.set_num_threads(1)
    set_random_seed(options["manual_seed"])
    total = options["train"]["total_iter"]
    config = options["datasets"]["train"]
    dataset = build_dataset(config)
    if len(dataset) == 0:
        raise ValueError("The training dataset is empty")
    loader = build_dataloader(dataset, config, num_gpu=0, dist=False, seed=options["manual_seed"])
    if len(loader) == 0:
        raise ValueError("The dataset/batch configuration produces no training batches")
    for name in ("models", "training_states"):
        Path(options["path"][name]).mkdir(parents=True, exist_ok=True)
    resume = options["path"].get("resume_state")
    if resume and not options["path"].get("pretrain_network_g"):
        raise ValueError("resume_state also requires its matching pretrain_network_g checkpoint")
    model = build_model(options)
    iteration, epoch = 0, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        model.resume_training(state)
        iteration, epoch = state["iter"], state["epoch"]
    while iteration < total:
        for batch in loader:
            iteration += 1
            model.update_learning_rate(iteration, options["train"].get("warmup_iter", -1))
            model.feed_data(batch)
            model.optimize_parameters(iteration)
            if iteration >= total:
                break
        epoch += 1
    model.save(epoch, iteration)
    return {"iteration": iteration, "epoch": epoch, "losses": dict(getattr(model, "log_dict", {}))}


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", required=True)
    args = parser.parse_args()
    result = train_pipeline(load_runtime_options(args.opt, is_train=True))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
