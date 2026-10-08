"""Generic CPU restoration evaluation for externally supplied images/checkpoint."""

import argparse
from pathlib import Path
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils.runtime import load_runtime_options


def test_pipeline(options):
    if options.get("num_gpu", 0) != 0 or options.get("dist"):
        raise NotImplementedError("GPU/distributed evaluation is not supported by this runner")
    torch.set_num_threads(1)
    model = build_model(options)
    results = {}
    for name, config in options["datasets"].items():
        dataset = build_dataset(config)
        loader = build_dataloader(dataset, config, num_gpu=0, dist=False)
        if options["val"].get("save_img", False):
            Path(options["path"]["visualization"]).mkdir(parents=True, exist_ok=True)
        model.validation(loader, current_iter=0, tb_logger=None,
                         save_img=options["val"].get("save_img", False))
        results[name] = dict(getattr(model, "metric_results", {}))
    return results


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", required=True)
    args = parser.parse_args()
    print(json.dumps(test_pipeline(load_runtime_options(args.opt, is_train=False)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
