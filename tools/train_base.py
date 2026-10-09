import argparse
import json
from pathlib import Path

from basicsr.restoration_training import train_base


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT. Fill the None run settings before training.
CONFIG = {
    "pair_csv": "data/restoration_pairs.csv",
    "lq_root": "data/inputs",
    "gt_root": "data/targets",
    "output_dir": "outputs/base_training",
    # Retain the existing RGB backbone example; checkpoint records this config.
    "model": {
        "inp_channels": 3,
        "out_channels": 3,
        "dim": 48,
        "num_blocks": [4, 6, 6, 8],
        "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8],
        "ffn_expansion_factor": 2.66,
        "bias": False,
        "LayerNorm_type": "BiasFree",
        "dual_pixel_task": False,
    },
    "run": {
        "seed": None, "epochs": None, "batch_size": None, "shuffle": None,
        "crop_size": None, "max_steps": None,
    },
    "optimizer": {"type": None, "lr": None},
    "scheduler": {"type": None},
    "resume": None,
}


def main(argv=None):
    argparse.ArgumentParser(description="Train the native Restormer base using CONFIG above.").parse_args(argv)
    print(json.dumps(train_base(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
