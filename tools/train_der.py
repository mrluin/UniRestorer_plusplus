import argparse
import json
from pathlib import Path

from basicsr.restoration_training.routing_stages import train_der_initialization


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT. Fill the None run settings before training.
CONFIG = {
    "pair_csv": "data/restoration_pairs.csv",
    "lq_root": "data/inputs",
    "gt_root": "data/targets",
    "hierarchy_dir": "outputs/hierarchical_clustering",
    "assembled_checkpoint": "outputs/assembled_restorer/checkpoint.pt",
    "output_dir": "outputs/der_initialization",
    "run": {
        "seed": None, "epochs": None, "batch_size": None, "shuffle": None,
        "crop_size": None, "max_steps": None,
    },
    "optimizer": {"type": None, "lr": None},
    "scheduler": {"type": None},
    # Optional deterministic CutMix needs explicit box and donor shift.
    "cutmix": {"enabled": False, "box": None, "donor_shift": None},
    "resume": None,
}


def main(argv=None):
    argparse.ArgumentParser(description="Initialize DER with true-leaf teacher CE using CONFIG above.").parse_args(argv)
    print(json.dumps(train_der_initialization(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
