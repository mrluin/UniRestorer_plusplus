import argparse
import json
from pathlib import Path

from basicsr.restoration_training.routing_stages import train_joint_finetuning


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT. Fill the None run and mixing settings first.
CONFIG = {
    "pair_csv": "data/restoration_pairs.csv",
    "lq_root": "data/inputs",
    "gt_root": "data/targets",
    "hierarchy_dir": "outputs/hierarchical_clustering",
    "der_checkpoint": "outputs/der_initialization/checkpoint.pt",
    "output_dir": "outputs/joint_finetuning",
    "run": {
        "seed": None, "epochs": None, "batch_size": None, "shuffle": None,
        "crop_size": None, "max_steps": None,
    },
    "optimizer": {"type": None, "lr": None},
    "scheduler": {"type": None},
    # Same box and donor assignment apply to LQ, GT and all source-label maps.
    "cutmix": {"enabled": True, "box": None, "donor_shift": None},
    "resume": None,
}


def main(argv=None):
    argparse.ArgumentParser(description="Jointly train routers, phi and LoRA using CONFIG above.").parse_args(argv)
    print(json.dumps(train_joint_finetuning(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
