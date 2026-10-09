import argparse
import json
from pathlib import Path

from basicsr.restoration_training import train_cluster_lora


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT. Fill the None run settings before training.
CONFIG = {
    "pair_csv": "data/restoration_pairs.csv",
    "lq_root": "data/inputs",
    "gt_root": "data/targets",
    "output_dir": "outputs/cluster_lora_training",
    "base_checkpoint": "outputs/base_training/checkpoint.pt",
    "hierarchy_dir": "outputs/hierarchical_clustering",
    "cluster_heap_ids": "all",  # Non-root heap IDs, or an explicit list.
    "rank": 8,
    "lora_alpha": None,  # The existing LoRA default is 2 * rank.
    "router_hidden_channels": 256,
    "run": {
        "seed": None, "epochs": None, "batch_size": None, "shuffle": None,
        "crop_size": None, "max_steps": None,
    },
    "optimizer": {"type": None, "lr": None},
    "scheduler": {"type": None},
    "resume": None,  # Optional mapping of heap IDs to local checkpoint paths.
}


def main(argv=None):
    argparse.ArgumentParser(description="Train cluster LoRA experts using CONFIG above.").parse_args(argv)
    print(json.dumps(train_cluster_lora(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
