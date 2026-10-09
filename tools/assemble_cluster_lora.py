import argparse
import json
from pathlib import Path

from basicsr.restoration_training import assemble_cluster_lora


ROOT = Path(__file__).resolve().parents[1]
# Paths are relative to ROOT; these are the preceding stages' output artifacts.
CONFIG = {
    "base_checkpoint": "outputs/base_training/checkpoint.pt",
    "hierarchy_dir": "outputs/hierarchical_clustering",
    "cluster_dir": "outputs/cluster_lora_training",
    "dr_checkpoint": "outputs/dr_training/checkpoint.pt",
    "output_dir": "outputs/assembled_restorer",
    "require_complete": True,
    "seed": 19,
}


def main(argv=None):
    argparse.ArgumentParser(description="Assemble the base, DR and cluster LoRA weights using CONFIG above.").parse_args(argv)
    print(json.dumps(assemble_cluster_lora(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
