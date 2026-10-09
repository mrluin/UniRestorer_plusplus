import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from basicsr.dr_training.text import file_sha256, load_frozen_clip_text_encoder


ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "captions_csv": "data/captions_input.csv",  # caption_id,caption
    "output_dir": "outputs/caption_bank",
    "model_name": "ViT-B-32",
    "checkpoint": "checkpoints/clip_ViT-B-32.pt",  # Existing local weights.
    "checkpoint_format": "state_dict",  # Or explicitly trusted-local torchscript.
    "checkpoint_key": None,
    "checkpoint_prefix": "",
    "force_quick_gelu": False,  # DA-CLIP LAION weights use GELU; OpenAI weights use True.
    "pretrained_identifier": "laion2b_s34b_b79k",  # Provenance label, never a download tag.
    "prompt_template": "{caption}",
    "batch_size": 32,
    "cpu_threads": 1,
}


def prepare_caption_bank(options: dict, base_dir=Path(".")) -> dict:
    """Write a new bank directory, deduplicating exact outer-trimmed captions."""
    base_dir = Path(base_dir).resolve()
    if "://" in str(options["checkpoint"]) or str(options["checkpoint"]).startswith("hf-hub:"):
        raise ValueError("checkpoint must be a local file, not a remote URI")
    checkpoint = (base_dir / options["checkpoint"]).resolve()
    captions_csv = (base_dir / options["captions_csv"]).resolve()
    output_dir = (base_dir / options["output_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    ids, rows_by_caption, unique, mappings, inputs = {}, {}, [], [], []
    with captions_csv.open(encoding="utf-8-sig", newline="") as stream:
        for record in csv.DictReader(stream):
            input_id, raw_caption = int(record["caption_id"]), record["caption"]
            caption = raw_caption.strip()
            if not caption:
                raise ValueError("caption text cannot be empty")
            if input_id in ids and ids[input_id] != caption:
                raise ValueError(f"caption_id {input_id} is assigned different captions")
            ids[input_id] = caption
            if caption not in rows_by_caption:
                rows_by_caption[caption] = len(unique)
                unique.append({"caption_id": input_id, "caption": caption})
            row = rows_by_caption[caption]
            mappings.append({"input_caption_id": input_id, "caption_id": unique[row]["caption_id"],
                             "bank_row": row, "caption": caption})
            inputs.append({"input_caption_id": input_id, "raw_caption": raw_caption})
    if not unique:
        raise ValueError("captions_csv contains no captions")
    template = options.get("prompt_template", "{caption}")
    prompts = [template.format(caption=row["caption"]) for row in unique]
    torch.set_num_threads(int(options.get("cpu_threads", 1)))
    model, tokenizer, encoder_info = load_frozen_clip_text_encoder(
        checkpoint, model_name=options.get("model_name", "ViT-B-32"),
        checkpoint_format=options.get("checkpoint_format", "state_dict"),
        checkpoint_key=options.get("checkpoint_key"),
        checkpoint_prefix=options.get("checkpoint_prefix", ""),
        force_quick_gelu=options.get("force_quick_gelu", False),
        pretrained_identifier=options.get("pretrained_identifier", "laion2b_s34b_b79k"),
    )
    tokens = tokenizer(prompts, context_length=77).cpu()
    if tokens.shape != (len(unique), 77):
        raise RuntimeError("the stock tokenizer must return N x 77 token IDs")
    token_ids, token_rows = tokens.tolist(), {}
    for row, row_tokens in enumerate(token_ids):
        key = tuple(row_tokens)
        if key in token_rows:
            other = token_rows[key]
            raise ValueError(f"distinct captions produce identical token IDs: {unique[other]['caption']!r} and "
                             f"{unique[row]['caption']!r}; shorten the captions or explicitly unify their aliases")
        token_rows[key] = row

    batch_size = int(options.get("batch_size", 32))
    output_dir.mkdir(parents=True, exist_ok=False)
    files = {"features": "caption_bank.npy", "captions": "captions.csv", "class_mapping": "class_mapping.csv"}
    features = np.lib.format.open_memmap(output_dir / files["features"], mode="w+",
                                         dtype=np.float32, shape=(len(unique), 512))
    try:
        with torch.no_grad():
            for start in range(0, len(unique), batch_size):
                batch = tokens[start:start + batch_size]
                vectors = model.encode_text(batch, normalize=False).float().cpu()
                if vectors.shape != (len(batch), 512):
                    raise RuntimeError("the stock teacher must return N x 512 text vectors")
                norms = vectors.norm(dim=1)
                if not torch.isfinite(vectors).all() or not torch.isfinite(norms).all() or (norms == 0).any():
                    raise RuntimeError("the teacher produced non-finite or zero text vectors")
                features[start:start + len(vectors)] = vectors.numpy()
        features.flush()
    finally:
        features._mmap.close()
    for key, records in (("captions", unique), ("class_mapping", mappings)):
        with (output_dir / files[key]).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    metadata = {
        "schema_version": 1, "completed": True, "p": 512, "num_classes": len(unique),
        "num_input_rows": len(inputs), "normalization": "none", "feature_dtype": "float32",
        "encoder": encoder_info, "files": files,
        "file_sha256": {key: file_sha256(output_dir / name) for key, name in files.items()},
        "captions_csv": {"path": str(options["captions_csv"]), "sha256": file_sha256(captions_csv)},
        "caption_deduplication": "outer whitespace stripped, exact case-sensitive match, first occurrence",
        "input_captions": inputs, "class_mapping": mappings,
        "prompt_template": template, "prompts": prompts, "token_ids": token_ids,
        "encoding": {"device": "cpu", "frozen": True, "eval": True, "no_grad": True,
                     "normalize": False, "batch_size": batch_size, "cpu_threads": torch.get_num_threads()},
        "training_quality_inferred": False,
    }
    # A completed manifest exists only after vectors, mappings and their hashes succeed.
    with (output_dir / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return {"output_dir": str(output_dir), "num_classes": len(unique), "p": 512,
            "files": files, "metadata": "metadata.json"}


def main():
    parser = argparse.ArgumentParser(description="Prepare a local caption bank using CONFIG above.")
    parser.parse_args()
    print(json.dumps(prepare_caption_bank(CONFIG, base_dir=ROOT), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
