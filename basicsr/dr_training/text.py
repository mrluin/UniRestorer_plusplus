import hashlib
from importlib import metadata
from pathlib import Path
import zipfile

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_checkpoint(path, key=None, prefix=""):
    """Load CPU weights from a local file with explicit key/prefix selection."""
    if "://" in str(path) or str(path).startswith("hf-hub:"):
        raise ValueError("checkpoint must be a local file, not a remote URI")
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"a local checkpoint file is required: {path}")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if any(name.rsplit("/", 1)[-1] == "constants.pkl" for name in archive.namelist()):
                raise ValueError("TorchScript archive requires explicit checkpoint_format='torchscript'")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if key is not None:
        state = state[key]
    if prefix:
        state = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
    return state


def load_frozen_clip_text_encoder(
    checkpoint, *, model_name="ViT-B-32", checkpoint_format="state_dict",
    checkpoint_key=None, checkpoint_prefix="", force_quick_gelu=False,
    pretrained_identifier="laion2b_s34b_b79k",
):
    """Return the frozen stock teacher, tokenizer and local checkpoint metadata."""
    if model_name != "ViT-B-32":
        raise NotImplementedError("offline text encoding supports only stock ViT-B-32")
    if "://" in str(pretrained_identifier) or str(pretrained_identifier).startswith("hf-hub:"):
        raise ValueError("pretrained_identifier is a provenance label, not a remote URI")
    removed_metadata = []
    if checkpoint_format == "state_dict":
        state = load_local_checkpoint(checkpoint, checkpoint_key, checkpoint_prefix)
    elif checkpoint_format == "torchscript":
        if "://" in str(checkpoint) or str(checkpoint).startswith("hf-hub:"):
            raise ValueError("checkpoint must be a local file, not a remote URI")
        state = torch.jit.load(str(Path(checkpoint).resolve()), map_location="cpu").state_dict()
        if checkpoint_key is not None:
            state = state[checkpoint_key]
        if checkpoint_prefix:
            state = {name[len(checkpoint_prefix):]: value for name, value in state.items()
                     if name.startswith(checkpoint_prefix)}
        removed_metadata = [name for name in ("input_resolution", "context_length", "vocab_size") if name in state]
        for name in removed_metadata:
            del state[name]
    else:
        raise ValueError("checkpoint_format must be 'state_dict' or 'torchscript'")
    checkpoint = Path(checkpoint).resolve()

    # No pretrained tag, remote path, HF model or tokenizer is resolved here.
    import open_clip

    try:
        package_version = getattr(open_clip, "__version__", None) or metadata.version("open_clip_torch")
    except metadata.PackageNotFoundError:
        package_version = "unknown/source checkout"
    module_file = getattr(open_clip, "__file__", None)
    module_file = str(module_file) if module_file is not None else None
    model_config = open_clip.get_model_config(model_name)
    if model_config["text_cfg"].get("hf_tokenizer_name") or model_config["text_cfg"].get("hf_model_name"):
        raise NotImplementedError("this tool requires the package-local stock BPE tokenizer")
    factory_args = {"pretrained": None, "precision": "fp32", "device": "cpu",
                    "force_quick_gelu": force_quick_gelu,
                    "pretrained_image": False, "pretrained_hf": False}
    model = open_clip.create_model(model_name, **factory_args)
    model.load_state_dict(state, strict=True)
    model.cpu().float().requires_grad_(False).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    info = {
        "model_name": model_name, "package": "open_clip_torch",
        "package_version": str(package_version), "module_file": module_file,
        "module_sha256": file_sha256(module_file) if module_file and Path(module_file).is_file() else None,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "factory_args": factory_args, "model_config": model_config,
        "pretrained_identifier": pretrained_identifier, "pretrained_identifier_verified": False,
        "activation": "QuickGELU" if force_quick_gelu else "GELU",
        "tokenizer": {"model_name": model_name, "assets": "package-local BPE", "context_length": 77},
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                       "format": checkpoint_format, "key": checkpoint_key,
                       "prefix": checkpoint_prefix, "removed_metadata": removed_metadata},
        "device": "cpu", "precision": "fp32", "frozen": True, "eval": True,
        "encode_text_normalize": False, "output_dimension": 512,
        "source": {"factory": "https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/factory.py",
                   "model": "https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/model.py",
                   "loader_sha256": file_sha256(__file__)},
    }
    return model, tokenizer, info


__all__ = ["load_frozen_clip_text_encoder", "load_local_checkpoint", "file_sha256"]
