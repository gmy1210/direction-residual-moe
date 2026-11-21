"""Persist classifier parameters, feature configuration and fold statistics."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .config import matches_recipe
from .model import DirectionResidualMoE


def feature_config(record: dict) -> dict:
    if "feature_config" in record:
        return record["feature_config"]
    kwargs = record["model_kwargs"]
    manifest = record["feature_manifest"]
    return {
        "n_channels": kwargs["n_channels"],
        "temporal_mode": kwargs["temporal_mode"],
        "lags_samples": kwargs["lags_samples"],
        "window_fractions": manifest["window_fractions"],
        "min_pairs": manifest["min_pairs"],
    }


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    kwargs = dict(checkpoint["model_kwargs"])
    variant = kwargs.pop("variant", "base")
    if variant != "base":
        raise ValueError("This package supports checkpoints for the main direction-residual model.")
    kwargs.pop("edge_dim", None)
    kwargs.pop("beta_init", None)
    config = feature_config(checkpoint)
    if not matches_recipe(checkpoint["dataset"], config):
        raise ValueError("Checkpoint does not match the benchmark feature recipe.")
    if checkpoint.get("feature_manifest", {}).get("config", {}).get("data", {}).get("normalize", False):
        raise ValueError("Checkpoint requires an unsupported input transform.")
    model = DirectionResidualMoE(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint, config


def save_checkpoint(checkpoint: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        path.hardlink_to(temporary)
    finally:
        temporary.unlink(missing_ok=True)
