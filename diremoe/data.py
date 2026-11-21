"""Portable feature stores and explicit train/validation/test indices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .checkpoint import feature_config
from .config import BENCHMARKS, matches_recipe
from .features import FeatureExtractor


def _feature_shapes(config, size):
    channels = config["n_channels"]
    views = len(config["window_fractions"]) * len(config["lags_samples"])
    temporal_shape = (channels, 512) if config["temporal_mode"] == "regional" else (512,)
    return {
        "temporal": (size, *temporal_shape),
        "asymmetric": (size, views, channels * (channels - 1) // 2),
        "valid": (size, views),
    }


def _labels(array):
    if array.ndim != 1 or not array.size or not np.isin(array, (0, 1)).all():
        raise ValueError("labels must be a nonempty vector containing only 0 and 1.")
    return array.astype(np.int64, copy=False)


def extract(input_path, output: Path, dataset, weights, device, batch_size=8, physical_lags=None, regional=False):
    """Extract reusable features from a dense numeric signals NPZ."""
    if output.exists():
        raise FileExistsError(output)
    with np.load(input_path, allow_pickle=False) as source:
        signals = source["signals"]
        labels = _labels(source["labels"])
        lengths = source["lengths"] if "lengths" in source else None
        intervals = source['tr_seconds'] if physical_lags is not None and 'tr_seconds' in source else None
    if physical_lags is not None and (intervals is None or intervals.shape != labels.shape
                                      or not np.isfinite(intervals).all() or (intervals <= 0).any()):
        raise ValueError('Physical lags require one positive tr_seconds per recording.')
    config = BENCHMARKS[dataset].feature_config(physical_lags, regional)
    if (
        signals.ndim != 3 or signals.shape[:2] != (len(labels), config["n_channels"])
        or signals.dtype.kind not in "fiu"
        or (lengths is not None and lengths.shape != labels.shape)
    ):
        raise ValueError("Expected numeric signals [N,C,T], labels [N], and optional lengths [N].")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "features"
        stage.mkdir()
        extractor = FeatureExtractor(config, weights, device)
        shapes = _feature_shapes(config, len(labels))
        types = (np.float16 if config["temporal_mode"] == "regional" else np.float32,
                 np.float16, np.bool_)
        arrays = [
            np.lib.format.open_memmap(stage / f"{key}.npy", "w+", dtype=dtype, shape=shape)
            for (key, shape), dtype in zip(shapes.items(), types)
        ]
        for start in range(0, len(labels), batch_size):
            rows = slice(start, start + batch_size)
            features = extractor(signals[rows], None if lengths is None else lengths[rows],
                                 tr_seconds=None if intervals is None else intervals[rows])
            for array, value in zip(arrays, features):
                array[rows] = value.cpu().numpy()
        for array in arrays:
            array.flush()
        del array, arrays
        np.save(stage / "labels.npy", labels, allow_pickle=False)
        manifest = {"complete": True, "dataset": dataset, "feature_config": config}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)


class FeatureStore:
    """Memory-map the feature bank and transfer only requested rows to the device."""

    def __init__(self, path: Path):
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if not manifest["complete"]:
            raise ValueError("Incomplete feature store.")
        self.dataset = manifest["dataset"]
        self.config = feature_config(manifest)
        if not matches_recipe(self.dataset, self.config):
            raise ValueError("Feature store does not match the benchmark recipe.")
        self.labels = _labels(np.load(path / "labels.npy", allow_pickle=False))
        shapes = _feature_shapes(self.config, len(self.labels))
        self.arrays = []
        for key, shape in shapes.items():
            array = np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
            dtype = np.bool_ if key == "valid" else (
                np.float32 if key == "temporal" and self.config["temporal_mode"] == "mean" else np.float16
            )
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"Unexpected shape or dtype for {key}.")
            self.arrays.append(array)
        self.model_kwargs = {
            key: self.config[key] for key in ("n_channels", "lags_samples", "temporal_mode")
        }
        self.model_kwargs["n_views"] = shapes["valid"][1]

    def batch(self, rows, device):
        return tuple(torch.as_tensor(array[rows], device=device) for array in self.arrays)


def load_split(path, size):
    """Read unique, disjoint, zero-based indices into the feature store."""
    result = {}
    with np.load(path, allow_pickle=False) as split:
        for key in ("train", "val", "test"):
            rows = split[key]
            if (
                rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
                or rows.min() < 0 or rows.max() >= size or len(np.unique(rows)) != len(rows)
            ):
                raise ValueError(f"Invalid {key} indices.")
            result[key] = rows.astype(np.int64, copy=False)
    combined = np.concatenate(list(result.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Train, validation and test indices must be disjoint.")
    return result


def split_hashes(split):
    return {key: hashlib.sha256(rows.astype("<i8", copy=False).tobytes()).hexdigest()
            for key, rows in split.items()}
