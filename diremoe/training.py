"""Fit one fold, select on validation, and evaluate the saved checkpoint."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from .calibration import binary_nll, fit_bias
from .checkpoint import load_checkpoint, save_checkpoint
from .config import BENCHMARKS
from .data import FeatureStore, load_split, split_hashes
from .model import DirectionResidualMoE

BATCH_SIZE = 32
WARMUP = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
AUXILIARY_WEIGHT = 0.2


@torch.no_grad()
def fit_normalizer(model, store, train, device):
    temporal, bank, valid = store.arrays
    anchor_rows = train[valid[train, 0]]
    if not len(anchor_rows):
        raise ValueError("At least one training recording needs a valid static FC view.")
    temporal_values = torch.as_tensor(temporal[train], device=device).float()
    static_values = torch.as_tensor(bank[anchor_rows, 0], device=device).float()
    temporal_values = model.pool_time(temporal_values)
    for name, values in (("temporal", temporal_values), ("static", static_values)):
        getattr(model, f"{name}_mean").copy_(values.mean(0))
        getattr(model, f"{name}_std").copy_(values.std(0, unbiased=False).clamp_min(0.001))

    total = torch.zeros(model.direction_dim, dtype=torch.float64, device=device)
    squares = torch.zeros_like(total)
    count = 0
    for start in range(0, len(train), 64):
        rows = train[start:start + 64]
        views = torch.as_tensor(bank[rows], device=device).double()
        mask = torch.as_tensor(valid[rows], device=device)
        residual, usable = model.directional_views(views, mask)
        values = residual[usable]
        total += values.sum(dim=0)
        squares += values.square().sum(dim=0)
        count += len(values)
    if count:
        mean = total / count
        model.extra_mean.copy_(mean)
        model.extra_std.copy_((squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(0.001))


@torch.inference_mode()
def predict_features(model, store, indices, device):
    model.eval()
    parts = {key: [] for key in ("margin", "expert_margin", "weights")}
    for rows in np.array_split(indices, math.ceil(len(indices) / BATCH_SIZE)):
        out = model(*store.batch(rows, device))
        parts["margin"].append((out["logits"][:, 1] - out["logits"][:, 0]).double().cpu().numpy())
        experts = out["expert_logits"]
        parts["expert_margin"].append((experts[..., 1] - experts[..., 0]).cpu().numpy())
        parts["weights"].append(out["weights"].cpu().numpy())
    return {key: np.concatenate(batches) for key, batches in parts.items()}


def metrics(labels, margin, bias=0.0):
    shifted = margin + bias
    auc = float(roc_auc_score(labels, shifted)) if len(np.unique(labels)) == 2 else None
    return {
        "acc": float(((shifted > 0) == labels).mean()),
        "auc": auc,
        "nll": binary_nll(margin, labels, bias),
    }


def _training_loss(output, labels, smoothing):
    loss = F.cross_entropy(output["logits"], labels, label_smoothing=smoothing)
    experts = output["expert_logits"].reshape(-1, 2)
    targets = labels[:, None].expand(-1, 2).reshape(-1)
    expert_losses = F.cross_entropy(
        experts, targets, reduction="none", label_smoothing=smoothing
    ).reshape(-1, 2)
    available = output["expert_valid"]
    return loss + AUXILIARY_WEIGHT * (expert_losses * available).sum() / available.sum()


def train(feature_path, split_path, checkpoint_path, epochs, seed, device, *, temporal_groups=None, direction_groups=None):
    if checkpoint_path.exists():
        raise FileExistsError(checkpoint_path)
    store = FeatureStore(feature_path)
    split = load_split(split_path, len(store.labels))
    recipe = BENCHMARKS[store.dataset]
    val_labels = store.labels[split["val"]]
    if recipe.selection == "auc" and len(np.unique(val_labels)) != 2:
        raise ValueError("Validation AUC requires both classes.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    kwargs = dict(store.model_kwargs)
    for name, path in (("temporal_groups", temporal_groups), ("direction_groups", direction_groups)):
        if path is not None:
            kwargs[name] = np.load(path, allow_pickle=False).tolist()
    model = DirectionResidualMoE(**kwargs).to(device)
    fit_normalizer(model, store, split["train"], device)
    decayed, unregularized = [], []
    for name, param in model.named_parameters():
        group = unregularized if any(key in name for key in ("logit", "gate", "group_bias")) else decayed
        group.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decayed, "weight_decay": WEIGHT_DECAY},
        {"params": unregularized, "weight_decay": 0.0},
    ], lr=LEARNING_RATE)

    best_score, best_nll = -math.inf, math.inf
    train_indices = torch.as_tensor(split["train"], device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        if epoch <= WARMUP:
            scale = epoch / WARMUP
        else:
            scale = 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP) / (epochs - WARMUP)))
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * scale
        permutation = torch.randperm(len(train_indices), device=device)
        for ids in train_indices[permutation].split(BATCH_SIZE):
            rows = ids.cpu().numpy()
            labels = torch.as_tensor(store.labels[rows], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = _training_loss(model(*store.batch(rows, device)), labels, recipe.smoothing)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is nonfinite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        margin = predict_features(model, store, split["val"], device)["margin"]
        result = metrics(val_labels, margin)
        score, nll = result[recipe.selection], result["nll"]
        if score > best_score + 1e-12 or (abs(score - best_score) < 1e-12 and nll < best_nll):
            best_score, best_nll = score, nll
            best_epoch, best_margin, best_validation = epoch, margin, result
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    bias = fit_bias(best_margin, val_labels)
    checkpoint = {
        "dataset": store.dataset,
        "feature_config": store.config,
        "model_kwargs": kwargs,
        "model_state_dict": best_state,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "seed": seed,
        "split_hashes": split_hashes(split),
        "selection": recipe.selection,
        "calibration": {"bias": bias, "fit_split": "val"},
        "validation": {"raw": best_validation, "calibrated": metrics(val_labels, best_margin, bias)},
    }
    save_checkpoint(checkpoint, checkpoint_path)
    return {"best_epoch": best_epoch, **checkpoint["validation"]}


def evaluate(feature_path, split_path, checkpoint_path, device):
    store = FeatureStore(feature_path)
    split = load_split(split_path, len(store.labels))
    model, checkpoint, config = load_checkpoint(checkpoint_path, device)
    if checkpoint["dataset"] != store.dataset or config != store.config:
        raise ValueError("Checkpoint and feature store use different datasets or feature recipes.")
    expected = checkpoint.get("split_hashes") or checkpoint.get("signature", {}).get("split_hashes")
    if split_hashes(split) != expected:
        raise ValueError("Indices differ from the split recorded in the checkpoint.")
    predictions = predict_features(model, store, split["test"], device)
    labels = store.labels[split["test"]]
    margin = predictions["margin"]
    return {
        "raw": metrics(labels, margin),
        "calibrated": metrics(labels, margin, checkpoint["calibration"]["bias"]),
    }
