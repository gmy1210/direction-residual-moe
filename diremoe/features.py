"""Frozen temporal embeddings and multi-scale, signed-lag Pearson views."""

from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MANTIS_REPO = "paris-noah/MantisV2"


def _window_starts(length: int, window: int) -> list[int]:
    last = length - window
    starts = list(range(0, last + 1, max(1, window // 2)))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _center_and_unit_norm(windows: Tensor) -> Tensor:
    centered = windows - windows.mean(dim=-1, keepdim=True)
    return centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _lagged_pairs(x: Tensor, delay: float):
    """Native-time anchors with a reflected, Lanczos-windowed fractional shift."""
    nearest = round(delay)
    if abs(delay - nearest) < 1e-9:
        return (x[..., nearest:], x[..., :-nearest]) if nearest else (x, x)
    overlap = x.shape[-1] - math.ceil(delay)
    floor = math.floor(delay)
    taps = torch.arange(-7, 9, device=x.device)
    distance = (delay - floor) - taps.to(torch.float64)
    weights = torch.sinc(distance) * torch.sinc(distance / 8)
    weights = (weights / weights.sum()).to(x.dtype)
    indices = torch.arange(overlap, device=x.device)[:, None] + floor + taps
    period = 2 * (x.shape[-1] - 1)
    indices = indices.remainder(period)
    indices = torch.minimum(indices, period - indices)
    late = (x[..., indices] * weights).sum(dim=-1)
    return late, x[..., :overlap]


def physical_pearson(signals, lengths, *, tr_seconds, lags_seconds, **options):
    """Estimate signed Pearson views in seconds on the native sampling grid."""
    tr = torch.as_tensor(tr_seconds, dtype=torch.float64, device=signals.device)
    if tr.shape != (len(signals),) or not bool((torch.isfinite(tr) & (tr > 0)).all()):
        raise ValueError("Provide a positive sampling interval in seconds per recording.")
    channels = signals.shape[1]
    views = len(options["window_fractions"]) * len(lags_seconds)
    bank = signals.new_zeros((len(signals), views, channels*(channels-1)//2), dtype=torch.float32)
    valid = torch.zeros((len(signals), views), device=signals.device, dtype=torch.bool)
    for interval in tr.unique():
        ids = torch.where(tr == interval)[0]
        delays = [float(v) / interval.item() for v in lags_seconds]
        part, mask = signed_pearson(signals[ids], lengths[ids], lags_samples=delays, **options)
        bank[ids], valid[ids] = part, mask
    return bank, valid


@torch.no_grad()
def signed_pearson(
    signals: Tensor, lengths: Tensor, *, window_fractions: list[float],
    lags_samples: list[int], min_pairs: int, include_short_static: bool = True,
) -> tuple[Tensor, Tensor]:
    """Average windowed upper-triangular Pearson edges in scale-major order.

    Positive lag correlates a later row ROI with an earlier column ROI.
    Negative lags reuse the same correlation matrix with its indices reversed.
    Set include_short_static=False to leave sub-full-length zero-lag slots
    empty and invalid when only the full-recording anchor is needed.
    """
    batch, channels, _ = signals.shape
    n_lags = len(lags_samples)
    n_views = len(window_fractions) * n_lags
    row, col = torch.triu_indices(channels, channels, 1, device=signals.device)
    bank = signals.new_zeros(batch, n_views, len(row), dtype=torch.float32)
    valid = torch.zeros(bank.shape[:2], dtype=torch.bool, device=signals.device)
    triangles = (row * channels + col, col * channels + row)

    by_length = defaultdict(list)
    for sample, length in enumerate(lengths.tolist()):
        by_length[length].append(sample)
    by_delay = defaultdict(list)
    for lag_index, lag in enumerate(lags_samples):
        by_delay[abs(lag)].append((lag_index, lag < 0))

    with torch.autocast(device_type=signals.device.type, enabled=False):
        for length, samples in by_length.items():
            ids = torch.tensor(samples, device=signals.device)
            for scale, fraction in enumerate(window_fractions):
                width = round(length * fraction)
                delays = [
                    (delay, views) for delay, views in by_delay.items()
                    if width - math.ceil(delay - 1e-9) >= min_pairs and (delay or include_short_static or fraction >= 1)
                ]
                if not delays:
                    continue
                starts = _window_starts(length, width)
                n_windows = len(starts)
                # Single-matrix products preserve rounding before float16 storage.
                rows_per_chunk = 1 if n_windows == 1 or signals.device.type == 'cpu' else max(1, 64 // n_windows)
                for rows in ids.split(rows_per_chunk):
                    native = signals[..., :length].index_select(0, rows).float()
                    windows = torch.stack(
                        [native[..., start:start + width] for start in starts], dim=1
                    ).flatten(0, 1)
                    for delay, views in delays:
                        shifted, anchors = _lagged_pairs(windows, delay)
                        late = _center_and_unit_norm(shifted)
                        early = _center_and_unit_norm(anchors) if delay else late
                        matrix = torch.bmm(late, early.transpose(1, 2)).clamp_(-1, 1)
                        matrix = matrix.flatten(1)
                        for lag_index, reverse in views:
                            view = scale * n_lags + lag_index
                            edges = matrix.index_select(1, triangles[reverse])
                            edges = edges.unflatten(0, (len(rows), n_windows)).mean(dim=1)
                            bank[:, view].index_copy_(0, rows, edges)
                            valid[:, view].index_fill_(0, rows, True)
    return bank, valid


class FeatureExtractor(nn.Module):
    """Feature extraction for preprocessed, channel-aligned recordings."""

    def __init__(self, config: dict, weights: str = MANTIS_REPO, device: str = "cpu"):
        super().__init__()
        try:
            from mantis.architecture import MantisV2
        except ImportError as exc:
            raise ImportError("Install diremoe[mantis] to extract features from signals.") from exc

        self.config = config
        encoder = MantisV2(
            num_patches=32, output_token="combined", return_transf_layer=2,
            pre_training=False, device="cpu",
        )
        self.encoder = encoder.from_pretrained(str(weights)).to(device)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FeatureExtractor:
        return super().train(False)

    @torch.no_grad()
    def forward(
        self, signals, lengths=None, *, tr_seconds=None, include_short_static: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = next(self.encoder.parameters()).device
        signals = torch.as_tensor(signals, dtype=torch.float32, device=device)
        if signals.ndim == 2:
            signals = signals.unsqueeze(0)
        if signals.ndim != 3 or not len(signals) or signals.shape[1] != self.config["n_channels"]:
            raise ValueError("signals must be [B,C,T] or [C,T] with the configured channel count.")
        batch, channels, time = signals.shape
        if lengths is None:
            lengths = [time] * batch
        lengths = torch.as_tensor(lengths, device=device)
        if (
            lengths.shape != (batch,)
            or not bool(torch.isfinite(lengths).all())
            or not bool(((lengths == lengths.long()) & (lengths >= 2) & (lengths <= time)).all())
        ):
            raise ValueError("lengths must contain one integer in [2,T] per recording.")
        lengths = lengths.long()
        time_valid = torch.arange(time, device=device) < lengths[:, None]
        signals = signals.masked_fill(~time_valid[:, None, :], 0.0)
        if not bool(torch.isfinite(signals).all()):
            raise ValueError("Signals contain NaN or Inf in the observed portion.")

        resized = torch.cat([
            F.interpolate(
                signals[i:i + 1, :, :n].reshape(channels, 1, n),
                size=512, mode="linear", align_corners=False,
            )
            for i, n in enumerate(lengths.tolist())
        ])
        embeddings = self.encoder(resized).reshape(batch, channels, -1).float()
        if self.config["temporal_mode"] == "regional":
            temporal = embeddings.half()
        else:
            temporal = embeddings.mean(dim=1)
        physical = "lags_seconds" in self.config
        if physical and tr_seconds is None:
            raise ValueError("A physical-lag model requires tr_seconds for every recording.")
        options = ({"tr_seconds": tr_seconds, "lags_seconds": self.config["lags_seconds"]}
                   if physical else {"lags_samples": self.config["lags_samples"]})
        bank, valid = (physical_pearson if physical else signed_pearson)(
            signals, lengths, window_fractions=self.config["window_fractions"],
            **options, min_pairs=self.config["min_pairs"],
            include_short_static=include_short_static,
        )
        return temporal, bank.half(), valid
