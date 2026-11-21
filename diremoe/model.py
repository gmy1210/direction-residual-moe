"""Temporal and spatial experts with masked, sample-wise logit fusion."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .topology import assignment_matrix, project_skew


class DirectionResidualMoE(nn.Module):
    """Binary classifier over temporal embeddings and signed Pearson views.

    View order is scale-major, lag-minor. View zero is full-recording static FC.
    """

    def __init__(
        self, *, n_channels: int, n_views: int, lags_samples: list[int],
        temporal_dim: int = 512, temporal_mode: str = "mean",
        hidden_dim: int = 32, roi_dim: int = 8, dropout: float = 0.3,
        temporal_prior: float = 0.1, direction_dim: int | None = None,
        temporal_nodes: int | None = None, temporal_groups=None, direction_groups=None,
    ) -> None:
        super().__init__()
        lags = tuple(lags_samples)
        if n_channels < 2 or min(temporal_dim, hidden_dim, roi_dim, n_views) < 1:
            raise ValueError("Invalid channel count or model dimensions.")
        if temporal_mode not in {"mean", "regional"}:
            raise ValueError("temporal_mode must be mean or regional.")
        if not 0 < temporal_prior < 1:
            raise ValueError("temporal_prior must lie in (0, 1).")
        if (
            not lags or lags[0] != 0 or len(lags) < 3 or len(set(lags)) != len(lags)
            or any(not isinstance(lag, int) or -lag not in lags for lag in lags)
            or n_views % len(lags)
        ):
            raise ValueError("Provide zero followed by unique paired signed integer lags for every scale.")
        self.n_channels = n_channels
        self.n_views = n_views
        self.edge_dim = n_channels * (n_channels - 1) // 2
        self.register_buffer("time_projection", assignment_matrix(temporal_groups, n_channels))
        self.register_buffer("direction_projection", assignment_matrix(direction_groups, n_channels))
        if self.time_projection is not None:
            if temporal_mode != "regional":
                raise ValueError("Temporal grouping requires regional embeddings.")
            temporal_nodes = self.time_projection.shape[1]
        if self.direction_projection is not None:
            k = self.direction_projection.shape[1]
            direction_dim = k * (k - 1) // 2
        self.direction_dim = self.edge_dim if direction_dim is None else direction_dim
        temporal_nodes = n_channels if temporal_nodes is None else temporal_nodes
        self.temporal_dim = temporal_dim
        self.temporal_mode = temporal_mode

        temporal_in = temporal_dim
        self.roi_projection = None
        if temporal_mode == "regional":
            self.roi_projection = nn.Sequential(nn.Linear(temporal_dim, roi_dim), nn.GELU())
            temporal_in = temporal_nodes * roi_dim
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_in, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.temporal_head = nn.Linear(hidden_dim, 2)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(self.edge_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.spatial_head = nn.Linear(hidden_dim, 2)
        self.router = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.router.weight)
        prior = torch.tensor([temporal_prior, 1 - temporal_prior])
        self.register_buffer("group_prior", prior)
        self.group_bias = nn.Parameter(prior.log())

        self.residual_encoder = nn.Sequential(
            nn.Linear(self.direction_dim, 8), nn.GELU(), nn.Linear(8, hidden_dim)
        )
        nn.init.zeros_(self.residual_encoder[-1].weight)
        nn.init.zeros_(self.residual_encoder[-1].bias)
        pairs = [
            (offset + index, offset + lags.index(-lag))
            for offset in range(0, n_views, len(lags))
            for index, lag in enumerate(lags) if lag > 0
        ]
        self.register_buffer("pairs", torch.tensor(pairs, dtype=torch.long))
        self.residual_logits = nn.Parameter(torch.zeros(len(pairs)))
        self.residual_gate = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))

        temporal_shape = (temporal_nodes, temporal_dim) if temporal_mode == "regional" else (temporal_dim,)
        for name, shape in (
            ("temporal", temporal_shape), ("static", (self.edge_dim,)), ("extra", (self.direction_dim,))
        ):
            self.register_buffer(f"{name}_mean", torch.zeros(shape))
            self.register_buffer(f"{name}_std", torch.ones(shape))

    def pool_time(self, temporal):
        if self.time_projection is None:
            return temporal
        return self.time_projection.T @ temporal.float()

    def directional_views(self, bank: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        positive, negative = self.pairs.unbind(dim=1)
        usable = valid[:, positive] & valid[:, negative] & valid[:, :1]
        residual = (bank[:, positive] - bank[:, negative]) * 0.5
        if self.direction_projection is not None:
            residual = project_skew(residual, self.direction_projection)
        return residual.masked_fill(~usable[..., None], 0.0), usable

    def forward(self, temporal: Tensor, bank: Tensor, valid: Tensor) -> dict[str, Tensor]:
        if bank.ndim != 3 or bank.shape[1:] != (self.n_views, self.edge_dim):
            raise ValueError("bank must have shape [batch, n_views, edge_dim].")
        if valid.shape != bank.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [batch, n_views].")
        temporal = self.pool_time(temporal)
        if temporal.shape != (len(bank), *self.temporal_mean.shape):
            raise ValueError("temporal embeddings do not match the configured mode and dimensions.")

        residual, usable = self.directional_views(bank.float(), valid)
        return self.forward_components(temporal, bank[:, 0], residual, valid, usable)

    def forward_components(self, temporal, anchor, residual, valid, usable):
        """Run on native or prepooled features; the static anchor always retains all edges."""
        temporal = ((temporal.float() - self.temporal_mean) / self.temporal_std).clamp(-8, 8)
        if self.roi_projection is not None:
            temporal = self.roi_projection(temporal).flatten(start_dim=1)
        temporal_hidden = self.temporal_encoder(temporal)

        anchor = ((anchor.float() - self.static_mean) / self.static_std).clamp(-8, 8)
        anchor = anchor.masked_fill(~valid[:, :1], 0.0)
        spatial_base = self.spatial_encoder(anchor)
        residual = ((residual.float() - self.extra_mean) / self.extra_std).clamp(-8, 8)
        residual = residual.masked_fill(~usable[..., None], 0.0)
        has_direction = usable.any(dim=1)
        scores = self.residual_logits.expand_as(usable).masked_fill(~usable, -torch.inf)
        scores = scores.masked_fill(~has_direction[:, None], 0.0)
        view_weights = scores.softmax(dim=1).masked_fill(~usable, 0.0)
        beta = 0.25 * self.residual_gate.sigmoid()
        effective_beta = beta * has_direction
        mixed_residual = (view_weights[..., None] * residual).sum(dim=1)
        correction = effective_beta[:, None] * self.residual_encoder(mixed_residual)
        spatial_hidden = spatial_base + correction

        expert_logits = torch.stack(
            [self.temporal_head(temporal_hidden), self.spatial_head(spatial_hidden)], dim=1
        )
        expert_valid = torch.stack([torch.ones_like(valid[:, 0]), valid[:, 0]], dim=1)
        hidden = torch.stack([temporal_hidden, spatial_hidden], dim=1)
        scores = self.group_bias + self.router(hidden.detach()).squeeze(-1)
        weights = scores.masked_fill(~expert_valid, -torch.inf).softmax(dim=1)
        anchor_weights = torch.zeros_like(valid, dtype=anchor.dtype)
        anchor_weights[:, 0] = valid[:, 0]
        base_rms = spatial_base.detach().square().mean(dim=-1).sqrt()
        return {
            "logits": (weights[..., None] * expert_logits).sum(dim=1),
            "expert_logits": expert_logits,
            "expert_valid": expert_valid,
            "weights": weights,
            "view_weights": anchor_weights,
            "residual_view_weights": view_weights,
            "beta": beta,
            "effective_beta": effective_beta,
            "residual_ratio": correction.square().mean(dim=-1).sqrt() / base_rms.clamp_min(1e-8),
        }
