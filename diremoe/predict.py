"""Inference from preprocessed, channel-aligned recordings."""

from torch import nn, inference_mode

from .checkpoint import load_checkpoint
from .features import FeatureExtractor, MANTIS_REPO


class Predictor(nn.Module):
    def __init__(self, checkpoint, weights: str = MANTIS_REPO, device: str = "cpu"):
        super().__init__()
        self.model, record, config = load_checkpoint(checkpoint, device)
        self.features = FeatureExtractor(config, weights, device)
        self.bias = float(record["calibration"]["bias"])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @inference_mode()
    def forward(self, signals, lengths=None, *, tr_seconds=None, include_short_static: bool = True):
        """Predict and report validity for the computed connectivity views."""
        temporal, bank, valid = self.features(
            signals, lengths, tr_seconds=tr_seconds, include_short_static=include_short_static,
        )
        output = self.model(temporal, bank, valid)
        margin = (output["logits"][:, 1] - output["logits"][:, 0]).double()
        return {
            **output,
            "valid": valid,
            "margin": margin,
            "calibrated_margin": margin + self.bias,
            "probability_class1": margin.sigmoid(),
            "calibrated_probability_class1": (margin + self.bias).sigmoid(),
        }
