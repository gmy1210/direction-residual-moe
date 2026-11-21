"""Feature and training settings for REST-MDD and ABIDE."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Benchmark:
    channels: int
    temporal_mode: str
    lags: tuple[int, ...]
    selection: str = "acc"
    smoothing: float = 0.0

    def feature_config(self, physical_lags=None, regional=False) -> dict:
        extra = {}
        if physical_lags is not None:
            if (len(physical_lags) != len(self.lags[1::2])
                    or any(not math.isfinite(v) or v <= 0 for v in physical_lags)
                    or list(physical_lags) != sorted(set(physical_lags))):
                raise ValueError("Provide two positive, increasing physical lags in seconds.")
            extra["lags_seconds"] = [0., *(d for lag in physical_lags for d in (lag, -lag))]
        return {
            **extra,
            "n_channels": self.channels,
            "temporal_mode": "regional" if regional else self.temporal_mode,
            "lags_samples": list(self.lags),
            "window_fractions": [1.0, 0.5, 0.25],
            "min_pairs": 16,
        }


def matches_recipe(dataset, config):
    lags = config.get("lags_seconds")
    try:
        return config == BENCHMARKS[dataset].feature_config(
            None if lags is None else lags[1::2], regional=config.get('temporal_mode') == 'regional')
    except (TypeError, ValueError, KeyError):
        return False


BENCHMARKS = {
    "rest_mdd": Benchmark(200, "mean", (0, 1, -1, 2, -2)),
    "abide": Benchmark(200, "mean", (0, 1, -1, 2, -2), "auc", 0.1),
}
