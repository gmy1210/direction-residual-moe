import unittest

import torch

from diremoe.features import signed_pearson


class SignedPearsonTest(unittest.TestCase):
    def test_mixed_lengths_match_single_records_and_preserve_model_views(self):
        lengths = torch.tensor([232, 190, 232, 17, 190, 232])
        generator = torch.Generator().manual_seed(42)
        signals = torch.randn(len(lengths), 200, 232, generator=generator)
        signals[:, 0] = 1.0
        padding = torch.arange(signals.shape[-1]) >= lengths[:, None]
        signals.masked_fill_(padding[:, None, :], float("nan"))
        options = {
            "window_fractions": [1.0, 0.5, 0.25],
            "lags_samples": [0, 1, -1, 2, -2],
            "min_pairs": 16,
        }

        views, valid = signed_pearson(signals, lengths, **options)
        single = [
            signed_pearson(signal, length, **options)
            for signal, length in zip(signals.split(1), lengths.split(1))
        ]
        single_views, single_valid = (torch.cat(parts) for parts in zip(*single))
        self.assertTrue(torch.equal(views, single_views))
        self.assertTrue(torch.equal(valid, single_valid))

        pruned, pruned_valid = signed_pearson(
            signals, lengths, **options, include_short_static=False,
        )
        omitted = [5, 10]
        valid[:, omitted] = False
        views[:, omitted] = 0
        self.assertTrue(torch.equal(pruned_valid, valid))
        self.assertTrue(torch.equal(pruned, views))
