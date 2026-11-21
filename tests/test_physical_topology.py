import unittest

import torch

from diremoe.config import BENCHMARKS, matches_recipe
from diremoe.features import physical_pearson, signed_pearson
from diremoe.model import DirectionResidualMoE
from diremoe.topology import assignment_matrix, project_skew


class PhysicalTopologyTest(unittest.TestCase):
    def test_native_identity_and_invalid_sampling(self):
        torch.manual_seed(14)
        x = torch.randn(2, 4, 54)
        lengths = torch.tensor([54, 47])
        x[1, :, 47:] = torch.nan
        options = dict(window_fractions=[1., .5], min_pairs=16)
        frames = signed_pearson(x, lengths, lags_samples=[0, 1, -1, 2, -2], **options)
        aligned = physical_pearson(x, lengths, tr_seconds=[2., 2.], lags_seconds=[0, 2, -2, 4, -4], **options)
        for a,b in zip(frames, aligned):
            self.assertTrue(torch.equal(a,b))
        with self.assertRaises(ValueError):
            physical_pearson(x, lengths, tr_seconds=[2.,0.], lags_seconds=[0,2,-2], **options)
        cfg = BENCHMARKS["rest_mdd"].feature_config([2,4], regional=True)
        self.assertTrue(matches_recipe("rest_mdd",cfg))

    def test_fractional_direction_and_overlap(self):
        t = 2.5 * torch.arange(201)
        x = torch.stack([torch.sin(2*torch.pi*.04*(t-2)), torch.sin(2*torch.pi*.04*t)])[None]
        bank, valid = physical_pearson(x, torch.tensor([201]), tr_seconds=[2.5],
                                      lags_seconds=[0,2,-2], window_fractions=[1.], min_pairs=16)
        self.assertTrue(valid.all())
        self.assertGreater(bank[0,1,0].item(),.999)
        self.assertLess(bank[0,2,0].item(),.57)
        _, short = physical_pearson(x[...,:16],torch.tensor([16]),tr_seconds=[2.5],
                                    lags_seconds=[0,2,-2],window_fractions=[1.],min_pairs=16)
        self.assertEqual(short.tolist(),[[True,False,False]])

    def test_topology_permutation_and_normal_entry(self):
        torch.manual_seed(2)
        g = [0,1,0,2,1]
        p = assignment_matrix(g,5)
        a = torch.randn(2,5,5)
        a = a-a.transpose(-1,-2)
        i,j = torch.triu_indices(5,5,1)
        perm = torch.tensor([4,1,3,0,2])
        reference = project_skew(a[...,i,j],p)
        other = a[:,perm][:,:,perm]
        torch.testing.assert_close(reference,project_skew(other[...,i,j],p[perm]))
        torch.testing.assert_close(-reference,project_skew(-a[...,i,j],p))
        opts=dict(n_channels=5, n_views=5, lags_samples=[0,1,-1,2,-2],temporal_dim=4,
                  temporal_mode="regional",temporal_groups=g,direction_groups=g)
        m=DirectionResidualMoE(**opts).eval()
        h,bank,valid=torch.randn(2,5,4),torch.randn(2,5,10),torch.ones(2,5,dtype=torch.bool)
        raw=m(h,bank,valid)
        r,mask=m.directional_views(bank,valid)
        compact=m.forward_components(p.T@h,bank[:,0],r,valid,mask)
        torch.testing.assert_close(raw["logits"],compact["logits"])
        self.assertEqual(r.shape,(2,2,3))
