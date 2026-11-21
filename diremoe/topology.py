"""Fixed node assignments and skew-preserving graph readout."""
import torch


def assignment_matrix(groups, channels):
    if groups is None:
        return None
    g = torch.as_tensor(groups)
    if g.ndim != 1 or len(g) != channels or g.dtype not in (torch.int32, torch.int64):
        raise ValueError("Expected one integer group ID for each input node.")
    unique = g.unique(sorted=True)
    if len(unique) < 2 or not torch.equal(unique, torch.arange(len(unique), device=g.device)):
        raise ValueError("Expected at least two nonempty groups numbered from zero.")
    matrix = torch.nn.functional.one_hot(g.long()).float()
    return matrix / matrix.sum(0)


def project_skew(edges, projection):
    channels, groups = projection.shape
    src, dst = torch.triu_indices(channels, channels, 1, device=edges.device)
    a, b = torch.triu_indices(groups, groups, 1, device=edges.device)
    matrix = edges.new_zeros((*edges.shape[:-1], channels, channels))
    matrix[..., src, dst] = edges
    matrix[..., dst, src] = -edges
    p = projection.to(edges.dtype)
    return (p.T @ matrix @ p)[..., a, b]
