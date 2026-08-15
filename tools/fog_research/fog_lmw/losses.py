import torch
import torch.nn.functional as F


def slot_diversity_loss(z: torch.Tensor) -> torch.Tensor:
    """Penalize collapse of distinct latent slots onto the same direction."""
    z = F.normalize(z, dim=-1)
    sim = torch.matmul(z, z.transpose(-1, -2))
    k = sim.size(-1)
    eye = torch.eye(k, device=z.device, dtype=z.dtype).unsqueeze(0)
    off = sim * (1.0 - eye)
    return off.pow(2).sum() / max(z.size(0) * k * (k - 1), 1)


def route_entropy(route_probs: torch.Tensor | None) -> torch.Tensor | None:
    if route_probs is None:
        return None
    # Compute the logarithm in FP32. 1e-8 underflows to zero in FP16, which can
    # otherwise turn an inactive (zero-weighted) auxiliary term into NaN.
    p = route_probs.float().clamp_min(1e-8)
    return -(p * p.log()).sum(dim=-1).mean()
