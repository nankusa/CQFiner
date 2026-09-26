"""Uniform continuous sampling in the union of atom-centered balls."""
import hashlib
import math

import torch
from torch import Tensor
from torch_cluster import radius as radius_neighbors


def sample_seed(seed: int, sample_id: str, epoch: int | None) -> int:
    phase = "evaluation" if epoch is None else f"training_epoch_{epoch}"
    digest = hashlib.sha256(f"{seed}:{sample_id}:{phase}:atom_volume".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


@torch.no_grad()
def sample_atom_volume(atoms: Tensor, count: int, radius: float, generator: torch.Generator) -> Tensor:
    """Uniform bounding-box proposals, accepted iff within radius of any atom.

    Overlapping atom neighborhoods do not get extra probability mass. Fixed-size
    proposals make smaller requested counts prefixes of larger counts with the
    same seed. No surface normals, offsets, discretized pool or repeated padding.
    """
    if atoms.ndim != 2 or atoms.shape[1] != 3 or not len(atoms):
        raise ValueError("Protein atoms must have nonempty shape [N, 3].")
    if not torch.isfinite(atoms).all():
        raise ValueError("Protein atoms contain nonfinite coordinates.")
    if type(count) is not int or count <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Expected positive integer count and finite positive radius.")
    with torch.autocast(device_type=atoms.device.type, enabled=False):
        atoms = atoms.float().contiguous()
        low, high = atoms.amin(0)-radius, atoms.amax(0)+radius
        accepted = []
        total = 0
        for _ in range(4096):
            proposals = low + torch.rand((1024,3),device=atoms.device,dtype=torch.float32,generator=generator)*(high-low)
            # One neighbor is sufficient for membership in the union, not for graph construction.
            edges = radius_neighbors(atoms, proposals, r=radius, max_num_neighbors=1)
            mask = torch.zeros(len(proposals),dtype=torch.bool,device=atoms.device)
            mask[edges[0]] = True
            points = proposals[mask]
            accepted.append(points)
            total += len(points)
            if total >= count:
                result = torch.cat(accepted)[:count]
                if len(torch.unique(result,dim=0)) != count:
                    raise RuntimeError("Continuous atom-volume sampling generated duplicate points.")
                return result
    raise RuntimeError(f"Atom-volume rejection sampler exhausted 4194304 proposals: accepted {total}/{count}.")
