"""SDFObject subclass that exposes MedShapeNet meshes as FDIF primitives.

A `MedShapeNetSDFObject` instance:
* picks a random mesh from the (dataset, organ_key) pool,
* lazily bakes its SDF onto a `grid_size`-shaped voxel volume,
* in `_sdf(x, y, z)` interpolates that volume at the transformed
  query coords (so existing rotation / translation / shear from
  `SDFObject` work unchanged), and
* returns an SDF tensor with **the same shape as the input coords**
  (consistent with all other `SDFObject` subclasses).

Cross-platform: no OpenGL, no display. Uses pysdf via `mesh_to_sdf.bake`.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fdslxsdf4seg.medshapenet import loader as _loader
from fdslxsdf4seg.medshapenet.mesh_to_sdf import bake
from fdslxsdf4seg.sdf_object import SDFObject

_FAR_SDF = 1e6  # fallback when no mesh is available


class MedShapeNetSDFObject(SDFObject):
    """SDFObject backed by a real medical mesh from MedShapeNetCore.

    Subclass parameters are bound by the dynamic class factory in
    `registry.py`: each concrete subclass fixes `dataset` and
    `organ_key` so that the resulting class behaves like any other
    primitive class (callable with `(grid_size, device, ...)`).
    """

    # Bound by subclasses generated in registry.py.
    dataset: str = ""
    organ_key: Optional[str] = None
    max_instances: int = 20

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center: Optional[List[float]] = None,
        transform: bool = False,
        instance_idx: Optional[int] = None,
        scale: Optional[float] = None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size

        # Voxel-space half-extent of the bounding cube the mesh occupies.
        # Matches the size range of other primitives (e.g. Sphere radius
        # in basic_sdf.Sphere uses 0.15..0.40 * min(D, H, W)).
        if scale is None:
            scale = random.uniform(0.20, 0.40) * float(min(D, H, W))
        self.scale = float(scale)

        meshes = _loader.load_meshes(
            self.dataset, self.organ_key, max_instances=self.max_instances
        )
        if not meshes:
            # No mesh available (download failed / organ_key absent).
            self._sdf_volume = None
            return

        if instance_idx is None:
            instance_idx = random.randrange(len(meshes))
        instance_idx = int(instance_idx) % len(meshes)
        self.instance_idx = instance_idx

        verts, faces = meshes[instance_idx]
        organ_tag = self.organ_key if self.organ_key is not None else "_single"
        sdf_np = bake(
            verts=verts,
            faces=faces,
            grid_size=(int(D), int(H), int(W)),
            dataset=self.dataset,
            organ_key=organ_tag,
        )
        # (1, 1, D, H, W) for grid_sample.
        self._sdf_volume = torch.from_numpy(sdf_np).to(device).unsqueeze(0).unsqueeze(0)

    def _sdf(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        if self._sdf_volume is None:
            return torch.full_like(x, _FAR_SDF)

        # Map from voxel space (object-centered) into the [-1, 1] normalized
        # cube the SDF was baked over.
        s = self.scale
        nx = x / s
        ny = y / s
        nz = z / s

        # grid_sample on a 5D volume needs grid of shape (N, D_o, H_o, W_o, 3)
        # with the last axis ordered (x, y, z) — where x indexes the W-dim
        # of the volume (the last axis), matching how `bake()` lays the
        # volume out as (D=z, H=y, W=x).
        grid = torch.stack([nx, ny, nz], dim=-1).unsqueeze(0)  # (1, D, H, W, 3)
        sampled = F.grid_sample(
            self._sdf_volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, 1, D, H, W)
        sdf_norm = sampled.squeeze(0).squeeze(0)
        # SDF values are in normalized units; rescale back to voxel units.
        return sdf_norm * s
