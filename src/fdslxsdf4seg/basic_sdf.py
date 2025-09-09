import random
from typing import List, Optional

import torch

from fdslxsdf4seg.sdf_object import (
    SDFObject,
    _ConcavePrismBase,
    _ConePrismBase,
    _ConvexPrismBase,
    _PrismBase,
)


class Sphere(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        transform=False,
        center=None,
        radius=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        # ランダム化
        if radius is None:
            radius = random.uniform(min(D, H, W) * 0.10, min(D, H, W) * 0.25)
        # パラメータ設定
        self.radius = radius

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
        return torch.norm(p, dim=0) - self.radius


class Torus(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if major_r is None:
            major_r = random.uniform(min(D, H) * 0.10, min(D, H) * 0.25)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.3, 0.7)
        self.R = major_r
        self.r = minor_r

    def _sdf(self, x, y, z):
        q = torch.stack([x, z], dim=0).norm(dim=0) - self.R
        q = torch.stack([q, y], dim=0).norm(dim=0) - self.r
        return q


class Cone(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H) * 0.10, min(D, H) * 0.250)
        if height is None:
            height = random.uniform(W * 0.1, W * 0.35)
        self.radius = radius
        self.height = height

    def _sdf(self, x, y, z):
        p = torch.stack(
            [
                torch.stack([x, z], dim=0).norm(dim=0) - self.radius,
                y + (self.height / 2),
            ],
            dim=0,
        )
        e = torch.stack(
            [
                torch.ones_like(p[0]) * -self.radius,
                torch.ones_like(p[0]) * self.height,
            ],
            dim=0,
        )
        q = p - e * torch.clamp(
            torch.sum(p * e, dim=0) / torch.sum(e * e, dim=0), min=0.0, max=1.0
        )
        d = torch.norm(q, dim=0)
        max_q = torch.max(q, dim=0).values
        mask = max_q > 0.0
        min_val = torch.min(torch.stack([d, p[1]], dim=0), dim=0).values
        return torch.where(mask, d, -min_val)


class Octahedron(SDFObject):
    def __init__(self, grid_size, device, center=None, transform=False, size=None):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if size is None:
            self.size = random.uniform(min(D, H, W) * 0.10, min(D, H, W) * 0.250)
        else:
            self.size = size

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0).abs()
        m = p.sum(dim=0) - self.size
        return m * 0.35773502691896257  # 1/sqrt(3)


class Cylinder(_PrismBase):
    """
    一定スケールの押し出し（通常のプリズム）
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.100 * perp_min, 0.250 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class ConvexCylinder(_ConvexPrismBase):
    """
    ふくらみ（barrel）。scale が中央（neck）で最大、両端で最小。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # > 1.0 推奨（ふくらみ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConvexPrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            neck,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.10 * perp_min, 0.250 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class ConcaveCylinder(_ConcavePrismBase):
    """
    くびれ（hourglass）。scale が中央（neck）で最小、両端で最大。
    Cylinder の Concave と同じ区分線形をスケールに適用。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # < 1.0 推奨（くびれ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConcavePrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            neck,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.10 * perp_min, 0.250 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class ConeCylinder(_ConePrismBase):
    """
    円錐台。scale が両端で一定、中央で second_scale。
    Cylinder の線形補間式を踏襲。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConePrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.10 * perp_min, 0.250 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius
