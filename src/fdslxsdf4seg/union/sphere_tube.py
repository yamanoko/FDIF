from typing import List

import torch

from fdslxsdf4seg.basic_sdf import Cylinder, Sphere
from fdslxsdf4seg.onioned_prism.onioned_sector_polygon_prism import (
    OnionedPentagonPrism,
    OnionedSquarePrism,
    OnionedTrianglePrism,
)
from fdslxsdf4seg.revolution.star_revolution import (
    FiveStarRevolution,
    FourStarRevolution,
    ThreeStarRevolution,
)
from fdslxsdf4seg.sdf_object import SphereTubeUnion


class SphereTriangleUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=Sphere,
            TubeClass=OnionedTrianglePrism,
        )


class SphereSquareUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=Sphere,
            TubeClass=OnionedSquarePrism,
        )


class SpherePentagonUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=Sphere,
            TubeClass=OnionedPentagonPrism,
        )


class SphereCylinderUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=Sphere,
            TubeClass=Cylinder,
        )


class ThreeStarRevolutionTriangleUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=ThreeStarRevolution,
            TubeClass=OnionedTrianglePrism,
        )


class ThreeStarRevolutionSquareUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=ThreeStarRevolution,
            TubeClass=OnionedSquarePrism,
        )


class ThreeStarRevolutionPentagonUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=ThreeStarRevolution,
            TubeClass=OnionedPentagonPrism,
        )


class ThreeStarRevolutionCylinderUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=ThreeStarRevolution,
            TubeClass=Cylinder,
        )


class FourStarRevolutionTriangleUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FourStarRevolution,
            TubeClass=OnionedTrianglePrism,
        )


class FourStarRevolutionSquareUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FourStarRevolution,
            TubeClass=OnionedSquarePrism,
        )


class FourStarRevolutionPentagonUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FourStarRevolution,
            TubeClass=OnionedPentagonPrism,
        )


class FourStarRevolutionCylinderUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FourStarRevolution,
            TubeClass=Cylinder,
        )


class FiveStarRevolutionTriangleUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FiveStarRevolution,
            TubeClass=OnionedTrianglePrism,
        )


class FiveStarRevolutionSquareUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FiveStarRevolution,
            TubeClass=OnionedSquarePrism,
        )


class FiveStarRevolutionPentagonUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FiveStarRevolution,
            TubeClass=OnionedPentagonPrism,
        )


class FiveStarRevolutionCylinderUnion(SphereTubeUnion):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
    ):
        super().__init__(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            sphere_radius=sphere_radius,
            tube_radius=tube_radius,
            tube_height=tube_height,
            SphereClass=FiveStarRevolution,
            TubeClass=Cylinder,
        )
