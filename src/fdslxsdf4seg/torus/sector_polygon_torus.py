from typing import List

import torch

from fdslxsdf4seg.sdf_object import SectorPolygonTorusBase


class SquareTorus(SectorPolygonTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=4,
            major_r=major_r,
            minor_r=minor_r,
        )


class PentagonTorus(SectorPolygonTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=5,
            major_r=major_r,
            minor_r=minor_r,
        )


class HexagonTorus(SectorPolygonTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=6,
            major_r=major_r,
            minor_r=minor_r,
        )


class HeptagonTorus(SectorPolygonTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=7,
            major_r=major_r,
            minor_r=minor_r,
        )


class OctagonTorus(SectorPolygonTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=8,
            major_r=major_r,
            minor_r=minor_r,
        )


class NonagonTorus(SectorPolygonTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=9,
            major_r=major_r,
            minor_r=minor_r,
        )
