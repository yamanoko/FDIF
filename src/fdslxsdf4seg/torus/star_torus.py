from typing import List

import torch

from fdslxsdf4seg.sdf_object import StarTorusBase


class FiveStarTorus(StarTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
        w=None,  # 0.1 < w < 0.7 推奨
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            major_r=major_r,
            minor_r=minor_r,
            n=5,
            w=w,
        )


class SixStarTorus(StarTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
        w=None,  # 0.1 < w < 0.7 推奨
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            major_r=major_r,
            minor_r=minor_r,
            n=6,
            w=w,
        )


class SevenStarTorus(StarTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
        w=None,  # 0.1 < w < 0.7 推奨
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            major_r=major_r,
            minor_r=minor_r,
            n=7,
            w=w,
        )


class EightStarTorus(StarTorusBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
        w=None,  # 0.1 < w < 0.7 推奨
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            major_r=major_r,
            minor_r=minor_r,
            n=8,
            w=w,
        )
