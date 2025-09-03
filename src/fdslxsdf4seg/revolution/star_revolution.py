from typing import List

import torch

from fdslxsdf4seg.sdf_object import _StarRevolutionBase


class ThreeStarRevolution(_StarRevolutionBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        distance=None,
        radius=None,
        w=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            distance=distance,
            radius=radius,
            n=3,
            w=w,
        )


class FourStarRevolution(_StarRevolutionBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        distance=None,
        radius=None,
        w=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            distance=distance,
            radius=radius,
            n=4,
            w=w,
        )


class FiveStarRevolution(_StarRevolutionBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        distance=None,
        radius=None,
        w=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            distance=distance,
            radius=radius,
            n=5,
            w=w,
        )
