from typing import List

import torch

from fdslxsdf4seg.sdf_object import ConvexStarPrism


class FiveStarConvexPrism(ConvexStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,
        height=None,
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            n=5,
            w=w,
            height=height,
            axis=axis,
            seed=seed,
        )


class SixStarConvexPrism(ConvexStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,
        height=None,
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            n=6,
            w=w,
            height=height,
            axis=axis,
            seed=seed,
        )


class SevenStarConvexPrism(ConvexStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,
        height=None,
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            n=7,
            w=w,
            height=height,
            axis=axis,
            seed=seed,
        )


class EightStarConvexPrism(ConvexStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,
        height=None,
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            n=8,
            w=w,
            height=height,
            axis=axis,
            seed=seed,
        )
