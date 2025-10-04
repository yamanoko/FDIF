from typing import List

import torch

from fdslxsdf4seg.sdf_object import StarPrism


class FiveStarPrism(StarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class SixStarPrism(StarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class SevenStarPrism(StarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class EightStarPrism(StarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )
