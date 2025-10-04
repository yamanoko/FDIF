from typing import List

import torch

from fdslxsdf4seg.sdf_object import ConcaveStarPrism


class FiveStarConcavePrism(ConcaveStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
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
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class SixStarConcavePrism(ConcaveStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
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
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class SevenStarConcavePrism(ConcaveStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
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
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class EightStarConcavePrism(ConcaveStarPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
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
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )
