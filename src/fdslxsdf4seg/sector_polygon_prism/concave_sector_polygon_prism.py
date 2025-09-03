from typing import List

import torch

from fdslxsdf4seg.sdf_object import ConcaveSectorPolygonPrism


class TriangleConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=3,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )


class SquareConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=4,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )


class PentagonConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=5,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )


class HexagonConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=6,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )


class HeptagonConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=7,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )


class OctagonConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=8,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )


class NonagonConcavePrism(ConcaveSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,  # < 1.0 推奨（くびれ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        axis=2,
        seed=None,
    ):
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            n=9,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            axis=axis,
            seed=seed,
        )
