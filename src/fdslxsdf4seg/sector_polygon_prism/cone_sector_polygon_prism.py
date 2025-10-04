from typing import List

import torch

from fdslxsdf4seg.sdf_object import ConeSectorPolygonPrism


class TriangleConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class SquareConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class PentagonConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class HexagonConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class HeptagonConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OctagonConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class NonagonConePrism(ConeSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
        second_scale=None,
        onion_ratio=None,
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
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )
