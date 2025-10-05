import random
from typing import List

import torch

from fdslxsdf4seg.star_polygon_prism.concave_star_prism import (
    EightStarConcavePrism,
    FiveStarConcavePrism,
    SevenStarConcavePrism,
    SixStarConcavePrism,
)
from fdslxsdf4seg.star_polygon_prism.cone_star_prism import (
    EightStarConePrism,
    FiveStarConePrism,
    SevenStarConePrism,
    SixStarConePrism,
)
from fdslxsdf4seg.star_polygon_prism.convex_star_prism import (
    EightStarConvexPrism,
    FiveStarConvexPrism,
    SevenStarConvexPrism,
    SixStarConvexPrism,
)
from fdslxsdf4seg.star_polygon_prism.star_prism import (
    EightStarPrism,
    FiveStarPrism,
    SevenStarPrism,
    SixStarPrism,
)


class OnionedFiveStarPrism(FiveStarPrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSixStarPrism(SixStarPrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSevenStarPrism(SevenStarPrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedEightStarPrism(EightStarPrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedFiveStarConvexPrism(FiveStarConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # > 1.0 推奨（ふくらみ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSixStarConvexPrism(SixStarConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # > 1.0 推奨（ふくらみ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSevenStarConvexPrism(SevenStarConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # > 1.0 推奨（ふくらみ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedEightStarConvexPrism(EightStarConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,  # > 1.0 推奨（ふくらみ）
        neck=None,  # 中央からのバイアス位置 [-h, h]
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedFiveStarConcavePrism(FiveStarConcavePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSixStarConcavePrism(SixStarConcavePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSevenStarConcavePrism(SevenStarConcavePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedEightStarConcavePrism(EightStarConcavePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedFiveStarConePrism(FiveStarConePrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSixStarConePrism(SixStarConePrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSevenStarConePrism(SevenStarConePrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedEightStarConePrism(EightStarConePrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        w=None,  # 0.1 < w < 0.7 推奨
        height=None,
        second_scale=None,
        onion_ratio=None,
        axis=2,
        seed=None,
    ):
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            radius=radius,
            w=w,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )
