import random
from typing import List

import torch

from fdslxsdf4seg.basic_sdf import Cylinder
from fdslxsdf4seg.sector_polygon_prism.concave_sector_polygon_prism import (
    HeptagonConcavePrism,
    HexagonConcavePrism,
    NonagonConcavePrism,
    OctagonConcavePrism,
    PentagonConcavePrism,
    SquareConcavePrism,
    TriangleConcavePrism,
)
from fdslxsdf4seg.sector_polygon_prism.cone_sector_polygon_prism import (
    HeptagonConePrism,
    HexagonConePrism,
    NonagonConePrism,
    OctagonConePrism,
    PentagonConePrism,
    SquareConePrism,
    TriangleConePrism,
)
from fdslxsdf4seg.sector_polygon_prism.convex_sector_polygon_prism import (
    HeptagonConvexPrism,
    HexagonConvexPrism,
    NonagonConvexPrism,
    OctagonConvexPrism,
    PentagonConvexPrism,
    SquareConvexPrism,
    TriangleConvexPrism,
)
from fdslxsdf4seg.sector_polygon_prism.sector_polygon_prism import (
    HeptagonPrism,
    HexagonPrism,
    NonagonPrism,
    OctagonPrism,
    PentagonPrism,
    SquarePrism,
    TrianglePrism,
)


class OnionedCylinder(Cylinder):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
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
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedTrianglePrism(TrianglePrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSquarePrism(SquarePrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedPentagonPrism(PentagonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHexagonPrism(HexagonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHeptagonPrism(HeptagonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedOctagonPrism(OctagonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedNonagonPrism(NonagonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedTriangleConvexPrism(TriangleConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSquareConvexPrism(SquareConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedPentagonConvexPrism(PentagonConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHexagonConvexPrism(HexagonConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHeptagonConvexPrism(HeptagonConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedOctagonConvexPrism(OctagonConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedNonagonConvexPrism(NonagonConvexPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedTriangleConcavePrism(TriangleConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSquareConcavePrism(SquareConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedPentagonConcavePrism(PentagonConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHexagonConcavePrism(HexagonConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHeptagonConcavePrism(HeptagonConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedOctagonConcavePrism(OctagonConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedNonagonConcavePrism(NonagonConcavePrism):
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
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            neck=neck,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedTriangleConePrism(TriangleConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedSquareConePrism(SquareConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedPentagonConePrism(PentagonConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHexagonConePrism(HexagonConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedHeptagonConePrism(HeptagonConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedOctagonConePrism(OctagonConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )


class OnionedNonagonConePrism(NonagonConePrism):
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
        if onion_ratio is None:
            onion_ratio = random.uniform(0.2, 0.5)
        onion_ratio = min(onion_ratio, 1.0)
        onion_ratio = max(onion_ratio, 0.0)
        super().__init__(
            grid_size,
            device,
            center,
            transform,
            r1=r1,
            r2=r2,
            height=height,
            second_scale=second_scale,
            onion_ratio=onion_ratio,
            axis=axis,
            seed=seed,
        )
