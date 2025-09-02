# file: generate_sdf_dataset.py
import math
import random
from typing import List, Optional

import torch

from fdslxsdf4seg.sdf_2d import Hexagon, _SectorPolygonBase


# --- SDFベースクラス ---
class SDFObject:
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center: List[float] = None,
        transform: bool = False,
    ):
        self.device = device
        self.grid_size = grid_size  # [D, H, W]
        self.transform = transform
        T, R, S = (
            torch.eye(4, device=self.device),  # 平行移動行列
            torch.eye(4, device=self.device),  # 回転行列
            torch.eye(4, device=self.device),  # せん断行列
        )
        if center is not None:
            # 中心座標を指定する場合は平行移動行列を設定
            t_x, t_y, t_z = center
        else:
            t_x = random.uniform(-0.35, 0.35) * grid_size[0]
            t_y = random.uniform(-0.35, 0.35) * grid_size[1]
            t_z = random.uniform(-0.35, 0.35) * grid_size[2]
        T = self.tranlate_matrix(t_x, t_y, t_z)
        if transform:
            # 回転角度を生成
            angle_x = random.uniform(-torch.pi, torch.pi)
            angle_y = random.uniform(-torch.pi, torch.pi)
            angle_z = random.uniform(-torch.pi, torch.pi)
            # せん断量を生成
            shx = random.uniform(-0.1, 0.1)
            shy = random.uniform(-0.1, 0.1)
            shz = random.uniform(-0.1, 0.1)
            R, S = (
                self.rotate_matrix(angle_x, angle_y, angle_z),
                self.shear_matrix(shx, shy, shz),
            )
        # 変換行列を計算
        # T: 平行移動行列, R: 回転行列, S: せん断行列
        # 変換行列は T * R * S の順で適用される
        # 変換行列は4x4の行列
        self.transform_matrix = torch.matmul(torch.matmul(T, R), S)
        # 逆変換行列を計算
        self.inv_transform_matrix = torch.inverse(self.transform_matrix)

    def sdf(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x, y, z: meshgrid 上の座標テンソル (shape=(D,H,W))
        戻り値: 各点の signed distance (同shape)
        """
        x, y, z = self.applied_transform(x, y, z)
        return self._sdf(x, y, z)

    def _sdf(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x, y, z: meshgrid 上の座標テンソル (shape=(D,H,W))
        戻り値: 各点の signed distance (同shape)
        """
        raise NotImplementedError

    def tranlate_matrix(self, tx, ty, tz):
        """
        X, Y, Z 軸方向の平行移動行列を生成
        tx, ty, tz: 平行移動量
        戻り値: 平行移動行列 (4x4)
        """
        T = torch.tensor(
            [
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0, 1],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        return T

    def rotate_matrix(self, angle_x, angle_y, angle_z):
        """
        X, Y, Z 軸周りの回転行列を生成
        angle_x, angle_y, angle_z: ラジアン単位の回転角度
        戻り値: 回転行列 (4x4)
        """

        Rx = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, math.cos(angle_x), -math.sin(angle_x), 0],
                [0, math.sin(angle_x), math.cos(angle_x), 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        Ry = torch.tensor(
            [
                [math.cos(angle_y), 0, math.sin(angle_y), 0],
                [0, 1, 0, 0],
                [-math.sin(angle_y), 0, math.cos(angle_y), 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        Rz = torch.tensor(
            [
                [math.cos(angle_z), -math.sin(angle_z), 0, 0],
                [math.sin(angle_z), math.cos(angle_z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        return torch.matmul(torch.matmul(Rz, Ry), Rx)

    def shear_matrix(self, shx, shy, shz):
        """
        X, Y, Z 軸方向のせん断行列を生成
        shx, shy, shz: せん断量
        戻り値: せん断行列 (4x4)
        """
        S = torch.tensor(
            [
                [1, shx, shy, 0],
                [shx, 1, shz, 0],
                [shy, shz, 1, 0],
                [0, 0, 0, 1],
            ],
            device=self.device,
        )
        return S

    def applied_transform(self, x, y, z):
        """
        座標 (x, y, z) に逆変換行列を適用
        戻り値: 逆変換後の座標 (x', y', z')
        """

        coords = torch.stack(
            [x.flatten(), y.flatten(), z.flatten(), torch.ones_like(x.flatten())], dim=0
        )
        coords = coords.to(self.device)
        transformed_coords = torch.matmul(self.inv_transform_matrix, coords)
        transformed_coords = transformed_coords[:3, :].view(3, *x.shape)  # (3, D, H, W)
        return transformed_coords[0], transformed_coords[1], transformed_coords[2]


# --- 各種プリミティブ実装 （ランダム化ロジックをコンストラクタ内に移動） ---
class Sphere(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        transform=False,
        center=None,
        radius=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        # ランダム化
        if radius is None:
            radius = random.uniform(min(D, H, W) * 0.05, min(D, H, W) * 0.2)
        # パラメータ設定
        self.radius = radius

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
        return torch.norm(p, dim=0) - self.radius


class Torus(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if major_r is None:
            major_r = random.uniform(min(D, H) * 0.1, min(D, H) * 0.3)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.1, 0.3)
        self.R = major_r
        self.r = minor_r

    def _sdf(self, x, y, z):
        q = torch.stack([x, z], dim=0).norm(dim=0) - self.R
        q = torch.stack([q, y], dim=0).norm(dim=0) - self.r
        return q


class Cone(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H) * 0.05, min(D, H) * 0.2)
        if height is None:
            height = random.uniform(W * 0.2, W * 0.3)
        self.radius = radius
        self.height = height

    def _sdf(self, x, y, z):
        p = torch.stack(
            [
                torch.stack([x, z], dim=0).norm(dim=0) - self.radius,
                y + (self.height / 2),
            ],
            dim=0,
        )
        e = torch.stack(
            [
                torch.ones_like(p[0]) * -self.radius,
                torch.ones_like(p[0]) * self.height,
            ],
            dim=0,
        )
        q = p - e * torch.clamp(
            torch.sum(p * e, dim=0) / torch.sum(e * e, dim=0), min=0.0, max=1.0
        )
        d = torch.norm(q, dim=0)
        max_q = torch.max(q, dim=0).values
        mask = max_q > 0.0
        min_val = torch.min(torch.stack([d, p[1]], dim=0), dim=0).values
        return torch.where(mask, d, -min_val)


class Octahedron(SDFObject):
    def __init__(self, grid_size, device, center=None, transform=False, size=None):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if size is None:
            self.size = random.uniform(min(D, H, W) * 0.05, min(D, H, W) * 0.2)
        else:
            self.size = size

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0).abs()
        m = p.sum(dim=0) - self.size
        return m * 0.5773502691896257  # 1/sqrt(3)


class _PrismBase(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        height=None,
        axis=2,
    ):
        SDFObject.__init__(self, grid_size, device, center, transform)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if height is None:
            height = random.uniform(perp_min * 0.2, perp_min * 0.3)
        self.axis = axis
        self.h = float(height) / 2.0

    def sdf2d_scaled(self, X, Y, scale):
        """
        一様スケール s による SDF の性質: f_s(p) = s * f(p/s)
        X,Y,scale: (N,)
        """
        s = torch.clamp(scale, min=1e-6)
        Xs = X / s
        Ys = Y / s
        return s * self.sdf2d_base(Xs, Ys)

    @staticmethod
    def split_axes(x, y, z, axis: int):
        # axis に沿う座標を along、それと直交する2軸を X,Y とする
        if axis == 2:  # z
            X, Y, A = x, y, z
        elif axis == 1:  # y
            X, Y, A = x, z, y
        elif axis == 0:  # x
            X, Y, A = y, z, x
        else:
            raise ValueError("axis must be 0, 1, or 2.")
        return X, Y, A

    @staticmethod
    def extrude_combine(perp, along):
        """
        距離の合成（有界押し出しの標準手法）
        perp: 断面SDF（内:負）
        along: |along|-h（内:負）
        """
        stacked = torch.stack([perp, along], dim=0)
        outside = torch.clamp(stacked, min=0.0)  # 外側成分
        inside = torch.clamp(torch.max(stacked, dim=0).values, max=0.0)  # 内側成分
        return torch.norm(outside, dim=0) + inside

    def _sdf(self, x, y, z):
        X, Y, A = self.split_axes(x, y, z, self.axis)
        shp = X.shape
        Xf, Yf, Af = X.reshape(-1), Y.reshape(-1), A.reshape(-1)

        scale = self._scale_at(Af)
        perp = self.sdf2d_scaled(Xf, Yf, scale)
        along = torch.abs(Af) - self.h
        d = self.extrude_combine(perp, along)
        return d.view(*shp)

    def _scale_at(self, a):
        return torch.ones_like(a)  # デフォルトは一定スケール

    def sdf2d_base(self, X, Y):
        raise NotImplementedError


class _ConcavePrismBase(_PrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        if second_scale is None:
            second_scale = random.uniform(0.3, 0.8)
        if neck is None:
            neck = (2 * self.h) * random.uniform(-0.7, 0.7)
        self.second_scale = float(second_scale)
        self.neck = float(neck)

    def _scale_at(self, a):
        """
        a: (N,), 押し出し方向の座標。両端(-h,+h)で scale=1、neck で second_scale。
        Cylinder の線形補間式を踏襲。
        """
        h = self.h
        s1 = 1.0
        s2 = self.second_scale
        # neck より下側 [-h, neck]: s1 -> s2
        neg = (s2 - s1) / (self.neck + h + 1e-12) * (a + h) + s1
        # neck より上側 [neck, +h]: s2 -> s1
        pos = (s2 - s1) / (self.neck - h + 1e-12) * (a - self.neck) + s2
        return torch.where(a < self.neck, neg, pos)

    def sdf2d_base(self, X, Y):
        raise NotImplementedError


class _ConvexPrismBase(_PrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # < 1.0 推奨（くびれ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        if second_scale is None:
            second_scale = random.uniform(1.2, 1.8)
        if neck is None:
            neck = (2 * self.h) * random.uniform(-0.7, 0.7)
        self.second_scale = float(second_scale)
        self.neck = float(neck)

    def _scale_at(self, a):
        h = self.h
        s1 = 1.0
        s2 = self.second_scale
        neg = (s2 - s1) / (self.neck + h + 1e-12) * (a + h) + s1
        pos = (s2 - s1) / (self.neck - h + 1e-12) * (a - self.neck) + s2
        return torch.where(a < self.neck, neg, pos)

    def sdf2d_base(self, X, Y):
        raise NotImplementedError


class _ConePrismBase(_PrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        if second_scale is None:
            second_scale = random.uniform(0.3, 1.6)
        self.second_scale = float(second_scale)

    def _scale_at(self, a):
        h = self.h
        s1 = 1.0
        s2 = self.second_scale
        return (s2 - s1) / (2 * h) * (a + h) + s1

    def sdf2d_base(self, X, Y):
        raise NotImplementedError


class _PyramidPrismBase(_ConePrismBase):
    def __init__(
        self,
        grid_size,
        device,
        center=None,
        transform=False,
        height=None,
        axis=2,
        seed=None,
    ):
        second_scale = 0.0  # 頂点で0スケール
        super().__init__(
            grid_size, device, center, transform, height, second_scale, axis, seed
        )


class Cylinder(_PrismBase):
    """
    一定スケールの押し出し（通常のプリズム）
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.03 * perp_min, 0.20 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class ConvexCylinder(_ConvexPrismBase):
    """
    ふくらみ（barrel）。scale が中央（neck）で最大、両端で最小。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # > 1.0 推奨（ふくらみ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConvexPrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            neck,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.03 * perp_min, 0.20 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class ConcaveCylinder(_ConcavePrismBase):
    """
    くびれ（hourglass）。scale が中央（neck）で最小、両端で最大。
    Cylinder の Concave と同じ区分線形をスケールに適用。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # < 1.0 推奨（くびれ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConcavePrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            neck,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.03 * perp_min, 0.20 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class ConeCylinder(_ConePrismBase):
    """
    円錐台。scale が両端で一定、中央で second_scale。
    Cylinder の線形補間式を踏襲。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConePrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.03 * perp_min, 0.20 * perp_min)
        self.radius = radius

    def sdf2d_base(self, X, Y):
        return torch.norm(torch.stack([X, Y], dim=0), dim=0) - self.radius


class HexagonalPrism(_PrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.03 * perp_min, 0.20 * perp_min)
        self.radius = radius
        self.hex_base = Hexagon(radius=radius, device=device)

    def sdf2d_base(self, X, Y):
        return self.hex_base.sdf2d_base(X, Y)


class SectorPolygonPrism(_PrismBase):
    """
    一定スケールの押し出し（通常のプリズム）
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        n: Optional[int] = None,
        r1: Optional[float] = None,
        r2: Optional[float] = None,
        height: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(self, grid_size, device, center, transform, height, axis)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.03 * perp_min
            hi = 0.20 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = _SectorPolygonBase(
            n=n, r1=r1, r2=r2, seed=seed, device=device
        )  # 内部基底

    def sdf2d_base(self, X, Y):
        return self.poly_base.sdf2d_base(X, Y)


class ConcaveSectorPolygonPrism(_ConcavePrismBase):
    """
    くびれ（hourglass）。scale が中央（neck）で最小、両端で最大。
    Cylinder の Concave と同じ区分線形をスケールに適用。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        n: Optional[int] = None,
        r1: Optional[float] = None,
        r2: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # < 1.0 推奨（くびれ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConcavePrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            neck,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.03 * perp_min
            hi = 0.20 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = _SectorPolygonBase(
            n=n, r1=r1, r2=r2, seed=seed, device=device
        )  # 内部基底

    def sdf2d_base(self, X, Y):
        return self.poly_base.sdf2d_base(X, Y)


class ConvexSectorPolygonPrism(_ConvexPrismBase):
    """
    ふくらみ（barrel）。scale が中央（neck）で最大、両端で最小。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        n: Optional[int] = None,
        r1: Optional[float] = None,
        r2: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # > 1.0 推奨（ふくらみ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConvexPrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            neck,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.03 * perp_min
            hi = 0.20 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = _SectorPolygonBase(
            n=n, r1=r1, r2=r2, seed=seed, device=device
        )  # 内部基底

    def sdf2d_base(self, X, Y):
        return self.poly_base.sdf2d_base(X, Y)


class ConeSectorPolygonPrism(_ConePrismBase):
    """
    コーン（線形スケール）：下端(-h)で scale=1、上端(+h)で second_scale。
    second_scale <1 なら先細り、>1 なら先広がり。
    """

    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        n: Optional[int] = None,
        r1: Optional[float] = None,
        r2: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _ConePrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            second_scale,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.03 * perp_min
            hi = 0.20 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = _SectorPolygonBase(
            n=n, r1=r1, r2=r2, seed=seed, device=device
        )  # 内部基底

    def sdf2d_base(self, X, Y):
        return self.poly_base.sdf2d_base(X, Y)


class PyramidSectorPolygonPrism(_PyramidPrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        n: Optional[int] = None,
        r1: Optional[float] = None,
        r2: Optional[float] = None,
        height: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PyramidPrismBase.__init__(
            self,
            grid_size,
            device,
            center,
            transform,
            height,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.03 * perp_min
            hi = 0.20 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = _SectorPolygonBase(n=n, r1=r1, r2=r2, seed=seed, device=device)

    def sdf2d_base(self, X, Y):
        return self.poly_base.sdf2d_base(X, Y)


class TrianglePrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class SquarePrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class PentagonPrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class HexagonPrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class HeptagonPrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class OctagonPrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class NonagonPrism(SectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class TriangleConvexPrism(ConvexSectorPolygonPrism):
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


class SquareConvexPrism(ConvexSectorPolygonPrism):
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


class PentagonConvexPrism(ConvexSectorPolygonPrism):
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


class HexagonConvexPrism(ConvexSectorPolygonPrism):
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


class HeptagonConvexPrism(ConvexSectorPolygonPrism):
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


class OctagonConvexPrism(ConvexSectorPolygonPrism):
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


class NonagonConvexPrism(ConvexSectorPolygonPrism):
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
            axis=axis,
            seed=seed,
        )


class TrianglePyramidPrism(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class Pyramid(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class PentagonPyramidPrism(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class HexagonPyramidPrism(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class HeptagonPyramidPrism(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class OctagonPyramidPrism(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )


class NonagonPyramidPrism(PyramidSectorPolygonPrism):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        r1=None,
        r2=None,
        height=None,
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
            axis=axis,
            seed=seed,
        )
