# file: generate_sdf_dataset.py
import math
import random
from typing import List, Optional

import torch

from fdslxsdf4seg.sdf_2d import SectorPolygonBase, StarBase


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
        T = self.translate_matrix(t_x, t_y, t_z)
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

    def onion(
        self,
        d: torch.Tensor,
        thickness_list: List[float],
    ) -> torch.Tensor:
        """
        オニオン化（複数の厚みで絶対値を取る操作）
        d: meshgrid 上の signed distance (shape=(D,H,W))
        thickness_list: オニオン化の厚みのリスト
        戻り値: 各点の signed distance (同shape)
        """
        d_onion = torch.abs(d)
        for thickness in thickness_list:
            d_onion = torch.abs(d_onion) - thickness
        return d_onion

    def translate_matrix(self, tx, ty, tz):
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

    def applied_transform(self, x, y, z, inv_matrix=None):
        """
        座標 (x, y, z) に逆変換行列を適用
        戻り値: 逆変換後の座標 (x', y', z')
        """
        if inv_matrix is not None:
            inv_matrix = inv_matrix.to(self.device)
        else:
            inv_matrix = self.inv_transform_matrix
        coords = torch.stack(
            [x.flatten(), y.flatten(), z.flatten(), torch.ones_like(x.flatten())], dim=0
        )
        coords = coords.to(self.device)
        transformed_coords = torch.matmul(inv_matrix, coords)
        transformed_coords = transformed_coords[:3, :].view(3, *x.shape)  # (3, D, H, W)
        return transformed_coords[0], transformed_coords[1], transformed_coords[2]


class SmoothUnionBase(SDFObject):
    def __init__(
        self,
        grid_size,
        device,
        center=None,
        transform=False,
    ):
        super().__init__(grid_size, device, center, transform)
        self.first_sdf = SDFObject(grid_size, device, center=(0, 0, 0), transform=False)
        self.second_sdf = SDFObject(
            grid_size, device, center=(0, 0, 0), transform=False
        )
        self.first_inv_matrix = torch.eye(4, device=device)
        self.second_inv_matrix = torch.eye(4, device=device)
        self.k = random.uniform(0.01, 0.15) * min(grid_size)  # スムーズパラメータ

    def _sdf(self, x, y, z):
        x1, y1, z1 = self.first_sdf.applied_transform(
            x, y, z, inv_matrix=self.first_inv_matrix
        )
        x2, y2, z2 = self.second_sdf.applied_transform(
            x, y, z, inv_matrix=self.second_inv_matrix
        )
        d1 = self.first_sdf._sdf(x1, y1, z1)
        d2 = self.second_sdf._sdf(x2, y2, z2)
        h = torch.clamp(self.k - torch.abs(d1 - d2), min=0.0)
        return torch.min(d1, d2) - h * h / (4.0 * self.k)


class SectorPolygonTorusBase(SDFObject):
    def __init__(
        self,
        grid_size,
        device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
        n=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != 1)
        if n is None:
            n = random.randint(3, 8)
        if major_r is None:
            major_r = perp_min * random.uniform(0.15, 0.40)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.4, 0.8)

        r1 = minor_r
        r2 = 0.03 * perp_min
        self.R = major_r
        self.r = minor_r
        self.sector_polygon_base = SectorPolygonBase(n=n, r1=r1, r2=r2, device=device)

    def _sdf(self, x, y, z):
        # 座標を平坦化してからトーラス座標系に変換
        shp = x.shape
        xf, yf, zf = x.reshape(-1), y.reshape(-1), z.reshape(-1)

        # トーラス座標系（主軸からの距離 - R、y軸方向の距離）
        q_dist = torch.stack([xf, zf], dim=0).norm(dim=0) - self.R
        result = self.sector_polygon_base.sdf2d_base(q_dist, yf)

        return result.view(*shp)


class StarTorusBase(SDFObject):
    def __init__(
        self,
        grid_size,
        device,
        center=None,
        transform=False,
        major_r=None,
        minor_r=None,
        n=None,
        w=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != 1)
        if major_r is None:
            major_r = perp_min * random.uniform(0.15, 0.40)
        if minor_r is None:
            minor_r = major_r * random.uniform(0.4, 0.8)
        self.R = major_r
        self.r = minor_r
        self.star_base = StarBase(n=n, w=w, radius=minor_r, device=device)

    def _sdf(self, x, y, z):
        # 座標を平坦化してからトーラス座標系に変換
        shp = x.shape
        xf, yf, zf = x.reshape(-1), y.reshape(-1), z.reshape(-1)

        # トーラス座標系（主軸からの距離 - R、y軸方向の距離）
        q_dist = torch.stack([xf, zf], dim=0).norm(dim=0) - self.R
        result = self.star_base.sdf2d_base(q_dist, yf)

        return result.view(*shp)


class _RevolutionBase(SDFObject):
    def __init__(
        self, grid_size, device, center=None, transform=False, axis=1, distance=None
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if distance is None:
            self.distance = random.uniform(min(D, H, W) * 0.05, min(D, H, W) * 0.1)
        else:
            self.distance = distance
        if axis >= 2 or axis < 0:
            raise ValueError("axis must be 0, 1, or 2.")
        self.axis = axis

    def sdf2d_base(self, X, Y):
        raise NotImplementedError

    def _sdf(self, x, y, z):
        # 座標を平坦化してから回転体座標系に変換
        shp = x.shape
        xf, yf, zf = x.reshape(-1), y.reshape(-1), z.reshape(-1)

        if self.axis == 0:
            q = torch.stack([yf, zf], dim=0).norm(dim=0) - self.distance
            result = self.sdf2d_base(xf, q)
        elif self.axis == 1:
            q = torch.stack([xf, zf], dim=0).norm(dim=0) - self.distance
            result = self.sdf2d_base(q, yf)

        return result.view(*shp)


class StarRevolutionBase(_RevolutionBase):
    def __init__(
        self,
        grid_size,
        device,
        center=None,
        transform=False,
        axis=1,
        distance=None,
        radius=None,
        n=None,
        w=None,
    ):
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H, W) * 0.15, min(D, H, W) * 0.4)
        if distance is None:
            distance = random.uniform(0.0, radius * 0.5)
        super().__init__(grid_size, device, center, transform, axis, distance)
        self.star_base = StarBase(n=n, w=w, radius=radius, device=device)

    def sdf2d_base(self, X, Y):
        return self.star_base.sdf2d_base(X, Y)


class _PrismBase(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        height=None,
        onion_ratio=None,
        axis=2,
    ):
        SDFObject.__init__(self, grid_size, device, center, transform)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if height is None:
            height = random.uniform(perp_min * 0.1, perp_min * 0.5)
        self.axis = axis
        self.h = float(height) / 2.0
        self.onion_ratio = onion_ratio

    def sdf2d_scaled(self, X, Y, scale):
        """
        一様スケール s による SDF の性質: f_s(p) = s * f(p/s)
        X,Y,scale: (N,)
        """
        s = torch.clamp(scale, min=1e-6)
        Xs = X / s
        Ys = Y / s
        return s * self.sdf2d_base(Xs, Ys)

    def sdf2d_onioned(self, d):
        if self.onion_ratio is not None:
            min_except_axis = torch.amin(
                d, dim=tuple(i for i in range(d.dim()) if i != self.axis), keepdim=True
            ).abs()
            thickness = self.onion_ratio * min_except_axis
            d = torch.abs(d) - thickness
        return d

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
        perp = self.sdf2d_onioned(perp)  # オニオン化
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
        onion_ratio: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(
            self, grid_size, device, center, transform, height, onion_ratio, axis
        )
        if second_scale is None:
            second_scale = random.uniform(0.2, 0.5)
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
        onion_ratio: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(
            self, grid_size, device, center, transform, height, onion_ratio, axis
        )
        if second_scale is None:
            second_scale = random.uniform(2.0, 4.0)
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
        onion_ratio: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(
            self, grid_size, device, center, transform, height, onion_ratio, axis
        )
        if second_scale is None:
            second_scale = random.uniform(0.0, 1.6)
        self.second_scale = float(second_scale)

    def _scale_at(self, a):
        h = self.h
        s1 = 1.0
        s2 = self.second_scale
        return (s2 - s1) / (2 * h) * (a + h) + s1

    def sdf2d_base(self, X, Y):
        raise NotImplementedError


class StarPrism(_PrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        n: Optional[int] = None,
        w: Optional[float] = None,
        height: Optional[float] = None,
        onion_ratio: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(
            self, grid_size, device, center, transform, height, onion_ratio, axis
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.15 * perp_min, 0.40 * perp_min)
        if n is None:
            n = random.randint(5, 10)
        if w is None:
            w = random.uniform(0.2, 0.7)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.star_base = StarBase(n=n, w=w, radius=radius, device=device)

    def sdf2d_base(self, X, Y):
        return self.star_base.sdf2d_base(X, Y)


class ConvexStarPrism(_ConvexPrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        n: Optional[int] = None,
        w: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # > 1.0 推奨（ふくらみ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        onion_ratio: Optional[float] = None,
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
            onion_ratio,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.05 * perp_min, 0.20 * perp_min)
        if n is None:
            n = random.randint(5, 10)
        if w is None:
            w = random.uniform(0.2, 0.7)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.star_base = StarBase(n=n, w=w, radius=radius, device=device)

    def sdf2d_base(self, X, Y):
        return self.star_base.sdf2d_base(X, Y)


class ConcaveStarPrism(_ConcavePrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        n: Optional[int] = None,
        w: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,  # < 1.0 推奨（くびれ）
        neck: Optional[float] = None,  # 中央からのバイアス位置 [-h, h]
        onion_ratio: Optional[float] = None,
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
            onion_ratio,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.15 * perp_min, 0.40 * perp_min)
        if n is None:
            n = random.randint(5, 10)
        if w is None:
            w = random.uniform(0.2, 0.7)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.star_base = StarBase(n=n, w=w, radius=radius, device=device)

    def sdf2d_base(self, X, Y):
        return self.star_base.sdf2d_base(X, Y)


class ConeStarPrism(_ConePrismBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius: Optional[float] = None,
        n: Optional[int] = None,
        w: Optional[float] = None,
        height: Optional[float] = None,
        second_scale: Optional[float] = None,
        onion_ratio: Optional[float] = None,
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
            onion_ratio,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if radius is None:
            radius = random.uniform(0.15 * perp_min, 0.40 * perp_min)
        if n is None:
            n = random.randint(5, 10)
        if w is None:
            w = random.uniform(0.2, 0.7)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.star_base = StarBase(n=n, w=w, radius=radius, device=device)

    def sdf2d_base(self, X, Y):
        return self.star_base.sdf2d_base(X, Y)


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
        onion_ratio: Optional[float] = None,
        axis: int = 2,
        seed: Optional[int] = None,
    ):
        _PrismBase.__init__(
            self, grid_size, device, center, transform, height, onion_ratio, axis
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.15 * perp_min
            hi = 0.40 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = SectorPolygonBase(
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
        onion_ratio: Optional[float] = None,
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
            onion_ratio,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.15 * perp_min
            hi = 0.40 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = SectorPolygonBase(
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
        onion_ratio: Optional[float] = None,
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
            onion_ratio,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.05 * perp_min
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
        self.poly_base = SectorPolygonBase(
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
        onion_ratio: Optional[float] = None,
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
            onion_ratio,
            axis,
            seed,
        )
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != axis)
        if n is None:
            n = random.randint(6, 16)
        if r1 is None or r2 is None:
            lo = 0.15 * perp_min
            hi = 0.40 * perp_min
            if r1 is None and r2 is None:
                r1 = random.uniform(lo, hi)
                r2 = random.uniform(lo, hi)
            elif r1 is None:
                r1 = random.uniform(lo, min(hi, r2))
            else:
                r2 = random.uniform(max(lo, r1), hi)
        if seed is None:
            seed = random.randint(0, 1 << 30)
        self.poly_base = SectorPolygonBase(
            n=n, r1=r1, r2=r2, seed=seed, device=device
        )  # 内部基底

    def sdf2d_base(self, X, Y):
        return self.poly_base.sdf2d_base(X, Y)


class SphereTubeUnion(SmoothUnionBase):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        sphere_radius=None,
        tube_radius=None,
        tube_height=None,
        SphereClass=None,
        TubeClass=None,
    ):
        if SphereClass is None:
            raise ValueError("SphereClass must be specified.")
        if TubeClass is None:
            raise ValueError("TubeClass must be specified.")
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        perp_min = min([D, H, W][i] for i in range(3) if i != 1)
        if sphere_radius is None:
            sphere_radius = perp_min * random.uniform(0.15, 0.40)
        if tube_height is None:
            tube_height = perp_min * random.uniform(0.1, 0.5)
        self.first_sdf = SphereClass(
            grid_size=grid_size, device=device, center=center, radius=sphere_radius
        )
        if tube_radius is None:
            tube_radius = self.first_sdf.radius * random.uniform(0.4, 0.8)

        if issubclass(
            TubeClass,
            (
                SectorPolygonPrism,
                ConeSectorPolygonPrism,
                ConvexSectorPolygonPrism,
                ConcaveSectorPolygonPrism,
            ),
        ):
            self.second_sdf = TubeClass(
                grid_size=grid_size,
                device=device,
                center=center,
                r2=tube_radius,
                height=tube_height,
            )
        else:
            self.second_sdf = TubeClass(
                grid_size=grid_size,
                device=device,
                center=center,
                radius=tube_radius,
                height=tube_height,
            )
        translate_distance = [
            (i == self.second_sdf.axis)
            * (tube_height / 2.0 + sphere_radius)
            * random.uniform(0.6, 0.9)
            for i in range(3)
        ]
        self.second_inv_matrix = self.translate_matrix(*translate_distance).inverse()
