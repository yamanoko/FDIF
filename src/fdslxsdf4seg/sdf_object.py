# file: generate_sdf_dataset.py
import random
from typing import List

import torch


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

    def max_distance(self) -> float:
        """
        このオブジェクトのバウンディング半径相当
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
        import math

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

    def max_distance(self):
        return self.radius


class Box(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        half_extents=None,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if half_extents is None:
            hx = random.uniform(D * 0.05, D * 0.2)
            hy = random.uniform(H * 0.05, H * 0.2)
            hz = random.uniform(W * 0.05, W * 0.2)
            half_extents = (hz, hy, hx)
        self.half = torch.tensor(half_extents, device=device).view(3, 1, 1, 1)

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
        q = torch.abs(p) - self.half
        outside = torch.clamp(q, min=0.0)
        inside = torch.clamp(torch.max(q, dim=0).values, max=0.0)
        return torch.norm(outside, dim=0) + inside

    def max_distance(self):
        return float(torch.norm(self.half))


class Cylinder(SDFObject):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
        axis=2,
    ):
        super().__init__(grid_size, device, center, transform)
        D, H, W = grid_size
        if radius is None:
            radius = random.uniform(min(D, H) * 0.1, min(D, H) * 0.2)
        if height is None:
            height = random.uniform(W * 0.2, W * 0.3)
        self.radius = radius
        self.h = height / 2.0
        self.axis = axis

    def _sdf(self, x, y, z):
        p = torch.stack([x, y, z], dim=0)
        perp = (
            torch.norm(torch.stack([p[i] for i in range(3) if i != self.axis]), dim=0)
            - self.radius
        )
        along = torch.abs(p[self.axis]) - self.h
        outside = torch.clamp(torch.stack([perp, along], dim=0), min=0.0)
        inside = torch.clamp(
            torch.max(torch.stack([perp, along], dim=0), dim=0).values, max=0.0
        )
        return torch.norm(outside, dim=0) + inside

    def max_distance(self):
        return float(max(self.h, self.radius))


class ThinCylinder(Cylinder):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
    ):
        if radius is None:
            radius = random.uniform(
                min(grid_size[0], grid_size[1]) * 0.025,
                min(grid_size[0], grid_size[1]) * 0.05,
            )
        if height is None:
            height = random.uniform(grid_size[2] * 0.3, grid_size[2] * 0.5)
        super().__init__(grid_size, device, center, transform, radius, height)


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

    def max_distance(self):
        return self.R + self.r


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

    def max_distance(self):
        return float(max(self.radius, self.height))


class ThinCone(Cone):
    def __init__(
        self,
        grid_size: List[int],
        device: torch.device,
        center=None,
        transform=False,
        radius=None,
        height=None,
    ):
        if radius is None:
            radius = random.uniform(
                min(grid_size[0], grid_size[1]) * 0.025,
                min(grid_size[0], grid_size[1]) * 0.05,
            )
        if height is None:
            height = random.uniform(grid_size[2] * 0.3, grid_size[2] * 0.5)
        super().__init__(grid_size, device, center, transform, radius, height)


class HexagonalPrism(SDFObject):
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
            height = random.uniform(W * 0.1, W * 0.3)
        self.radius = radius
        self.height = height

    def _sdf(self, x, y, z):
        k = torch.tensor(
            [[-0.8660254], [0.5], [0.57735]], device=self.device
        )  # cos(30°), sin(30°), 1/sqrt(3)
        k = k.view(3, 1, 1, 1).expand(3, *self.grid_size)
        p = torch.stack([x, y], dim=0)
        p = torch.abs(p)
        p -= (
            2.0
            * torch.min(
                torch.stack(
                    [torch.sum(k[:2] * p, dim=0), torch.zeros_like(k[0])], dim=0
                ),
                dim=0,
            ).values
            * k[:2]
        )
        p -= torch.stack(
            [
                torch.clamp(p[0], min=-k[2] * self.radius, max=k[2] * self.radius),
                torch.ones_like(p[1]) * self.radius,
            ],
            dim=0,
        )
        perp = torch.norm(p, dim=0) * torch.sign(p[1])
        along = torch.abs(z) - self.height / 2.0
        outside = torch.clamp(torch.stack([perp, along], dim=0), min=0.0)
        inside = torch.clamp(
            torch.max(torch.stack([perp, along], dim=0), dim=0).values, max=0.0
        )
        return torch.norm(outside, dim=0) + inside

    def max_distance(self):
        return float(max(self.radius, self.height))
