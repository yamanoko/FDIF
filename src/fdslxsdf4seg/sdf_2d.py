import math
import random
from typing import Optional

import torch


class StarBase:
    def __init__(self, radius=None, n=None, w=None, device=None, dtype=torch.float32):
        self.r = 1.0 if radius is None else float(radius)
        self.device = device
        self.dtype = dtype
        self.n = 5 if n is None else max(3, int(n))
        self.w = random.uniform(0.1, 0.7) if w is None else float(w)

    def sdf2d_base(self, x, y):
        m = self.n + self.w * (2.0 - self.n)
        an = 3.1415927 / self.n
        en = 3.1415927 / m
        racs = self.r * torch.tensor([[torch.cos(an)], [torch.sin(an)]])
        ecs = torch.tensor([[torch.cos(en)], [torch.sin(en)]])
        x = torch.abs(x)
        p = torch.stack([x, y], dim=0)  # (2,N)
        bn = torch.remainder(torch.atan2(y, x), 2.0 * an) - an
        p = p.norm(dim=0) * torch.stack(
            [torch.cos(bn), torch.abs(torch.sin(bn))], dim=0
        )  # (2,N)
        p = p - racs
        p += ecs * torch.clamp(-ecs.t() @ p, min=0.0, max=racs[1] / ecs[1])
        return torch.sign(p[0]) * p.norm(dim=0)  # (N,)


class _SectorPolygonBase:  # SDFObject にミックスインして使う内部基底
    EPS = 1e-12
    TAU = 2.0 * math.pi

    def __init__(
        self,
        n: int,
        r1: float,
        r2: float,
        seed: Optional[int] = None,
        device=None,
        dtype=torch.float32,
    ):
        self.n = max(3, int(n))
        self.rmin = float(min(r1, r2))
        self.rmax = float(max(r1, r2))
        self.seed = random.randrange(1 << 30) if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self.device = device
        self.dtype = dtype
        self.verts = self._build_vertices().to(device=device, dtype=dtype)  # (n,2)

    def _build_vertices(self) -> torch.Tensor:
        vs = []
        for i in range(self.n):
            a0 = self.TAU * (i / self.n)
            a1 = self.TAU * ((i + 1) / self.n)
            ang = self._rng.uniform(a0, a1)  # セクタ内の角度
            rad = self._rng.uniform(self.rmin, self.rmax)  # 半径レンジ
            vs.append([rad * math.cos(ang), rad * math.sin(ang)])
        return torch.tensor(vs, dtype=torch.float32)

    @staticmethod
    def _segment_dist2(Px, Py, ax, ay, bx, by, eps=1e-12):
        ex = bx - ax
        ey = by - ay
        wpx = Px - ax
        wpy = Py - ay
        ee = ex * ex + ey * ey
        t = ((wpx * ex + wpy * ey) / (ee + eps)).clamp(0.0, 1.0)
        nx = wpx - t * ex
        ny = wpy - t * ey
        return nx * nx + ny * ny

    @staticmethod
    def _winding_number(Px, Py, ax, ay, bx, by):
        up = (ay <= Py) & (by > Py)
        down = (ay > Py) & (by <= Py)
        ex = bx - ax
        ey = by - ay
        crossv = ex * (Py - ay) - ey * (Px - ax)  # cross(e, p-a)
        wn = torch.zeros_like(Px, dtype=torch.int32)
        wn = wn + (up & (crossv > 0)).to(torch.int32)
        wn = wn - (down & (crossv < 0)).to(torch.int32)
        return wn

    def sdf2d_base(self, X, Y):
        """
        2D 多角形 SDF（内:負, 外:正）。X,Y: (N,)
        """
        device, dtype = X.device, X.dtype
        d2 = torch.full_like(X, 1e30, dtype=dtype, device=device)
        wn_total = torch.zeros_like(X, dtype=torch.int32, device=device)

        V = self.verts.to(device=device, dtype=dtype)
        n = V.shape[0]
        for i in range(n):
            ax, ay = V[i, 0], V[i, 1]
            bx, by = V[(i + 1) % n, 0], V[(i + 1) % n, 1]
            d2_i = self._segment_dist2(X, Y, ax, ay, bx, by, eps=self.EPS)
            d2 = torch.minimum(d2, d2_i)
            wn_total = wn_total + self._winding_number(X, Y, ax, ay, bx, by)

        dist = torch.sqrt(d2 + self.EPS)
        sign_inside = torch.where(
            wn_total == 0, torch.ones_like(dist), -torch.ones_like(dist)
        )
        return sign_inside * dist
