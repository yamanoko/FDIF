"""
Displacement関数の定義とレジストリ

SDFオブジェクトに適用できる変位関数を提供します。
これらの関数はSDFの表面を変形させるために使用されます。
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class DisplacementFunction(ABC):
    """変位関数の抽象基底クラス"""

    @abstractmethod
    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """変位量を計算

        Args:
            x, y, z: 座標テンソル (shape=(D,H,W))

        Returns:
            変位量 (shape=(D,H,W))
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """この変位関数の識別用名前を取得"""
        pass


class NoDisplacement(DisplacementFunction):
    """変位なし（デフォルト）"""

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def get_name(self) -> str:
        return "none"


class PerlinNoiseDisplacement(DisplacementFunction):
    """疑似パーリンノイズによる変位（簡易実装）"""

    def __init__(self, amplitude: float = 0.05, scale: float = 10.0):
        """
        Args:
            amplitude: 振幅
            scale: スケール（ノイズの細かさ）
        """
        self.amplitude = min(max(amplitude, 0.0), 0.1)
        self.scale = scale

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # 簡易的なノイズ生成（実際のパーリンノイズではない）
        amplitude = self.amplitude * torch.max(x)
        x, y, z = (
            x.float() / torch.max(x),
            y.float() / torch.max(y),
            z.float() / torch.max(z),
        )
        noise = torch.sin(x * self.scale) * torch.sin(y * self.scale) + torch.sin(
            z * self.scale
        )
        return amplitude * noise

    def get_name(self) -> str:
        return f"perlin_amp_{self.amplitude}_scale_{self.scale}"


class TurbulenceDisplacement(DisplacementFunction):
    """乱流による変位（複数の周波数の組み合わせ）"""

    def __init__(self, amplitude: float = 0.03, scale: float = 10.0):
        """
        Args:
            amplitude: 振幅
            scale: スケール（ノイズの細かさ）
        """
        self.amplitude = min(max(amplitude, 0.0), 0.1)
        self.scale = scale

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # 複数の周波数を組み合わせた乱流効果
        amplitude = self.amplitude * torch.max(x)
        x, y, z = (
            x.float() / torch.max(x) * self.scale,
            y.float() / torch.max(y) * self.scale,
            z.float() / torch.max(z) * self.scale,
        )
        turb = (
            torch.sin(0.1 * x) * torch.cos(0.15 * y)
            + torch.sin(0.2 * y) * torch.cos(0.25 * z)
            + torch.sin(0.3 * z) * torch.cos(0.35 * x)
        )
        return amplitude * turb

    def get_name(self) -> str:
        return f"turbulence_amp_{self.amplitude}_scale_{self.scale}"


class RidgeDisplacement(DisplacementFunction):
    """リッジ（稜線）による変位"""

    def __init__(self, amplitude: float = 0.04, frequency: float = 5.0):
        """
        Args:
            amplitude: 振幅
            frequency: 周波数（リッジの密度）
        """
        self.amplitude = min(max(amplitude, 0.0), 0.1)
        self.frequency = frequency

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # リッジ効果の生成
        amplitude = self.amplitude * torch.max(x)
        x, y, z = (
            x.float() / torch.max(x),
            y.float() / torch.max(y),
            z.float() / torch.max(z),
        )
        ridge = (
            torch.abs(torch.sin(self.frequency * x))
            + torch.abs(torch.sin(self.frequency * y))
            + torch.abs(torch.sin(self.frequency * z))
        )
        return amplitude * ridge

    def get_name(self) -> str:
        return f"ridge_amp_{self.amplitude}_freq_{self.frequency}"


class SharpMaxDisplacement(DisplacementFunction):
    """max(sin)を用いたシャープな変位関数"""

    def __init__(self, amplitude: float = 0.05, frequency: float = 10.0):
        """
        Args:
            amplitude: 振幅
            frequency: 周波数（変位の密度）
        """
        self.amplitude = min(max(amplitude, 0.0), 0.1)
        self.frequency = frequency

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # シャープなmax(sin)効果の生成
        amplitude = self.amplitude * torch.max(x)
        x, y, z = (
            x.float() / torch.max(x),
            y.float() / torch.max(y),
            z.float() / torch.max(z),
        )
        sharp_max = torch.max(
            torch.stack(
                [
                    torch.abs(torch.sin(self.frequency * x)),
                    torch.abs(torch.sin(self.frequency * y)),
                    torch.abs(torch.sin(self.frequency * z)),
                ],
                dim=0,
            ),
            dim=0,
        ).values
        return amplitude * sharp_max

    def get_name(self) -> str:
        return f"sharpmax_amp_{self.amplitude}_freq_{self.frequency}"


class AxisAlignedTwistDisplacement(DisplacementFunction):
    """特定の軸に沿ったストライプパターンによる変位関数"""

    def __init__(
        self, amplitude: float = 0.03, frequency: float = 10.0, axis: str = "x"
    ):
        """
        Args:
            amplitude: 振幅
            frequency: 周波数（ストライプの密度）
            axis: ストライプを生成する軸 ('x', 'y', 'z' のいずれか)
        """
        self.amplitude = min(max(amplitude, 0.0), 0.1)
        self.frequency = frequency
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.axis = axis

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # ストライプパターンの生成
        amplitude = self.amplitude * torch.max(x)
        coord = {"x": x, "y": y, "z": z}[self.axis]
        coord = coord.float() / torch.max(coord)
        stripes = torch.abs(torch.sin(self.frequency * coord))
        return amplitude * stripes

    def get_name(self) -> str:
        return f"twisted_amp_{self.amplitude}_freq_{self.frequency}_axis_{self.axis}"


class SawtoothDisplacement(DisplacementFunction):
    """ノコギリ波による変位関数"""

    def __init__(self, amplitude: float = 0.03, frequency: float = 10.0):
        """
        Args:
            amplitude: 振幅
            frequency: 周波数（ノコギリ波の密度）
        """
        self.amplitude = min(max(amplitude, 0.0), 0.1)
        self.frequency = frequency

    def apply(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # ノコギリ波効果の生成
        amplitude = self.amplitude * torch.max(x)
        x, y, z = (
            x.float() / torch.max(x),
            y.float() / torch.max(y),
            z.float() / torch.max(z),
        )
        sawtooth = (
            (2 / torch.pi)
            * (self.frequency * x - torch.floor(0.5 + self.frequency * x))
            + (2 / torch.pi)
            * (self.frequency * y - torch.floor(0.5 + self.frequency * y))
            + (2 / torch.pi)
            * (self.frequency * z - torch.floor(0.5 + self.frequency * z))
        )
        return amplitude * sawtooth

    def get_name(self) -> str:
        return f"sawtooth_amp_{self.amplitude}_freq_{self.frequency}"


class DisplacementRegistry:
    """変位関数のレジストリ

    利用可能な変位関数を管理し、名前でアクセスできるようにします。
    """

    # 利用可能な変位関数を辞書で管理
    DISPLACEMENTS: Dict[str, DisplacementFunction] = {
        "none": NoDisplacement(),
        "perlin_more_fine": PerlinNoiseDisplacement(amplitude=0.03, scale=50.0),
        "perlin_fine": PerlinNoiseDisplacement(amplitude=0.01, scale=30.0),
        "turbulence": TurbulenceDisplacement(amplitude=0.05, scale=30.0),
        "ridge": RidgeDisplacement(amplitude=0.03, frequency=30.0),
        "ridge_coarse": RidgeDisplacement(amplitude=0.05, frequency=10.0),
        "sharpmax": SharpMaxDisplacement(amplitude=0.05, frequency=20.0),
        "sharpmax_fine": SharpMaxDisplacement(amplitude=0.03, frequency=40.0),
        "twisted_x": AxisAlignedTwistDisplacement(
            amplitude=0.04, frequency=20.0, axis="x"
        ),
        "sawtooth": SawtoothDisplacement(amplitude=0.03, frequency=15.0),
        "sawtooth_fine": SawtoothDisplacement(amplitude=0.02, frequency=30.0),
    }

    @classmethod
    def get(cls, name: str) -> DisplacementFunction:
        """名前から変位関数を取得

        Args:
            name: 変位関数の名前

        Returns:
            DisplacementFunctionインスタンス

        Raises:
            KeyError: 指定された名前の変位関数が存在しない場合
        """
        if name not in cls.DISPLACEMENTS:
            available = ", ".join(cls.DISPLACEMENTS.keys())
            raise KeyError(
                f"Unknown displacement function: {name}. Available: {available}"
            )
        return cls.DISPLACEMENTS[name]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """利用可能な変位関数名のリストを取得

        Returns:
            変位関数名のリスト
        """
        return list(cls.DISPLACEMENTS.keys())

    @classmethod
    def add(cls, name: str, displacement: DisplacementFunction):
        """新しい変位関数を追加

        Args:
            name: 変位関数の名前
            displacement: DisplacementFunctionインスタンス
        """
        cls.DISPLACEMENTS[name] = displacement
