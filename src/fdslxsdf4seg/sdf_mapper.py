"""
SDF値をマッピングするための抽象基底クラスと具体的な実装
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch


class SDFMapper(ABC):
    """SDF値のマッピング処理の抽象基底クラス

    SDFスタックをマッピングして単一のボリュームに集約する処理を定義します。
    """

    @abstractmethod
    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """SDF値をマッピングして単一ボリュームに集約

        Args:
            sdfs: shape (D, H, W) のSDF値テンソル

        Returns:
            shape (D, H, W) のマッピング後の値
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """このマッパーの識別用名前を取得"""
        pass


class InverseCubeMapper(SDFMapper):
    """逆立方体マッピング: 128.0 / (|x|^3 + 1)"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """逆立方体関数でマッピングして合計を返す"""
        mapped = 128.0 / (torch.pow(torch.abs(sdfs), 3.0) + 1.0)
        return mapped

    def get_name(self) -> str:
        return "inverse_cube"


class ExponentialMapper(SDFMapper):
    """指数関数マッピングの基底クラス"""

    def __init__(self, base: float = 2.0):
        self.base = base

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """指数関数でマッピングして合計を返す"""
        mask = sdfs > 0.0
        mapped = torch.pow(self.base, torch.clamp(sdfs, max=0.0))
        mapped = mapped * 128.0
        return torch.where(mask, torch.zeros_like(mapped), mapped)

    def get_name(self) -> str:
        return f"exponential_base_{self.base}"


class LinearMapper(SDFMapper):
    """線形マッピング: 128.0 + slope * x"""

    def __init__(self, slope: float = 1.0):
        self.slope = slope

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """線形関数でマッピングして合計を返す"""
        mask = sdfs > 0.0
        mapped = torch.clamp(128.0 + self.slope * sdfs, min=0.0, max=128.0)
        return torch.where(mask, torch.zeros_like(mapped), mapped)

    def get_name(self) -> str:
        return f"linear_slope_{self.slope}"


class FloorMapper(SDFMapper):
    def __init__(self, width: float = 10.0, decrement: float = 10.0):
        self.width = width
        self.decrement = decrement

    def apply(self, sdfs):
        mask = sdfs > 0.0
        mapped = torch.ceil(sdfs / self.width) * self.decrement + 128.0
        mapped = torch.clamp(mapped, min=0.0, max=128.0)
        return torch.where(mask, torch.zeros_like(mapped), mapped)

    def get_name(self) -> str:
        return f"floor_width_{self.width}"


class ModularMapper(SDFMapper):
    """モジュラー関数マッピング: SDF値を特定の範囲で折り返す"""

    def __init__(self, width: float = 5.0, modulus: int = 5):
        self.width = width
        self.modulus = modulus

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        mask = sdfs > 0.0
        mapped = torch.remainder(
            -torch.floor(-sdfs / self.width) + self.modulus - 1.0, self.modulus
        ) * (128.0 / (self.modulus - 1.0))
        return torch.where(mask, torch.zeros_like(mapped), mapped)

    def get_name(self) -> str:
        return f"modular_{self.modulus}"


class SinusoidalMapper(SDFMapper):
    """正弦関数マッピング: SDF値に基づいて正弦波でマッピング"""

    def __init__(self, wavelength: float = 20.0):
        self.wavelength = wavelength

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        mask = sdfs > 0.0
        mapped = (torch.cos((sdfs / self.wavelength) * 2.0 * torch.pi) + 1.0) * 64.0
        return torch.where(mask, torch.zeros_like(mapped), mapped)

    def get_name(self) -> str:
        return f"sinusoidal_wavelength_{self.wavelength}"


class MapperRegistry:
    """登録されたSDFマッパーを管理するレジストリ"""

    _mappers: Dict[str, SDFMapper] = {
        "inverse_cube": InverseCubeMapper(),
        "exponential_base_2.0": ExponentialMapper(base=2.0),
        "exponential_base_1.5": ExponentialMapper(base=1.5),
        "linear_slope_20.0": LinearMapper(slope=20.0),
        "linear_slope_10.0": LinearMapper(slope=10.0),
        "floor_width_1.0": FloorMapper(width=1.0),
        "floor_width_0.5": FloorMapper(width=0.5),
        "modular_5": ModularMapper(width=1.0, modulus=5),
        "modular_10": ModularMapper(width=0.5, modulus=10),
        "sinusoidal_wavelength_1.0": SinusoidalMapper(wavelength=1.0),
        "sinusoidal_wavelength_3.0": SinusoidalMapper(wavelength=3.0),
    }

    @classmethod
    def register(cls, name: str, mapper: SDFMapper) -> None:
        """新しいマッパーを登録

        Args:
            name: マッパーの識別用名前
            mapper: SDFMapperのインスタンス

        Raises:
            ValueError: 既に同じ名前のマッパーが登録されている場合
        """
        if name in cls._mappers:
            raise ValueError(f"Mapper '{name}' is already registered")
        cls._mappers[name] = mapper

    @classmethod
    def get(cls, name: str) -> SDFMapper:
        """マッパーを名前で取得

        Args:
            name: マッパーの識別用名前

        Returns:
            SDFMapperのインスタンス

        Raises:
            ValueError: 指定された名前のマッパーが見つからない場合
        """
        if name not in cls._mappers:
            raise ValueError(
                f"Mapper '{name}' not registered. Available mappers: {', '.join(cls._mappers.keys())}"
            )
        return cls._mappers[name]

    @classmethod
    def get_all_names(cls) -> List[str]:
        """全てのマッパー名を取得"""
        return sorted(list(cls._mappers.keys()))

    @classmethod
    def get_all_mappers(cls) -> Dict[str, SDFMapper]:
        """全てのマッパーを取得"""
        return cls._mappers.copy()
