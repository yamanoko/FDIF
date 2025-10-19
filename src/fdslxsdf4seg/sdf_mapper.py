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
            sdfs: shape (n_objs, D, H, W) のSDF値テンソル

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
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "inverse_cube"


class LinearMapper(SDFMapper):
    """線形マッピング: 64.0 - |x|"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """線形関数でマッピングして合計を返す"""
        mapped = 64.0 - torch.abs(sdfs)
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "linear"


class GaussianMapper(SDFMapper):
    """ガウシアンマッピング: 128.0 * exp(-x^2/2)"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """ガウシアン関数でマッピングして合計を返す"""
        mapped = 128.0 * torch.exp(-(sdfs**2) / 2.0)
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "gaussian"


class ReciprocalMapper(SDFMapper):
    """逆数マッピング: 128.0 / (|x| + 1)"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """逆数関数でマッピングして合計を返す"""
        mapped = 128.0 / (torch.abs(sdfs) + 1.0)
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "reciprocal"


class SquareMapper(SDFMapper):
    """二乗マッピング: 128.0 / (x^2 + 1)"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """二乗関数でマッピングして合計を返す"""
        mapped = 128.0 / (torch.pow(sdfs, 2.0) + 1.0)
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "square"


class TanhMapper(SDFMapper):
    """双曲正接マッピング: 64.0 * (1 + tanh(x))"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """双曲正接関数でマッピングして合計を返す"""
        mapped = 64.0 * (1.0 + torch.tanh(sdfs))
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "tanh"


class SoftmaxMapper(SDFMapper):
    """ソフトマックス風マッピング: exp(-|x|) のみ"""

    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        """指数関数でマッピングして合計を返す"""
        mapped = 128.0 * torch.exp(-torch.abs(sdfs))
        return mapped.sum(dim=0)

    def get_name(self) -> str:
        return "softmax"


class MapperRegistry:
    """登録されたSDFマッパーを管理するレジストリ"""

    _mappers: Dict[str, SDFMapper] = {
        "inverse_cube": InverseCubeMapper(),
        "linear": LinearMapper(),
        "gaussian": GaussianMapper(),
        "reciprocal": ReciprocalMapper(),
        "square": SquareMapper(),
        "tanh": TanhMapper(),
        "softmax": SoftmaxMapper(),
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
