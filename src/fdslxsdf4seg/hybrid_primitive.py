"""
プリミティブとSDFマッパーの組み合わせを表現するハイブリッドプリミティブクラス
"""

from typing import Dict, List, Type

from fdslxsdf4seg.sdf_mapper import MapperRegistry, SDFMapper


class HybridPrimitive:
    """プリミティブクラスとSDFマッパーの組み合わせを表現

    単一の「プリミティブ + マッパー」の組み合わせを識別可能なクラスとして扱います。
    このクラスは、プリミティブをインスタンス化する際に使用されます。
    """

    def __init__(self, primitive_class: Type, mapper: SDFMapper):
        """ハイブリッドプリミティブを初期化

        Args:
            primitive_class: プリミティブのクラス（Cylinder, Sphere等）
            mapper: SDFMapperのインスタンス
        """
        self.primitive_class = primitive_class
        self.mapper = mapper
        self._name = None
        self._display_name = None

    def __call__(self, *args, **kwargs):
        """プリミティブをインスタンス化（クラスのコンストラクタとして機能）

        Returns:
            プリミティブのインスタンス
        """
        return self.primitive_class(*args, **kwargs)

    def get_hybrid_name(self) -> str:
        """一意の識別用複合名を生成

        Returns:
            "{プリミティブ名}_{マッパー名}" の形式の文字列
        """
        if self._name is None:
            primitive_name = self.primitive_class.__name__
            mapper_name = self.mapper.get_name()
            self._name = f"{primitive_name}_{mapper_name}"
        return self._name

    def get_display_name(self) -> str:
        """人間が読みやすい表示用の名前

        Returns:
            "{プリミティブ名} + {マッパー名}" の形式の文字列
        """
        if self._display_name is None:
            primitive_name = self.primitive_class.__name__
            mapper_name = self.mapper.get_name()
            self._display_name = f"{primitive_name} + {mapper_name}"
        return self._display_name

    def get_shape_name(self) -> str:
        """プリミティブのベース形状名を取得（マルチタスク用）

        Returns:
            プリミティブクラス名
        """
        return self.primitive_class.__name__

    def get_displacement_name(self) -> str:
        """displacement関数名を取得（マルチタスク用）

        Returns:
            HybridPrimitiveはdisplacementなしなので"none"を返す
        """
        return "none"

    def get_mapper_name(self) -> str:
        """マッパー名を取得（マルチタスク用）

        Returns:
            マッパー名
        """
        return self.mapper.get_name()

    def __repr__(self) -> str:
        return f"HybridPrimitive({self.get_display_name()})"


def create_hybrid_primitives(
    primitive_classes: List[Type],
    mapper_names: List[str],
) -> Dict[str, HybridPrimitive]:
    """全てのプリミティブ・マッパー組み合わせを生成

    Args:
        primitive_classes: プリミティブクラスのリスト
        mapper_names: マッパー名のリスト

    Returns:
        {hybrid_name: HybridPrimitive} の辞書

    Example:
        >>> primitives = [Cylinder, Sphere]
        >>> mappers = ["inverse_cube", "linear"]
        >>> hybrids = create_hybrid_primitives(primitives, mappers)
        >>> len(hybrids)  # 2 * 2 = 4
        4
    """
    hybrids = {}

    for prim_class in primitive_classes:
        for mapper_name in mapper_names:
            mapper = MapperRegistry.get(mapper_name)
            hybrid = HybridPrimitive(prim_class, mapper)
            hybrid_key = hybrid.get_hybrid_name()
            hybrids[hybrid_key] = hybrid

    return hybrids


def get_mapper_choices() -> List[str]:
    """利用可能なマッパー名のリストを取得

    Returns:
        マッパー名のリスト
    """
    return MapperRegistry.get_all_names()
