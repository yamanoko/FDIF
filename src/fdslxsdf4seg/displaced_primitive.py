"""
Displacement関数とプリミティブの組み合わせを表現するクラス

DisplacedPrimitiveは、SDFオブジェクトクラスとdisplacement関数の組み合わせを表現し、
さらにマッパーとの組み合わせによってハイブリッドプリミティブを生成できます。
"""

from typing import Dict, List, Type

from fdslxsdf4seg.displacement_functions import DisplacementFunction
from fdslxsdf4seg.sdf_mapper import MapperRegistry, SDFMapper


class DisplacedPrimitive:
    """プリミティブクラスとdisplacement関数の組み合わせを表現

    単一の「プリミティブ + displacement関数」の組み合わせを識別可能なクラスとして扱います。
    このクラスは、プリミティブをインスタンス化する際に使用されます。
    """

    def __init__(
        self, primitive_class: Type, displacement_function: DisplacementFunction
    ):
        """DisplacedPrimitiveを初期化

        Args:
            primitive_class: プリミティブのクラス（Cylinder, Sphere等）
            displacement_function: DisplacementFunctionのインスタンス
        """
        self.primitive_class = primitive_class
        self.displacement_function = displacement_function
        self._name = None
        self._display_name = None

    def __call__(self, *args, **kwargs):
        """プリミティブをインスタンス化し、displacement関数を設定

        Returns:
            displacement関数が設定されたプリミティブのインスタンス
        """
        instance = self.primitive_class(*args, **kwargs)
        # displacement関数を設定
        instance.set_displacement_function(self.displacement_function.apply)
        return instance

    def get_displaced_name(self) -> str:
        """一意の識別用複合名を生成

        Returns:
            "{プリミティブ名}_disp_{displacement名}" の形式の文字列
        """
        if self._name is None:
            primitive_name = self.primitive_class.__name__
            displacement_name = self.displacement_function.get_name()
            self._name = f"{primitive_name}_disp_{displacement_name}"
        return self._name

    def get_display_name(self) -> str:
        """人間が読みやすい表示用の名前

        Returns:
            "{プリミティブ名} + displacement({displacement名})" の形式の文字列
        """
        if self._display_name is None:
            primitive_name = self.primitive_class.__name__
            displacement_name = self.displacement_function.get_name()
            self._display_name = f"{primitive_name} + displacement({displacement_name})"
        return self._display_name

    def __repr__(self) -> str:
        return f"DisplacedPrimitive({self.get_display_name()})"


class HybridDisplacedPrimitive:
    """DisplacedPrimitiveとSDFマッパーの組み合わせを表現

    「プリミティブ + displacement関数 + マッパー」の3つの組み合わせを表現します。
    """

    def __init__(self, displaced_primitive: DisplacedPrimitive, mapper: SDFMapper):
        """HybridDisplacedPrimitiveを初期化

        Args:
            displaced_primitive: DisplacedPrimitiveインスタンス
            mapper: SDFMapperのインスタンス
        """
        self.displaced_primitive = displaced_primitive
        self.mapper = mapper
        self._name = None
        self._display_name = None

    def __call__(self, *args, **kwargs):
        """プリミティブをインスタンス化（displacement関数が既に設定されている）

        Returns:
            displacement関数が設定されたプリミティブのインスタンス
        """
        return self.displaced_primitive(*args, **kwargs)

    def get_hybrid_name(self) -> str:
        """一意の識別用複合名を生成

        Returns:
            "{プリミティブ名}_disp_{displacement名}_{マッパー名}" の形式の文字列
        """
        if self._name is None:
            displaced_name = self.displaced_primitive.get_displaced_name()
            mapper_name = self.mapper.get_name()
            self._name = f"{displaced_name}_{mapper_name}"
        return self._name

    def get_display_name(self) -> str:
        """人間が読みやすい表示用の名前

        Returns:
            "{プリミティブ名} + displacement({displacement名}) + {マッパー名}" の形式の文字列
        """
        if self._display_name is None:
            primitive_name = self.displaced_primitive.primitive_class.__name__
            displacement_name = (
                self.displaced_primitive.displacement_function.get_name()
            )
            mapper_name = self.mapper.get_name()
            self._display_name = (
                f"{primitive_name} + displacement({displacement_name}) + {mapper_name}"
            )
        return self._display_name

    def get_shape_name(self) -> str:
        """プリミティブのベース形状名を取得（マルチタスク用）

        Returns:
            プリミティブクラス名
        """
        return self.displaced_primitive.primitive_class.__name__

    def get_displacement_name(self) -> str:
        """displacement関数名を取得（マルチタスク用）

        Returns:
            displacement関数名
        """
        return self.displaced_primitive.displacement_function.get_name()

    def get_mapper_name(self) -> str:
        """マッパー名を取得（マルチタスク用）

        Returns:
            マッパー名
        """
        return self.mapper.get_name()

    def __repr__(self) -> str:
        return f"HybridDisplacedPrimitive({self.get_display_name()})"


def create_displaced_primitives(
    primitive_classes: List[Type],
    displacement_names: List[str],
) -> Dict[str, DisplacedPrimitive]:
    """全てのプリミティブ・displacement関数の組み合わせを生成

    Args:
        primitive_classes: プリミティブクラスのリスト
        displacement_names: displacement関数名のリスト

    Returns:
        {displaced_name: DisplacedPrimitive} の辞書

    Example:
        >>> primitives = [Cylinder, Sphere]
        >>> displacements = ["sine", "perlin"]
        >>> displaced = create_displaced_primitives(primitives, displacements)
        >>> len(displaced)  # 2 * 2 = 4
        4
    """
    from fdslxsdf4seg.displacement_functions import DisplacementRegistry

    displaced_primitives = {}

    for prim_class in primitive_classes:
        for disp_name in displacement_names:
            displacement_func = DisplacementRegistry.get(disp_name)
            displaced = DisplacedPrimitive(prim_class, displacement_func)
            displaced_key = displaced.get_displaced_name()
            displaced_primitives[displaced_key] = displaced

    return displaced_primitives


def create_hybrid_displaced_primitives(
    primitive_classes: List[Type],
    displacement_names: List[str],
    mapper_names: List[str],
) -> Dict[str, HybridDisplacedPrimitive]:
    """プリミティブ・displacement関数・マッパーの全組み合わせを生成

    Args:
        primitive_classes: プリミティブクラスのリスト
        displacement_names: displacement関数名のリスト
        mapper_names: マッパー名のリスト

    Returns:
        {hybrid_name: HybridDisplacedPrimitive} の辞書

    Example:
        >>> primitives = [Cylinder, Sphere]
        >>> displacements = ["sine", "perlin"]
        >>> mappers = ["inverse_cube", "linear"]
        >>> hybrids = create_hybrid_displaced_primitives(primitives, displacements, mappers)
        >>> len(hybrids)  # 2 * 2 * 2 = 8
        8
    """
    from fdslxsdf4seg.displacement_functions import DisplacementRegistry

    hybrid_displaced_primitives = {}

    for prim_class in primitive_classes:
        for disp_name in displacement_names:
            displacement_func = DisplacementRegistry.get(disp_name)
            displaced = DisplacedPrimitive(prim_class, displacement_func)

            for mapper_name in mapper_names:
                mapper = MapperRegistry.get(mapper_name)
                hybrid_displaced = HybridDisplacedPrimitive(displaced, mapper)
                hybrid_key = hybrid_displaced.get_hybrid_name()
                hybrid_displaced_primitives[hybrid_key] = hybrid_displaced

    return hybrid_displaced_primitives


def get_displacement_choices() -> List[str]:
    """利用可能なdisplacement関数名のリストを取得

    Returns:
        displacement関数名のリスト
    """
    from fdslxsdf4seg.displacement_functions import DisplacementRegistry

    return DisplacementRegistry.get_all_names()
