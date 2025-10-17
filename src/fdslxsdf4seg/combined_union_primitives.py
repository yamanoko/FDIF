#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CombinedObjectUnionを使用した自動プリミティブ生成機能
"""

import random
from typing import Dict, List, Type

from fdslxsdf4seg.primitive_registry import ALL_PRIMITIVES, DEFAULT_PRIMITIVES
from fdslxsdf4seg.sdf_object import CombinedObjectUnion


class CombinedUnionPrimitive:
    """CombinedObjectUnionのプリミティブを表すクラス"""

    def __init__(
        self,
        name: str,
        first_class: Type,
        second_class: Type,
        first_params: dict = None,
        second_params: dict = None,
    ):
        self.name = name
        self.first_class = first_class
        self.second_class = second_class
        self.first_params = first_params or {}
        self.second_params = second_params or {}

    def create_instance(self, grid_size, device, center=None, transform=False):
        """CombinedObjectUnionのインスタンスを作成"""
        return CombinedObjectUnion(
            grid_size=grid_size,
            device=device,
            center=center,
            transform=transform,
            FirstClass=self.first_class,
            SecondClass=self.second_class,
            first_params=self.first_params,
            second_params=self.second_params,
        )


def generate_combined_union_primitives(
    num_combinations: int, available_primitives: List[str] = None, seed: int = None
) -> Dict[str, CombinedUnionPrimitive]:
    """
    指定された数のCombinedObjectUnionプリミティブを自動生成

    Args:
        num_combinations: 生成する組み合わせの数
        available_primitives: 使用可能なプリミティブリスト（Noneの場合はDEFAULT_PRIMITIVESを使用）
        seed: ランダムシード

    Returns:
        生成されたCombinedUnionPrimitiveの辞書（名前 -> CombinedUnionPrimitive）
    """
    if seed is not None:
        random.seed(seed)

    if available_primitives is None:
        available_primitives = DEFAULT_PRIMITIVES

    # 利用可能なプリミティブクラスを取得
    available_classes = []
    for primitive_name in available_primitives:
        if primitive_name in ALL_PRIMITIVES:
            available_classes.append((primitive_name, ALL_PRIMITIVES[primitive_name]))

    if len(available_classes) < 2:
        raise ValueError(
            f"At least 2 primitive classes are required, but only {len(available_classes)} available"
        )

    combined_primitives = {}
    used_combinations = set()

    for i in range(num_combinations):
        # ユニークな組み合わせを生成するまでリトライ
        max_retries = 100
        retry_count = 0

        while retry_count < max_retries:
            # ランダムに2つのプリミティブを選択
            first_name, first_class = random.choice(available_classes)
            second_name, second_class = random.choice(available_classes)

            # 組み合わせのキー（順序を考慮してソート）
            combination_key = tuple(sorted([first_name, second_name]))

            if combination_key not in used_combinations:
                used_combinations.add(combination_key)
                break

            retry_count += 1

        if retry_count >= max_retries:
            print(
                f"Warning: Could not generate unique combination for index {i}, using duplicate"
            )

        # プリミティブ名を生成
        union_name = f"Union_{first_name}_{second_name}_{i:03d}"

        # デフォルトパラメータを生成（必要に応じて調整）
        first_params = _generate_default_params(first_name, first_class)
        second_params = _generate_default_params(second_name, second_class)

        # CombinedUnionPrimitiveを作成
        combined_primitive = CombinedUnionPrimitive(
            name=union_name,
            first_class=first_class,
            second_class=second_class,
            first_params=first_params,
            second_params=second_params,
        )

        combined_primitives[union_name] = combined_primitive

    return combined_primitives


def _generate_default_params(primitive_name: str, primitive_class: Type) -> dict:
    """
    プリミティブクラスに応じたデフォルトパラメータを生成

    Args:
        primitive_name: プリミティブ名
        primitive_class: プリミティブクラス

    Returns:
        デフォルトパラメータの辞書
    """
    params = {}

    # 基本的なパラメータパターンを定義
    if "Sphere" in primitive_name:
        params = {"radius": random.uniform(8.0, 20.0)}

    elif "Cylinder" in primitive_name or "Tube" in primitive_name:
        params = {
            "radius": random.uniform(6.0, 15.0),
            "height": random.uniform(10.0, 25.0),
        }

    elif "Prism" in primitive_name:
        params = {"height": random.uniform(10.0, 25.0)}

        # StarPrismの場合
        if "Star" in primitive_name:
            params.update(
                {
                    "radius": random.uniform(8.0, 18.0),
                    "n": random.randint(5, 8),
                    "w": random.uniform(0.3, 0.6),
                }
            )

        # SectorPolygonPrismの場合
        elif "SectorPolygon" in primitive_name:
            r1 = random.uniform(8.0, 18.0)
            r2 = random.uniform(8.0, 18.0)
            params.update({"n": random.randint(6, 12), "r1": r1, "r2": r2})

    elif "Torus" in primitive_name:
        major_r = random.uniform(15.0, 25.0)
        minor_r = random.uniform(6.0, major_r * 0.6)
        params = {"major_r": major_r, "minor_r": minor_r}

        if "Star" in primitive_name:
            params.update({"n": random.randint(5, 8), "w": random.uniform(0.3, 0.6)})
        elif "SectorPolygon" in primitive_name:
            params.update({"n": random.randint(6, 10)})

    elif "Revolution" in primitive_name:
        params = {
            "radius": random.uniform(10.0, 20.0),
            "distance": random.uniform(2.0, 8.0),
        }

        if "Star" in primitive_name:
            params.update({"n": random.randint(5, 8), "w": random.uniform(0.3, 0.6)})

    return params


def is_combined_union_primitive(primitive_name: str) -> bool:
    """
    プリミティブ名がCombinedUnionプリミティブかどうかを判定

    Args:
        primitive_name: プリミティブ名

    Returns:
        CombinedUnionプリミティブの場合True
    """
    return primitive_name.startswith("Union_")


def create_combined_union_instance(
    primitive_name: str,
    combined_primitives: Dict[str, CombinedUnionPrimitive],
    grid_size: List[int],
    device,
    center=None,
    transform=False,
):
    """
    CombinedUnionプリミティブのインスタンスを作成

    Args:
        primitive_name: プリミティブ名
        combined_primitives: CombinedUnionプリミティブの辞書
        grid_size: グリッドサイズ
        device: デバイス
        center: 中心座標
        transform: 変換を適用するかどうか

    Returns:
        作成されたCombinedObjectUnionインスタンス
    """
    if primitive_name not in combined_primitives:
        raise ValueError(f"Combined union primitive '{primitive_name}' not found")

    combined_primitive = combined_primitives[primitive_name]
    return combined_primitive.create_instance(
        grid_size=grid_size, device=device, center=center, transform=transform
    )


def get_combined_union_primitive_names(
    combined_primitives: Dict[str, CombinedUnionPrimitive],
) -> List[str]:
    """
    CombinedUnionプリミティブの名前リストを取得

    Args:
        combined_primitives: CombinedUnionプリミティブの辞書

    Returns:
        プリミティブ名のリスト
    """
    return list(combined_primitives.keys())


if __name__ == "__main__":
    # テスト用のコード
    print("Testing CombinedUnionPrimitive generation...")

    # 5つの組み合わせを生成
    combined_primitives = generate_combined_union_primitives(
        num_combinations=5, seed=42
    )

    print(f"Generated {len(combined_primitives)} combined primitives:")
    for name, primitive in combined_primitives.items():
        print(
            f"  {name}: {primitive.first_class.__name__} + {primitive.second_class.__name__}"
        )
        print(f"    First params: {primitive.first_params}")
        print(f"    Second params: {primitive.second_params}")
