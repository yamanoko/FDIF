# SDF Mapper実装ガイド

## 概要

SDFセグメンテーションデータセットで、複数のSDF変換関数（マッパー）とプリミティブの組み合わせを明示的に管理するように設計を変更しました。

これにより、「Cylinder + inverse_cube」と「Cylinder + linear」を異なるラベルクラスとして識別できるようになりました。

## 新しいファイル

### 1. `sdf_mapper.py`
**概要**: SDF値のマッピング処理を定義する抽象基底クラスと具体的な実装

#### 主要なクラス

- **`SDFMapper`** (抽象基底クラス)
  - `apply(sdfs: torch.Tensor) -> torch.Tensor`: SDF値をマッピング
  - `get_name() -> str`: マッパーの識別名を取得

#### 実装済みのマッパー

1. **`InverseCubeMapper`**: `128.0 / (|x|^3 + 1)`
2. **`LinearMapper`**: `64.0 - |x|`
3. **`GaussianMapper`**: `128.0 * exp(-x^2/2)`
4. **`ReciprocalMapper`**: `128.0 / (|x| + 1)`
5. **`SquareMapper`**: `128.0 / (x^2 + 1)`
6. **`TanhMapper`**: `64.0 * (1 + tanh(x))`
7. **`SoftmaxMapper`**: `128.0 * exp(-|x|)`

#### `MapperRegistry`

- `register(name: str, mapper: SDFMapper)`: 新しいマッパーを登録
- `get(name: str) -> SDFMapper`: マッパーを取得
- `get_all_names() -> List[str]`: 利用可能なマッパー名のリストを取得

#### 使用例

```python
from fdslxsdf4seg.sdf_mapper import MapperRegistry

# マッパーを取得
mapper = MapperRegistry.get("inverse_cube")
stacked_sdfs = torch.stack(sdfs, dim=0)  # shape: (n_objs, D, H, W)
x_vol = mapper.apply(stacked_sdfs)  # shape: (D, H, W)
```

### 2. `hybrid_primitive.py`
**概要**: プリミティブとマッパーの組み合わせを表現

#### 主要なクラス

- **`HybridPrimitive`**
  - `__init__(primitive_class: Type, mapper: SDFMapper)`
  - `__call__(*args, **kwargs)`: プリミティブをインスタンス化
  - `get_hybrid_name() -> str`: 一意の識別名を取得（例: `Cylinder_inverse_cube`）
  - `get_display_name() -> str`: 表示用の名前を取得（例: `Cylinder + inverse_cube`）

#### ヘルパー関数

- **`create_hybrid_primitives(primitive_classes, mapper_names) -> Dict[str, HybridPrimitive]`**
  - 全てのプリミティブ・マッパー組み合わせを生成
  - 例: `[Cylinder, Sphere]` × `[inverse_cube, linear]` = 4個のハイブリッドプリミティブ

- **`get_mapper_choices() -> List[str]`**
  - 利用可能なマッパー名のリストを返す

#### 使用例

```python
from fdslxsdf4seg.hybrid_primitive import create_hybrid_primitives
from fdslxsdf4seg.primitive_registry import Cylinder, Sphere

primitives = [Cylinder, Sphere]
mappers = ["inverse_cube", "linear"]
hybrids = create_hybrid_primitives(primitives, mappers)

# hybrids = {
#     "Cylinder_inverse_cube": HybridPrimitive(...),
#     "Cylinder_linear": HybridPrimitive(...),
#     "Sphere_inverse_cube": HybridPrimitive(...),
#     "Sphere_linear": HybridPrimitive(...),
# }

# インスタンス化
hybrid = hybrids["Cylinder_inverse_cube"]
obj = hybrid(grid_size=[64, 64, 64], device=device, transform=True)
sdf = obj.sdf(X, Y, Z)
```

## 修正したファイル

### `generate_sdf_dataset.py`

#### 主な変更点

1. **新しいパラメータの追加**
   - `SDFSegmentationDataset.__init__`に`sdf_mappers: Optional[List[str]] = None`を追加
   - `generate_and_save`関数に同じパラメータを追加

2. **`__init__`メソッドの修正**
   ```python
   # ハイブリッドプリミティブを生成（プリミティブ × マッパーの全組み合わせ）
   self.hybrid_primitives = create_hybrid_primitives(
       selected_primitives, sdf_mappers or ["inverse_cube"]
   )
   
   # primitive_classesにハイブリッドプリミティブを保存
   self.primitive_classes = {}
   class_id = 1
   for hybrid_key, hybrid in self.hybrid_primitives.items():
       self.primitive_classes[class_id] = hybrid
       class_id += 1
   ```

3. **`__getitem__`メソッドの修正**
   - CombinedUnionプリミティブとハイブリッドプリミティブの処理を分離
   - ハイブリッドプリミティブから自動的にマッパーを取得
   ```python
   if isinstance(primitive_or_name, str) and is_combined_union_primitive(...):
       # CombinedUnion処理
       obj = create_combined_union_instance(...)
   else:
       # ハイブリッドプリミティブ
       hybrid = primitive_or_name
       obj = hybrid(...)
   
   # マッパーの取得と適用
   current_mapper = first_hybrid.mapper
   x_vol = current_mapper.apply(stacked_sdfs)
   ```

4. **ログ出力の改善**
   - ハイブリッドプリミティブの表示名を出力
   - 使用されたマッパーの情報をログに記録

5. **CLIパラメータの追加**
   ```bash
   --sdf_mappers {inverse_cube,linear,gaussian,reciprocal,square,tanh,softmax}
   ```

## データセットのラベルマッピング

### クラスID割り当ての例

プリミティブ: `[Cylinder, Sphere]`
マッパー: `[inverse_cube, linear]`

| Class ID | プリミティブ | マッパー | 表示名 |
|----------|-----------|---------|-------|
| 1 | Cylinder | inverse_cube | Cylinder + inverse_cube |
| 2 | Cylinder | linear | Cylinder + linear |
| 3 | Sphere | inverse_cube | Sphere + inverse_cube |
| 4 | Sphere | linear | Sphere + linear |

## 使用方法

### 基本的な使用例

```bash
python generate_sdf_dataset.py \
    --out_dir outputs/hybrid_dataset \
    --num_samples 100 \
    --primitives Cylinder Sphere \
    --sdf_mappers inverse_cube linear gaussian \
    --num_val_samples 20
```

### 結果

- 各プリミティブとマッパーの組み合わせが異なるクラスIDを持つ
- `generation_log.txt`にクラスIDマッピングが記録される
- `y_vol`（セグメンテーションマスク）各ピクセルが対応するハイブリッドプリミティブのクラスIDを保持

## ジェネレーションログの例

```
Actually selected primitives (2): Cylinder, Sphere
SDF Mappers: inverse_cube, linear, gaussian
Primitive class ID mapping:
  1: Cylinder + inverse_cube
  2: Cylinder + linear
  3: Cylinder + gaussian
  4: Sphere + inverse_cube
  5: Sphere + linear
  6: Sphere + gaussian
```

## カスタムマッパーの追加

新しいマッパーを追加する場合：

1. `sdf_mapper.py`に新しいマッパークラスを作成
2. `MapperRegistry`に自動登録される（または手動で`register`メソッドを使用）

```python
class MyCustomMapper(SDFMapper):
    def apply(self, sdfs: torch.Tensor) -> torch.Tensor:
        # カスタムマッピング処理
        return custom_result
    
    def get_name(self) -> str:
        return "my_custom"

# 登録
from fdslxsdf4seg.sdf_mapper import MapperRegistry
mapper = MyCustomMapper()
MapperRegistry.register("my_custom", mapper)

# これでCLIから使用可能
# --sdf_mappers my_custom
```

## 設計のメリット

- ✅ プリミティブとマッパーの組み合わせを明示的に管理
- ✅ 各組み合わせが一意のクラスIDを持つ
- ✅ 新しいマッパーを簡単に追加可能
- ✅ ログで全クラスの対応関係を確認可能
- ✅ CombinedUnionプリミティブとの互換性を保持

## 既知の制限

- 現在、単一のマッパーのみで複数オブジェクトを処理（異なるマッパーの混在は未対応）
  - 複数マッパー混在の実装が必要な場合は、`__getitem__`メソッドを拡張してください
