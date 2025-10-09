# プリミティブ バリエーション可視化機能

この機能は、同じプリミティブが生成時にどれほどのバリエーションを持っているかを確認するために、同じプリミティブの可視化を複数回行い、その結果の画像を並べて一枚の画像にして出力します。

## 使用方法

### 基本的な使用方法

```bash
# Sphereプリミティブの6つのバリエーションを生成
python visualize_primitives.py --variations

# 特定のプリミティブのバリエーションを生成
python visualize_primitives.py --variations --variation_primitive FiveStarPrism

# バリエーション数を指定
python visualize_primitives.py --variations --variation_primitive Cylinder --num_variations 9

# 出力ディレクトリを指定
python visualize_primitives.py --variations --output_dir my_output --variation_primitive Torus
```

### オプション

- `--variations`: バリエーション分析モードを有効にする
- `--variation_primitive`: 分析するプリミティブ名（デフォルト: "Sphere"）
- `--num_variations`: 生成するバリエーション数（デフォルト: 6）
- `--output_dir`: 出力ディレクトリ（デフォルト: "visualize_output"）
- `--3d`: 3D可視化も有効にする（バリエーション分析では通常無効）

## 対応プリミティブ

### 基本プリミティブ
- `Sphere` - 球体
- `Torus` - トーラス
- `Cone` - コーン
- `Octahedron` - 八面体
- `Cylinder` - シリンダー
- `ConvexCylinder` - 凸シリンダー
- `ConcaveCylinder` - 凹シリンダー
- `ConeCylinder` - コーンシリンダー

### セクターポリゴンプリズム
- `TrianglePrism` - 三角プリズム
- `SquarePrism` - 四角プリズム
- `PentagonPrism` - 五角プリズム
- `HexagonPrism` - 六角プリズム
- `HeptagonPrism` - 七角プリズム
- `OctagonPrism` - 八角プリズム

### コーンプリズム
- `TriangleConePrism` - 三角コーンプリズム
- `SquareConePrism` - 四角コーンプリズム
- `PentagonConePrism` - 五角コーンプリズム
- `HexagonConePrism` - 六角コーンプリズム

### 凸/凹プリズム
- `TriangleConvexPrism` - 三角凸プリズム
- `SquareConvexPrism` - 四角凸プリズム
- `TriangleConcavePrism` - 三角凹プリズム
- `SquareConcavePrism` - 四角凹プリズム

### スタープリズム
- `FiveStarPrism` - 五角星プリズム
- `SixStarPrism` - 六角星プリズム

### トーラス系
- `SquareTorus` - 四角トーラス
- `PentagonTorus` - 五角トーラス
- `HexagonTorus` - 六角トーラス
- `FiveStarTorus` - 五角星トーラス
- `SixStarTorus` - 六角星トーラス

### Revolution系（回転体）
- `ThreeStarRevolution` - 三角星回転体
- `FourStarRevolution` - 四角星回転体
- `FiveStarRevolution` - 五角星回転体

### Onioned系（オニオン形状）
- `OnionedCylinder` - オニオンシリンダー
- `OnionedTrianglePrism` - オニオン三角プリズム
- `OnionedSquarePrism` - オニオン四角プリズム
- `OnionedFiveStarPrism` - オニオン五角星プリズム

### Union系（組み合わせ形状）
- `SphereCylinderUnion` - 球体シリンダー組み合わせ
- `SphereTriangleUnion` - 球体三角組み合わせ
- `FiveStarRevolutionCylinderUnion` - 五角星回転体シリンダー組み合わせ
- `FiveStarRevolutionPentagonUnion` - 五角星回転体五角組み合わせ

## 出力

### ファイル構造
```
visualize_output/
└── variations/
    ├── sphere_var_01.png              # 個別のバリエーション画像
    ├── sphere_var_02.png
    ├── ...
    ├── sphere_var_06.png
    ├── sphere_var_01_slice.png        # スライス表示
    ├── sphere_var_02_slice.png
    ├── ...
    └── sphere_variations_combined.png # 結合された比較画像
```

### 結合画像の特徴
- すべてのバリエーションが一枚の画像に配置される
- 各バリエーションには番号が振られる
- プリミティブ名とサンプル数がタイトルに表示される
- グリッド形式で整理された見やすいレイアウト

## 使用例

### 基本的なバリエーション分析
```bash
# Sphereの基本的なバリエーション確認
python visualize_primitives.py --variations

# より多くのバリエーションで詳細分析
python visualize_primitives.py --variations --num_variations 12 --variation_primitive FiveStarPrism
```

### 複雑なプリミティブの分析
```bash
# Union系プリミティブのバリエーション
python visualize_primitives.py --variations --variation_primitive FiveStarRevolutionCylinderUnion --num_variations 9

# Onioned系プリミティブの分析
python visualize_primitives.py --variations --variation_primitive OnionedFiveStarPrism --num_variations 6
```

## 必要な依存関係

バリエーション画像の結合機能を使用するには、以下のライブラリが必要です：

```bash
pip install Pillow
```

Pillowがインストールされていない場合、個別の画像は生成されますが、結合画像は作成されません。

## 統計情報

実行時には、各バリエーションについて以下の統計情報が表示されます：
- `inside`: プリミティブ内部のボクセル数
- `outside`: プリミティブ外部のボクセル数

これにより、各バリエーションの大きさや形状の違いを数値的に確認できます。

## 注意事項

1. **ランダム性**: プリミティブによってはランダムパラメータが含まれているため、実行するたびに異なるバリエーションが生成される場合があります。

2. **メモリ使用量**: 多数のバリエーションを生成する場合は、メモリ使用量にご注意ください。

3. **実行時間**: プリミティブの複雑さとバリエーション数に応じて、実行時間が長くなる場合があります。

4. **3D可視化**: バリエーション分析では通常3D可視化は無効ですが、`--3d`オプションで有効にできます。ただし、多数のバリエーションで3D可視化を有効にすると、処理時間が大幅に増加します。