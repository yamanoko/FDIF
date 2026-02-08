# FDSLxSDF4Seg

**F**oundation **D**ataset for **S**egmentation **L**earning with **SDF**s for **3D Seg**mentation

3Dセグメンテーションタスクのための、SDF（Signed Distance Function）ベースの合成データセット生成と深層学習モデル訓練のための統合フレームワーク。

## 🎯 概要

このプロジェクトは、医療画像解析や3Dコンピュータビジョンの研究において、高品質な3Dセグメンテーションモデルを効率的に開発するためのツールセットです。

### 主要特徴

- 🎲 **豊富な合成データ生成**: 100種類以上のSDFプリミティブと変位関数・マッパーの組み合わせによる高品質な合成データセット
- 🔧 **ハイブリッドプリミティブシステム**: Base Primitives × SDF Mappers × Displacement Functionsの4層構成で数百種類の独自クラスを生成可能
- 🏗️ **最新アーキテクチャ**: VNet、UNETR、SwinUNETRをサポート（マルチタスク学習にも対応）
- 🔄 **転移学習**: 事前訓練済みモデルからのファインチューニング
- 📊 **実データ対応**: BTCV等の実データセット、nnUNetフォーマットとの互換性
- 🎯 **分類タスク対応**: SSL3D_classification用のデータセット生成をサポート
- ⚡ **高速訓練**: 混合精度とスライディングウィンドウ推論による最適化

## 🚀 クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/your-username/FDSLxSDF4Seg.git
cd FDSLxSDF4Seg

# 依存関係のインストール（uvパッケージマネージャーを推奨）
uv pip install -r requirements.txt
# または通常のpip
pip install -r requirements.txt
```

### 2. 合成データセット生成（セグメンテーション用）

```bash
# 基本的なデータセット生成
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./synthetic_dataset \
    --num_samples 500 \
    --num_val_samples 100 \
    --num_visualize 10

# ハイブリッドプリミティブを使用（プリミティブ × マッパー）
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./hybrid_dataset \
    --D 96 --H 96 --W 96 \
    --num_samples 500 \
    --primitives sphere cylinder torus \
    --sdf_mappers inverse_cube linear exponential

# 変位関数を追加（プリミティブ × マッパー × 変位）
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./displaced_dataset \
    --D 96 --H 96 --W 96 \
    --num_samples 500 \
    --primitives sphere cylinder \
    --sdf_mappers inverse_cube linear \
    --displacement_functions perlin turbulence sine

# マルチタスク学習用データセット（shape, displacement, mapper を個別に予測）
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./multi_task_dataset \
    --D 96 --H 96 --W 96 \
    --num_samples 500 \
    --primitives sphere cylinder torus \
    --sdf_mappers inverse_cube linear \
    --displacement_functions perlin turbulence \
    --multi_task

# nnUNetフォーマットで生成
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --nnunet_format \
    --dataset_id 999 \
    --dataset_name SDFSynthetic \
    --D 96 --H 96 --W 96 \
    --num_samples 500 \
    --primitives sphere cylinder torus
```

### 3. 分類タスク用データセット生成（SSL3D_classification用）

```bash
# 分類用データセット（1サンプル = 1オブジェクト、Blosc2形式）
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir ./cls_dataset \
    --D 96 --H 96 --W 96 \
    --samples_per_class 50 \
    --primitives sphere cylinder torus cone \
    --sdf_mappers inverse_cube linear \
    --grid_scale 0.45 \
    --dataset_name my_sdf_classification
```

### 3. 分類タスク用データセット生成（SSL3D_classification用）

```bash
# 分類用データセット（1サンプル = 1オブジェクト、Blosc2形式）
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir ./cls_dataset \
    --D 96 --H 96 --W 96 \
    --samples_per_class 50 \
    --primitives sphere cylinder torus cone \
    --sdf_mappers inverse_cube linear \
    --grid_scale 0.45 \
    --dataset_name my_sdf_classification
```

### 4. モデル訓練

```bash
# VNetでの基本訓練
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./synthetic_dataset/data/data.json \
    --model_name vnet \
    --out_channel 5

# SwinUNETRでの高精度訓練
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./synthetic_dataset/data/data.json \
    --model_name swin_unetr \
    --feature_size 48 \
    --grid_size 96 96 96 \
    --max_iterations 50000

# マルチタスク訓練（UNETR/SwinUNETRのみ）
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./multi_task_dataset/data/data.json \
    --model_name swin_unetr \
    --multi_task \
    --grid_size 96 96 96 \
    --max_iterations 30000
```

## 📁 プロジェクト構造

```
FDSLxSDF4Seg/
├── src/fdslxsdf4seg/
│   ├── generate_sdf_dataset.py              # 合成データ生成（セグメンテーション用）
│   ├── generate_sdf_dataset_classification.py  # 分類用データ生成（SSL3D互換）
│   ├── training.py                          # モデル訓練
│   ├── sdf_object.py                        # SDFObject基底クラス
│   ├── basic_sdf.py                         # 基本プリミティブ
│   ├── sdf_mapper.py                        # SDF → 強度値マッピング
│   ├── displacement_functions.py            # 表面変形関数
│   ├── hybrid_primitive.py                  # Primitive × Mapper
│   ├── displaced_primitive.py               # Primitive × Displacement
│   ├── primitive_registry.py                # 100+ プリミティブカタログ
│   ├── sector_polygon_prism/               # 多角柱プリミティブ
│   ├── star_polygon_prism/                 # 星型多角柱プリミティブ
│   ├── onioned_prism/                      # 多層プリミティブ
│   └── revolution/                         # 回転体プリミティブ
├── outputs/                                 # 生成データの出力先
├── training_output/                        # 訓練モデルの出力先
├── visualize_output/                       # 可視化出力
├── paper_figures/                          # 論文用図生成スクリプト
├── BTCV/                                   # 実データセット（オプション）
├── visualize_primitives.py                # プリミティブ可視化ツール
├── requirements.txt                        # 依存関係
├── pyproject.toml                         # プロジェクト設定
├── README.md                              # このファイル
├── generate_sdf_dataset.md                # データ生成の詳細ドキュメント
├── training.md                            # 訓練の詳細ドキュメント
├── SDF_CLASSIFICATION_DATASET.md          # 分類データセット仕様
├── VISUALIZATION_README.md                # 可視化ツールガイド
└── VARIATIONS_README.md                   # プリミティブバリエーション

## 🛠️ 詳細ドキュメント

- **[generate_sdf_dataset.md](generate_sdf_dataset.md)** - SDF合成データセット生成の詳細（セグメンテーション用）
- **[training.md](training.md)** - 3Dセグメンテーションモデル訓練の詳細
- **[SDF_CLASSIFICATION_DATASET.md](SDF_CLASSIFICATION_DATASET.md)** - 分類タスク用データセット仕様
- **[VISUALIZATION_README.md](VISUALIZATION_README.md)** - プリミティブ可視化ツールの使い方
- **[VARIATIONS_README.md](VARIATIONS_README.md)** - プリミティブバリエーション生成ガイド

## 🎨 ハイブリッドプリミティブシステム

このプロジェクトのコア機能は、**4層のプリミティブ構成システム**です：

### 1. Base Primitives（基本プリミティブ）
100種類以上の3D幾何学形状を提供：

| カテゴリ | プリミティブ例 | 数 |
|---------|--------------|-----|
| **Basic** | Sphere, Cylinder, Torus, Cone, Octahedron | 9種類 |
| **Sector Polygon Prisms** | Triangle/Square/Pentagon/Hexagon/Heptagon/Octagon/Nonagon Prism (各4種) | 28種類 |
| **Star Polygon Prisms** | 5/6/7/8-Star Prism (各4種) | 16種類 |
| **Onioned Sector Prisms** | 多層Triangle～Nonagon Prism (各4種) | 28種類 |
| **Onioned Star Prisms** | 多層5/6/7/8-Star Prism (各4種) | 16種類 |
| **Revolution Shapes** | Star Revolution (3/4/5-Star) | 3種類 |

**使用例**：
```bash
# 特定のカテゴリから選択
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --categories basic sector_polygon \
    --num_samples 500

# 特定のプリミティブのみ
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --primitives sphere cylinder torus \
    --num_samples 500
```

### 2. SDF Mappers（マッピング関数）
SDF距離値を強度値に変換する関数：

| Mapper | 数式 | 特徴 |
|--------|------|------|
| **inverse_cube** | 128/(|x|³+1) | 強い境界強調（デフォルト） |
| **linear** | -x | シンプルな線形変換 |
| **exponential** | 2^(-|x|) | 指数的減衰 |
| **floor** | floor(-x/5)*16 | 階段状の強度 |
| **modular** | (x mod 10)*12.8 | 周期的パターン |
| **sinusoidal** | 64*sin(x/5)+64 | 正弦波パターン |

**使用例**：
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --primitives sphere cylinder \
    --sdf_mappers inverse_cube linear exponential \
    --num_samples 500
```
→ 2 primitives × 3 mappers = 6クラス + 背景 = `--out_channel 7`

### 3. Displacement Functions（変位関数）
プリミティブ表面を変形させる関数：

| Displacement | 説明 | パラメータ |
|-------------|------|-----------|
| **sine** / **sine_large** | 正弦波変位 | amplitude: 0.05/0.15 |
| **perlin** / **perlin_fine** | パーリンノイズ | scale: 10.0/20.0 |
| **turbulence** / **turbulence_strong** | 乱流効果 | amplitude: 0.1/0.2 |
| **ridge** | リッジパターン | amplitude: 0.1 |
| **sharp_max** | 鋭い凹凸 | amplitude: 0.1 |
| **twist** | ねじれ変形 | angle: 45° |
| **sawtooth** | のこぎり波 | amplitude: 0.1 |

**使用例**：
```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --primitives sphere cylinder \
    --sdf_mappers inverse_cube linear \
    --displacement_functions perlin turbulence sine \
    --num_samples 500
```
→ (2×2) + (2×3×2) = 4 + 12 = 16クラス + 背景 = `--out_channel 17`

### 4. Hybrid Combinations（ハイブリッド組み合わせ）
上記を組み合わせることで、数百種類の独自クラスを生成可能：

- **HybridPrimitive**: Primitive × Mapper（例: `Cylinder_inverse_cube`）
- **DisplacedPrimitive**: Primitive × Displacement（例: `Sphere_disp_perlin`）
- **HybridDisplacedPrimitive**: Primitive × Displacement × Mapper（例: `Cylinder_disp_turbulence_linear`）

## 🎯 高度な機能

### マルチタスク学習
Shape、Displacement、Mapperを個別に予測：

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --multi_task \
    --primitives sphere cylinder torus \
    --sdf_mappers inverse_cube linear \
    --displacement_functions perlin turbulence \
    --num_samples 500

uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./output/data/data.json \
    --model_name swin_unetr \
    --multi_task
```

### データ拡張としてのMapper/Displacement
訓練時にランダムにMapper/Displacementを適用：

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --primitives sphere cylinder torus \
    --sdf_mappers inverse_cube linear exponential \
    --mapper_as_augmentation \
    --num_samples 500
```
→ クラス数はプリミティブ数のみ（3 + 1 = 4クラス）、Mapperは訓練時に動的適用

### nnUNetフォーマット対応
nnUNetで直接使用可能なデータセット生成：

```bash
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --nnunet_format \
    --dataset_id 999 \
    --dataset_name MySDFDataset \
    --primitives sphere cylinder \
    --num_samples 500
```

## 📊 サポートするプリミティブ

詳細は前述の「ハイブリッドプリミティブシステム」セクションを参照してください。100種類以上のプリミティブを利用可能です。

### プリミティブ可視化

```bash
# 全プリミティブの可視化
uv run python visualize_primitives.py

# 変位関数適用後の可視化
uv run python visualize_primitives.py --displaced --3d \
    --displaced_primitives Sphere Cylinder \
    --displacement_functions perlin turbulence

# 利用可能な変位関数一覧
uv run python visualize_primitives.py --list_displacements

# バリエーション生成
uv run python visualize_primitives.py --variations \
    --variation_primitive Cylinder --num_variations 9
```

### プリミティブ選択オプション

- **カテゴリ指定**:
  ```bash
  uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
      --categories basic sector_polygon star_polygon
  ```

- **個別指定**:
  ```bash
  uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
      --primitives sphere cylinder torus cone
  ```

- **クラス数指定**:
  ```bash
  uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
      --num_classes 10  # ランダムに10個選択
  ```

### データ生成のコマンドラインオプション（主要なもの）

| オプション | 説明 | デフォルト | 例 |
|-----------|------|-----------|-----|
| `--out_dir` | 出力ディレクトリ | 自動生成 | `./my_dataset` |
| `--D`, `--H`, `--W` | グリッドサイズ | 64 | `--D 128 --H 128 --W 128` |
| `--num_samples` | 訓練サンプル数 | 200 | `--num_samples 1000` |
| `--num_val_samples` | 検証サンプル数 | 0 | `--num_val_samples 100` |
| `--min_objects` | 最小オブジェクト数 | 2 | `--min_objects 1` |
| `--max_objects` | 最大オブジェクト数 | 5 | `--max_objects 10` |
| `--primitives` | 使用プリミティブ | デフォルト4種 | `--primitives sphere cylinder` |
| `--categories` | プリミティブカテゴリ | - | `--categories basic revolution` |
| `--sdf_mappers` | SDFマッパー | inverse_cube | `--sdf_mappers inverse_cube linear` |
| `--displacement_functions` | 変位関数 | なし | `--displacement_functions perlin sine` |
| `--multi_task` | マルチタスク学習 | False | `--multi_task` |
| `--mapper_as_augmentation` | Mapperを拡張として使用 | False | `--mapper_as_augmentation` |
| `--nnunet_format` | nnUNet形式で出力 | False | `--nnunet_format` |
| `--seed` | 乱数シード | None | `--seed 42` |
| `--num_visualize` | 可視化サンプル数 | 0 | `--num_visualize 10` |

詳細は [generate_sdf_dataset.md](generate_sdf_dataset.md) を参照してください。

## 🤖 サポートモデル

| モデル | 特徴 | 用途 | メモリ要件 |
|--------|------|------|------------|
| **VNet** | V字型3D CNN | 一般的なセグメンテーション | 低 |
| **UNETR** | Vision Transformer + CNN | 高精度セグメンテーション | 中 |
| **SwinUNETR** | Swin Transformer | 最高精度セグメンテーション | 高 |

## 📈 ベンチマーク性能

### 合成データ（SDF）
- **データサイズ**: 64³ / 96³
- **クラス数**: 4～20+クラス（プリミティブ × Mapper × Displacement による）
- **期待Dice係数**: 0.85-0.95
- **訓練時間**: 1-2時間（RTX 3080）

### 実データ（BTCV）
- **データサイズ**: 96³
- **クラス数**: 14クラス
- **期待Dice係数**: 0.75-0.85
- **訓練時間**: 8-12時間（RTX 3080）

### マルチタスク学習
- **タスク数**: 3（shape, displacement, mapper）
- **期待Dice係数**: タスクごとに0.80-0.90
- **訓練時間**: 2-3時間（RTX 3080）

## 🔧 システム要件

### 最小要件
- Python 3.8+
- PyTorch 1.12+
- CUDA対応GPU（4GB VRAM以上）

### 推奨要件
- Python 3.9+
- PyTorch 2.0+
- CUDA対応GPU（16GB VRAM以上）
- SSDストレージ

### 依存関係
```
torch>=1.12.0
monai>=1.0.0
nibabel>=3.2.0
plotly>=5.0.0
kaleido>=0.2.0
numpy>=1.21.0
tqdm>=4.60.0
blosc2>=2.0.0
```

推奨: `uv` パッケージマネージャーを使用（高速・再現性の高い依存関係管理）

## 🎯 使用例とユースケース

### 研究・開発
- 新しいセグメンテーションアルゴリズムの評価
- 合成データでの事前訓練 → 実データでのファインチューニング
- データ拡張手法の検証
- マルチタスク学習の研究

### 教育
- 3Dセグメンテーションの学習教材
- 深層学習の実践的な演習
- SDFとプリミティブの理解
- 幾何学形状とDeep Learningの関係性

### プロトタイピング
- 新しいアーキテクチャの迅速な検証
- ハイパーパラメータ最適化
- ベースライン性能の確立
- Mapper/Displacement関数の効果検証

### 段階的学習戦略
1. **基本プリミティブ**: 最も単純なケースから開始
   ```bash
   uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
       --primitives sphere --max_objects 3
   ```

2. **ハイブリッドプリミティブ**: Mapperを追加して複雑度を増加
   ```bash
   uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
       --primitives sphere cylinder \
       --sdf_mappers inverse_cube linear --max_objects 5
   ```

3. **変位プリミティブ**: Displacementで表面を変形
   ```bash
   uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
       --primitives sphere cylinder torus \
       --sdf_mappers inverse_cube linear \
       --displacement_functions perlin turbulence --max_objects 5
   ```

4. **マルチタスク学習**: 最終的な複雑なタスク
   ```bash
   uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
       --primitives sphere cylinder torus cone \
       --sdf_mappers inverse_cube linear exponential \
       --displacement_functions perlin turbulence sine \
       --multi_task --max_objects 8
   ```

## 📝 ワークフロー例

### 1. 合成データでの事前訓練（ハイブリッドプリミティブ使用）
```bash
# Step 1: ハイブリッド合成データ生成
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./pretraining_data \
    --num_samples 1000 \
    --D 96 --H 96 --W 96 \
    --primitives sphere cylinder torus cone \
    --sdf_mappers inverse_cube linear exponential \
    --displacement_functions perlin turbulence

# Step 2: 事前訓練
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./pretraining_data/data/data.json \
    --model_name swin_unetr \
    --out_channel 17 \
    --max_iterations 30000 \
    --out_dir ./pretrained_models
```

### 2. 実データでのファインチューニング
```bash
# Step 3: ファインチューニング
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./BTCV/dataset.json \
    --model_name swin_unetr \
    --is_real_data \
    --pretrained_model ./pretrained_models/best_metric_model.pth \
    --pretraining_out_channel 17 \
    --out_channel 14 \
    --max_iterations 20000
```

### 3. マルチタスク学習ワークフロー
```bash
# Step 1: マルチタスクデータセット生成
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./multi_task_data \
    --num_samples 1000 \
    --D 96 --H 96 --W 96 \
    --primitives sphere cylinder torus \
    --sdf_mappers inverse_cube linear \
    --displacement_functions perlin turbulence sine \
    --multi_task

# Step 2: マルチタスク訓練
uv run python src/fdslxsdf4seg/training.py \
    --data_json_path ./multi_task_data/data/data.json \
    --model_name swin_unetr \
    --multi_task \
    --max_iterations 30000
```

### 4. 分類タスクワークフロー（SSL3D_classification用）
```bash
# Step 1: 分類用データセット生成
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir ./classification_data \
    --D 96 --H 96 --W 96 \
    --samples_per_class 100 \
    --primitives sphere cylinder torus cone \
    --sdf_mappers inverse_cube linear \
    --grid_scale 0.45 \
    --dataset_name sdf_classification

# Step 2: SSL3D_classificationで訓練
# （別途SSL3D_classificationリポジトリで実行）
```

## 🔍 トラブルシューティング

### よくある問題と解決策

#### メモリ不足エラー
```bash
# 解決策: バッチサイズとグリッドサイズを減らす
uv run python src/fdslxsdf4seg/training.py ... \
    --batch_size 1 --grid_size 64 64 64
```

#### クラス数の不一致エラー
```bash
# 問題: --out_channel と実際のクラス数が一致しない
# 解決策: generation_log.txt でクラス数を確認
cat ./output_dir/generation_log.txt | grep "Total classes"

# または data.json から確認
python -c "import json; print(len(json.load(open('data.json'))['labels']))"
```

#### 収束しない
```bash
# 解決策: 学習率を下げて長時間訓練
uv run python src/fdslxsdf4seg/training.py ... --max_iterations 50000
```

#### 可視化エラー
```bash
# 解決策: 可視化用パッケージをインストール
uv pip install kaleido
# または
pip install kaleido
```

#### Displacement/Mapper名のエラー
```bash
# 利用可能なDisplacement関数を確認
uv run python visualize_primitives.py --list_displacements

# 利用可能なMapper一覧（コード内で定義）:
# inverse_cube, linear, exponential, floor, modular, sinusoidal
```

#### マルチタスク学習で通常データセットを使用
```bash
# 問題: --multi_task フラグを使用したがデータセットが通常形式
# 解決策: データセット生成時にも --multi_task を指定
uv run python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --multi_task ...
```

詳細なトラブルシューティングは各ドキュメントを参照してください。

## 🤝 コントリビューション

プロジェクトへの貢献を歓迎します！以下の方法で貢献できます：

1. **バグレポート**: Issuesでバグを報告
2. **機能提案**: 新機能のアイデアを提案
3. **プルリクエスト**: コードの改善を提案
4. **ドキュメント改善**: 説明の追加・修正

### 開発の流れ
1. リポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add amazing feature'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトは [MIT License](LICENSE) の下で公開されています。

## 📚 引用

研究でこのプロジェクトを使用される場合は、以下の形式での引用をお願いします：

```bibtex
@software{fdslxsdf4seg2025,
  title={FDSLxSDF4Seg: Foundation Dataset for Segmentation Learning with SDFs},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/FDSLxSDF4Seg}
}
```

## 📞 サポート

- 📧 **Email**: your.email@example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-username/FDSLxSDF4Seg/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-username/FDSLxSDF4Seg/discussions)

## 🔗 関連リンク

- [MONAI Framework](https://monai.io/)
- [PyTorch](https://pytorch.org/)
- [NiBabel](https://nipy.org/nibabel/)
- [Plotly](https://plotly.com/python/)

---

⭐ このプロジェクトが役に立った場合は、スターをつけていただけると嬉しいです！
