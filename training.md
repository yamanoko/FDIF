# training.py

## 概要

`training.py`は、MONAIフレームワークを使用して3Dセグメンテーションモデルを訓練するためのPythonスクリプトです。VNet、UNETR、SwinUNETRなどの最新の3Dセグメンテーションアーキテクチャをサポートし、実データと合成データの両方に対応しています。事前訓練済みモデルからのファインチューニングも可能です。

## 主要機能

### 1. サポートモデル
- **VNet**: 3D医療画像セグメンテーション用のV字型ネットワーク
- **UNETR**: Vision Transformer基盤のU字型ネットワーク
- **SwinUNETR**: Swin Transformer基盤のU字型ネットワーク

### 2. データ処理機能
- **実データ対応**: BTCV等の医療画像データセット
- **合成データ対応**: SDF生成データセット
- **複数モダリティ対応**: DWI+ADC（ISLES）など複数チャンネル入力
- **データ拡張**: フリップ、回転、強度シフト等
- **自動前処理**: 正規化、クロップ、リサンプリング

### 3. 訓練機能
- **混合精度訓練**: CUDA AMPによる高速化
- **スライディングウィンドウ推論**: 大きな画像の効率的な処理
- **メトリクス追跡**: Dice係数による性能評価
- **モデル保存**: 最高性能モデルの自動保存

### 4. 事前訓練サポート
- 事前訓練済みモデルの読み込み
- 出力層の調整による転移学習
- ファインチューニングと从零训练の両方に対応

## 使用法

### 基本的な使用方法

```bash
python training.py \
    --data_json_path ./data/data.json \
    --model_name vnet \
    --out_channel 5
```

### パラメータ詳細

| パラメータ | 必須 | デフォルト値 | 説明 |
|-----------|------|-------------|------|
| `--data_json_path` | ✓ | - | データセットJSONファイルのパス |
| `--model_name` | ✓ | - | モデル名（vnet/unetr/swin_unetr） |
| `--is_real_data` | - | False | 実データ使用フラグ |
| `--pretrained_model` | - | None | 事前訓練済みモデルのパス |
| `--pretraining_out_channel` | - | 14 | 事前訓練モデルの出力チャンネル数 |
| `--grid_size` | - | [96,96,96] | 入力グリッドサイズ |
| `--out_channel` | - | 14 | 出力チャンネル数（クラス数+1、背景含む） |
| `--in_channels` | - | 1 | 入力チャンネル数（モダリティ数） |
| `--feature_size` | - | 自動設定 | 特徴量サイズ |
| `--batch_size` | - | 1 | バッチサイズ |
| `--max_iterations` | - | 30000 | 最大訓練イテレーション数 |
| `--out_dir` | - | 自動生成 | 出力ディレクトリ |

### 出力チャンネル数の設定

プリミティブ選択により出力チャンネル数が変わります：

| プリミティブ構成 | 出力チャンネル数 | 例 |
|-----------------|----------------|-----|
| 単一プリミティブ | 2 | `--primitives sphere --out_channel 2` |
| 2種類プリミティブ | 3 | `--primitives sphere box --out_channel 3` |
| 3種類プリミティブ | 4 | `--primitives sphere box cylinder --out_channel 4` |
| 全プリミティブ | 5 | `--primitives sphere box cylinder torus --out_channel 5` |
| 実データ（BTCV） | 14 | `--is_real_data --out_channel 14` |

### 使用例

#### 1. 合成データでの基本訓練
```bash
python training.py \
    --data_json_path ./synthetic_dataset/data/data.json \
    --model_name vnet \
    --out_channel 5 \
    --grid_size 64 64 64 \
    --max_iterations 10000
```

#### 2. 実データでの訓練
```bash
python training.py \
    --data_json_path ./BTCV/dataset.json \
    --model_name swin_unetr \
    --is_real_data \
    --out_channel 14 \
    --feature_size 48 \
    --batch_size 2
```

#### 3. 事前訓練済みモデルのファインチューニング
```bash
python training.py \
    --data_json_path ./data/data.json \
    --model_name unetr \
    --pretrained_model ./pretrained/model_best.pth \
    --pretraining_out_channel 14 \
    --out_channel 5 \
    --grid_size 96 96 96
```

#### 4. 高解像度データでの訓練
```bash
python training.py \
    --data_json_path ./data/data.json \
    --model_name swin_unetr \
    --grid_size 128 128 128 \
    --feature_size 48 \
    --batch_size 1 \
    --max_iterations 50000
```

#### 5. 特定プリミティブデータでの訓練
```bash
# 単一プリミティブデータ（2クラス: 背景+プリミティブ）
python training.py \
    --data_json_path ./sphere_only_dataset/data/data.json \
    --model_name vnet \
    --out_channel 2 \
    --grid_size 64 64 64

# 複数プリミティブデータ（3クラス: 背景+2プリミティブ）
python training.py \
    --data_json_path ./sphere_box_dataset/data/data.json \
    --model_name swin_unetr \
    --out_channel 3 \
    --grid_size 96 96 96
```

#### 6. 複数モダリティデータでの訓練（例：ISLES DWI+ADC）
```bash
# DWI+ADCの2モダリティ（ISLESデータ）
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name vnet \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data \
    --grid_size 96 96 96 \
    --max_iterations 30000

# 複数モダリティ+SwinUNETR
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name swin_unetr \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data \
    --feature_size 48 \
    --grid_size 96 96 96
```

## 複数モダリティ（マルチモーダル）入力

### 概要
トレーニングスクリプトは複数のモダリティ（例：DWI、ADC、T1、T2など）を持つ医療画像をサポートしています。

### 入力画像形式
複数モダリティデータは4次元NIfTIファイルである必要があります：
- **形状**: (C, D, H, W) ここで C = モダリティ数
- **例**:
  - ISLES (DWI + ADC): (2, 96, 112, 96)
  - T1 + T2 + FLAIR: (3, 256, 256, 128)

### パラメータ設定
```bash
--in_channels 2      # DWI + ADC
--in_channels 3      # T1 + T2 + FLAIR
--in_channels 4      # 4つのモダリティ
```

### 実装例
```bash
# ISLESデータセット（DWI + ADC）の訓練
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name vnet \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data \
    --grid_size 96 96 96
```

### 注意点
1. すべての入力画像は同じ数のチャンネルを持つ必要があります
2. チャンネルの順序は統一されている必要があります（例：常にDWI、ADCの順）
3. データセット変換時に正しくチャンネルが統合されていることを確認してください
4. 事前訓練済みモデルを使用する場合、`--in_channels`と訓練時の値が一致する必要があります

### データセット変換例
ISLES-2022をFDSLxSDF4Seg形式に変換する場合：
```bash
# 1. nnUNet形式に変換
python convert_isles_nnunet.py \
    -i /path/to/ISLES-2022 \
    -o $nnUNet_raw

# 2. FDSLxSDF4Seg形式に変換（チャンネル統合）
python convert_nnunet_to_fdsls4seg.py \
    --input $nnUNet_raw/Dataset500_ISLES2022 \
    --output ~/ISLES_fdsls4seg

# 3. 2チャンネルで訓練
python training.py \
    --data_json_path ~/ISLES_fdsls4seg/data.json \
    --model_name vnet \
    --in_channels 2 \
    --out_channel 2 \
    --is_real_data
```

## データ要件

### データセットJSON形式
```json
{
    "training": [
        {
            "image": "path/to/image.nii.gz",
            "label": "path/to/label.nii.gz",
            "id": "sample_id"
        }
    ],
    "validation": [...]
}
```

### 画像形式
- **ファイル形式**: NIfTI (.nii.gz)
- **次元**: 
  - 単一モダリティ: 3D (D × H × W)
  - 複数モダリティ: 4D (C × D × H × W) ここで C = `--in_channels`
- **画像**: グレースケール強度値（正規化推奨）
- **ラベル**: 整数値クラスID（0=背景、1-N=各クラス）

### プリミティブ選択によるクラス数変更

生成データセットの使用プリミティブによってクラス数が変わります：

```bash
# 例：sphere、boxのみ使用した場合
# クラス: 0=背景, 1=sphere, 2=box → 3クラス
--out_channel 3

# 例：全プリミティブ使用した場合  
# クラス: 0=背景, 1=sphere, 2=box, 3=cylinder, 4=torus → 5クラス
--out_channel 5
```

## 出力構造

訓練完了後、以下の構造でファイルが出力されます：

```
output_directory/
├── training_log.txt           # 訓練ログと設定
├── best_metric_model.pth      # 最高性能モデル
└── (将来的に追加される可能性のあるファイル)
```

## モデル詳細

### VNet
- **特徴**: 3D畳み込みベースのV字型アーキテクチャ
- **用途**: 中程度サイズの3Dセグメンテーション
- **メモリ**: 比較的軽量

### UNETR
- **特徴**: Vision Transformerエンコーダ + CNNデコーダ
- **用途**: 高精度セグメンテーション
- **パラメータ**: `feature_size`（デフォルト: 16）

### SwinUNETR
- **特徴**: Swin Transformerベースの階層的アーキテクチャ
- **用途**: 最高精度のセグメンテーション
- **パラメータ**: `feature_size`（デフォルト: 48）

## データ拡張

### 実データの場合
- スケール強度正規化（-175〜250 → 0〜1）
- 前景クロップ
- 方向正規化（RAS）
- 空間リサンプリング（1.5×1.5×2.0mm）

### 共通拡張
- ランダムクロップ（正負サンプル考慮）
- ランダムフリップ（各軸10%確率）
- ランダム90度回転（10%確率）
- ランダム強度シフト（50%確率、±10%）

## 訓練設定

### 最適化
- **オプティマイザ**: AdamW
- **学習率**: 1e-4
- **重み減衰**: 1e-5
- **混合精度**: CUDA AMP使用

### 損失関数
- **DiceCELoss**: Dice損失 + Cross Entropy損失
- **One-hot変換**: 自動実行
- **Softmax**: 自動適用

### 評価
- **メトリクス**: Dice係数
- **評価頻度**: 500イテレーション毎
- **推論方式**: スライディングウィンドウ（重複度4）

## パフォーマンス最適化

### メモリ最適化
1. **バッチサイズ調整**: GPUメモリに応じて1-2に設定
2. **グリッドサイズ**: メモリ制約に応じて64-128に調整
3. **キャッシュ設定**: データローダーのキャッシュ数を調整

### 速度最適化
1. **混合精度**: 自動有効化
2. **データローダー**: マルチプロセッシング
3. **キャッシュデータセット**: 頻繁アクセスデータの高速化

## 依存関係

### 必須パッケージ
- torch
- monai
- tqdm
- argparse

### 推奨環境
- CUDA対応GPU
- 16GB以上のGPUメモリ（高解像度データの場合）
- SSDストレージ

## トラブルシューティング

### よくある問題

#### 1. メモリ不足
```
解決策:
- batch_sizeを1に減らす
- grid_sizeを小さくする（64×64×64など）
- num_workersを0に設定
```

#### 2. 収束しない
```
解決策:
- 学習率を下げる（1e-5など）
- より多くのイテレーション実行
- データ拡張を調整
```

#### 3. 事前訓練済みモデル読み込みエラー
```
解決策:
- pretraining_out_channelを正しく設定
- モデルアーキテクチャの一致確認
- パスの正確性確認
```

### デバッグのヒント

1. **小規模実験**: 少数サンプルで動作確認
2. **ログ確認**: training_log.txtで設定確認
3. **GPUメモリ監視**: nvidia-smiでメモリ使用量確認

## ベンチマークと期待性能

### 合成データ（SDF）
- **データサイズ**: 64³
- **クラス数**: 2-5（使用プリミティブ数+1）
- **期待Dice**: 0.85-0.95
- **訓練時間**: 1-2時間（RTX 3080）

### 実データ（BTCV）
- **データサイズ**: 96³
- **クラス数**: 14
- **期待Dice**: 0.75-0.85
- **訓練時間**: 8-12時間（RTX 3080）

### 段階的学習の期待性能

| 学習段階 | プリミティブ数 | クラス数 | 期待Dice | 訓練時間 |
|---------|---------------|---------|---------|---------|
| レベル1 | 1種類 | 2 | 0.90-0.95 | 30分 |
| レベル2 | 2種類 | 3 | 0.88-0.93 | 45分 |
| レベル3 | 3種類 | 4 | 0.87-0.92 | 1時間 |
| レベル4 | 4種類 | 5 | 0.85-0.90 | 1.5時間 |
