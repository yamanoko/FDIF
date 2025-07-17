# FDSLxSDF4Seg

**F**oundation **D**ataset for **S**egmentation **L**earning with **SDF**s for **3D Seg**mentation

3Dセグメンテーションタスクのための、SDF（Signed Distance Function）ベースの合成データセット生成と深層学習モデル訓練のための統合フレームワーク。

## 🎯 概要

このプロジェクトは、医療画像解析や3Dコンピュータビジョンの研究において、高品質な3Dセグメンテーションモデルを効率的に開発するためのツールセットです。

### 主要特徴

- 🎲 **合成データ生成**: SDFベースの3Dプリミティブを使用した高品質な合成データセット
- 🏗️ **最新アーキテクチャ**: VNet、UNETR、SwinUNETRをサポート
- 🔄 **転移学習**: 事前訓練済みモデルからのファインチューニング
- 📊 **実データ対応**: BTCV等の実データセットとの互換性
- ⚡ **高速訓練**: 混合精度とスライディングウィンドウ推論による最適化

## 🚀 クイックスタート

### 1. 環境構築

```bash
# リポジトリのクローン
git clone https://github.com/your-username/FDSLxSDF4Seg.git
cd FDSLxSDF4Seg

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 合成データセット生成

```bash
# 基本的なデータセット生成
python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./synthetic_dataset \
    --num_samples 500 \
    --num_val_samples 100 \
    --num_visualize 10

# 高解像度データセット生成
python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./high_res_dataset \
    --D 128 --H 128 --W 128 \
    --num_samples 200 \
    --min_objects 2 --max_objects 4

# 特定のプリミティブのみを使用した生成
python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./sphere_box_dataset \
    --num_samples 300 \
    --primitives sphere box \
    --max_objects 6

# 単一プリミティブでの生成
python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./cylinder_only_dataset \
    --num_samples 200 \
    --primitives cylinder \
    --max_objects 8
```

### 3. モデル訓練

```bash
# VNetでの基本訓練
python src/fdslxsdf4seg/training.py \
    --data_json_path ./synthetic_dataset/data/data.json \
    --model_name vnet \
    --out_channel 5

# SwinUNETRでの高精度訓練
python src/fdslxsdf4seg/training.py \
    --data_json_path ./synthetic_dataset/data/data.json \
    --model_name swin_unetr \
    --feature_size 48 \
    --grid_size 96 96 96 \
    --max_iterations 50000
```

## 📁 プロジェクト構造

```
FDSLxSDF4Seg/
├── src/fdslxsdf4seg/
│   ├── generate_sdf_dataset.py   # 合成データ生成
│   ├── training.py               # モデル訓練
│   └── __init__.py
├── outputs/                      # 生成データの出力先
├── BTCV/                        # 実データセット（オプション）
├── requirements.txt             # 依存関係
├── pyproject.toml              # プロジェクト設定
├── README.md                   # このファイル
├── generate_sdf_dataset.md     # データ生成の詳細ドキュメント
└── training.md                 # 訓練の詳細ドキュメント
```

## 🛠️ 詳細ドキュメント

- **[generate_sdf_dataset.md](generate_sdf_dataset.md)** - SDF合成データセット生成の詳細
- **[training.md](training.md)** - 3Dセグメンテーションモデル訓練の詳細

## 📊 サポートするプリミティブ

| プリミティブ | 説明 | パラメータ | 使用例 |
|-------------|------|------------|--------|
| 🔵 **Sphere** | 球体 | 半径 | `--primitives sphere` |
| 📦 **Box** | 直方体 | 幅・高さ・奥行き | `--primitives box` |
| 🗼 **Cylinder** | 円柱 | 半径・高さ | `--primitives cylinder` |
| 🍩 **Torus** | トーラス | 大半径・小半径 | `--primitives torus` |

### プリミティブ選択オプション

- **全プリミティブ使用**（デフォルト）:
  ```bash
  python generate_sdf_dataset.py --primitives sphere box cylinder torus
  ```

- **複数プリミティブ選択**:
  ```bash
  python generate_sdf_dataset.py --primitives sphere box
  ```

- **単一プリミティブ**:
  ```bash
  python generate_sdf_dataset.py --primitives cylinder
  ```

**注意**: 選択したプリミティブの種類により、セグメンテーションのクラス数が変わります。`--max_objects`パラメータで、同じプリミティブの複数インスタンスを生成できます。

### データ生成のコマンドラインオプション

| オプション | 説明 | デフォルト | 例 |
|-----------|------|-----------|-----|
| `--out_dir` | 出力ディレクトリ | 自動生成 | `./my_dataset` |
| `--D`, `--H`, `--W` | グリッドサイズ | 64 | `--D 128 --H 128 --W 128` |
| `--num_samples` | 訓練サンプル数 | 200 | `--num_samples 1000` |
| `--num_val_samples` | 検証サンプル数 | 0 | `--num_val_samples 100` |
| `--min_objects` | 最小オブジェクト数 | 2 | `--min_objects 1` |
| `--max_objects` | 最大オブジェクト数 | 5 | `--max_objects 10` |
| `--primitives` | 使用プリミティブ | 全て | `--primitives sphere box` |
| `--seed` | 乱数シード | None | `--seed 42` |
| `--num_visualize` | 可視化サンプル数 | 0 | `--num_visualize 10` |

## 🤖 サポートモデル

| モデル | 特徴 | 用途 | メモリ要件 |
|--------|------|------|------------|
| **VNet** | V字型3D CNN | 一般的なセグメンテーション | 低 |
| **UNETR** | Vision Transformer + CNN | 高精度セグメンテーション | 中 |
| **SwinUNETR** | Swin Transformer | 最高精度セグメンテーション | 高 |

## 📈 ベンチマーク性能

### 合成データ（SDF）
- **データサイズ**: 64³ / 96³
- **クラス数**: 4-5クラス
- **期待Dice係数**: 0.85-0.95
- **訓練時間**: 1-2時間（RTX 3080）

### 実データ（BTCV）
- **データサイズ**: 96³
- **クラス数**: 14クラス
- **期待Dice係数**: 0.75-0.85
- **訓練時間**: 8-12時間（RTX 3080）

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
numpy>=1.21.0
tqdm>=4.60.0
```

## 🎯 使用例とユースケース

### 研究・開発
- 新しいセグメンテーションアルゴリズムの評価
- 合成データでの事前訓練 → 実データでのファインチューニング
- データ拡張手法の検証

### 教育
- 3Dセグメンテーションの学習教材
- 深層学習の実践的な演習
- SDFとプリミティブの理解

### プロトタイピング
- 新しいアーキテクチャの迅速な検証
- ハイパーパラメータ最適化
- ベースライン性能の確立

### 段階的学習戦略
1. **単一プリミティブ**: 最も単純なケースから開始
   ```bash
   python generate_sdf_dataset.py --primitives sphere --max_objects 3
   ```

2. **複数プリミティブ**: 複雑度を徐々に増加
   ```bash
   python generate_sdf_dataset.py --primitives sphere box --max_objects 5
   ```

3. **全プリミティブ**: 最終的な複雑なシーンで評価
   ```bash
   python generate_sdf_dataset.py --primitives sphere box cylinder torus --max_objects 8
   ```

## 📝 ワークフロー例

### 1. 合成データでの事前訓練
```bash
# Step 1: 合成データ生成（全プリミティブ使用）
python src/fdslxsdf4seg/generate_sdf_dataset.py \
    --out_dir ./pretraining_data \
    --num_samples 1000 \
    --D 96 --H 96 --W 96 \
    --primitives sphere box cylinder torus

# Step 2: 事前訓練
python src/fdslxsdf4seg/training.py \
    --data_json_path ./pretraining_data/data/data.json \
    --model_name swin_unetr \
    --out_channel 5 \
    --max_iterations 30000 \
    --out_dir ./pretrained_models
```

### 2. 実データでのファインチューニング
```bash
# Step 3: ファインチューニング
python src/fdslxsdf4seg/training.py \
    --data_json_path ./BTCV/dataset.json \
    --model_name swin_unetr \
    --is_real_data \
    --pretrained_model ./pretrained_models/best_metric_model.pth \
    --pretraining_out_channel 5 \
    --out_channel 14 \
    --max_iterations 20000
```

## 🔍 トラブルシューティング

### よくある問題と解決策

#### メモリ不足エラー
```bash
# 解決策: バッチサイズとグリッドサイズを減らす
python training.py ... --batch_size 1 --grid_size 64 64 64
```

#### 収束しない
```bash
# 解決策: 学習率を下げて長時間訓練
python training.py ... --max_iterations 50000
```

#### 可視化エラー
```bash
# 解決策: 可視化用パッケージをインストール
pip install kaleido
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
