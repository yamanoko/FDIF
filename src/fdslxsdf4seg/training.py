import argparse
import json
import os
import random

# Import visualization functions from visualize_training_metrics.py
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import (
    CacheDataset,
    Dataset,
    ThreadDataLoader,
    decollate_batch,
    load_decathlon_datalist,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.blocks import UnetOutBlock
from monai.networks.nets import UNETR, SwinUNETR, VNet
from monai.networks.nets.vnet import OutputTransition
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from torch.nn import CrossEntropyLoss, ModuleDict
from tqdm import tqdm

from fdslxsdf4seg.lr_scheduler import LinearWarmupCosineAnnealingLR
from fdslxsdf4seg.visualize_training_metrics import (
    plot_metrics,
    print_summary,
)


class MultiHeadSegmentationModel(torch.nn.Module):
    """マルチタスク学習用のラッパーモデル

    UNETR/SwinUNETRのデコーダ出力から複数の独立したセグメンテーションヘッドへ分岐する。
    各タスク（shape, displacement, mapper）に対して独立したUnetOutBlockを持つ。
    """

    def __init__(self, base_model, task_out_channels: dict, feature_size: int):
        """
        Args:
            base_model: UNETR or SwinUNETR model
            task_out_channels: dict of task name -> number of output channels
                e.g., {"shape": 5, "displacement": 3, "mapper": 3}
            feature_size: feature size of the base model (16 for UNETR, 48 for SwinUNETR)
        """
        super().__init__()
        self.base_model = base_model
        self.task_out_channels = task_out_channels
        self.feature_size = feature_size

        # 元のoutヘッドを削除し、タスク別ヘッドを作成
        self.task_heads = ModuleDict(
            {
                task: UnetOutBlock(
                    spatial_dims=3, in_channels=feature_size, out_channels=nc
                )
                for task, nc in task_out_channels.items()
            }
        )

        # フック用の変数
        self._decoder_features = None
        self._hook_handle = None

        # デコーダの最終特徴マップを取得するフックを登録
        self._register_decoder_hook()

    def _register_decoder_hook(self):
        """デコーダの最終出力（outヘッドの入力）をキャプチャするフックを登録"""

        def hook_fn(module, input, output):
            # UnetOutBlockの入力（デコーダの出力）をキャプチャ
            self._decoder_features = input[0]

        # base_model.outにフックを登録
        if hasattr(self.base_model, "out"):
            self._hook_handle = self.base_model.out.register_forward_hook(hook_fn)

    def forward(self, x):
        # ベースモデルのforward実行（フックでデコーダ特徴をキャプチャ）
        _ = self.base_model(x)

        # キャプチャしたデコーダ特徴を各タスクヘッドに通す
        outputs = {}
        for task, head in self.task_heads.items():
            outputs[task] = head(self._decoder_features)

        return outputs

    def state_dict(self, *args, **kwargs):
        """state_dictをオーバーライドして、ベースモデルとタスクヘッドを含める"""
        state = {}
        # ベースモデルのstate_dict
        for k, v in self.base_model.state_dict().items():
            state[f"base_model.{k}"] = v
        # タスクヘッドのstate_dict
        for k, v in self.task_heads.state_dict().items():
            state[f"task_heads.{k}"] = v
        return state

    def load_state_dict(self, state_dict, strict=True):
        """load_state_dictをオーバーライド"""
        base_state = {}
        head_state = {}
        for k, v in state_dict.items():
            if k.startswith("base_model."):
                base_state[k[len("base_model.") :]] = v
            elif k.startswith("task_heads."):
                head_state[k[len("task_heads.") :]] = v
        self.base_model.load_state_dict(base_state, strict=strict)
        self.task_heads.load_state_dict(head_state, strict=strict)


class MultiTaskLoss(torch.nn.Module):
    """マルチタスク学習用の損失関数

    各タスクに対してDiceCELossを計算し、重み付け合計を返す。
    """

    def __init__(
        self, task_out_channels: dict, weights: dict = None, use_ce_loss: bool = False
    ):
        """
        Args:
            task_out_channels: dict of task name -> number of output channels
            weights: dict of task name -> loss weight (default: all 1.0)
            use_ce_loss: If True, use CrossEntropyLoss instead of DiceCELoss
        """
        super().__init__()
        self.task_out_channels = task_out_channels
        self.weights = weights or {task: 1.0 for task in task_out_channels}
        self.use_ce_loss = use_ce_loss

        if use_ce_loss:
            self.losses = ModuleDict(
                {task: CrossEntropyLoss() for task in task_out_channels}
            )
        else:
            self.losses = ModuleDict(
                {
                    task: DiceCELoss(to_onehot_y=True, softmax=True)
                    for task in task_out_channels
                }
            )

    def forward(self, outputs: dict, targets: dict):
        """各タスクの損失を計算して重み付け合計を返す

        Args:
            outputs: dict of task name -> model output tensor (B, C, D, H, W)
            targets: dict of task name -> target tensor (B, 1, D, H, W)

        Returns:
            total_loss: weighted sum of per-task losses
        """
        total_loss = 0.0
        for task, loss_fn in self.losses.items():
            if self.use_ce_loss:
                # CrossEntropyLoss expects labels in (B, D, H, W) format
                target = targets[task].squeeze(1).long()
            else:
                target = targets[task]
            task_loss = loss_fn(outputs[task], target)
            total_loss = total_loss + self.weights[task] * task_loss
        return total_loss


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Additional settings for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def load_multi_task_info(data_json_path: str) -> dict:
    """Load multi-task information from data.json.

    Args:
        data_json_path: Path to data.json

    Returns:
        dict with keys:
            - multi_task: bool, whether multi-task mode is enabled
            - tasks: dict of task_name -> {"labels": {label_name: label_id, ...}}
            - task_out_channels: dict of task_name -> num_classes
            - task_names: list of task names in order (e.g., ["shape", "displacement", "mapper"])
    """
    with open(data_json_path, "r") as f:
        data_json = json.load(f)

    result = {
        "multi_task": data_json.get("multi_task", False),
        "tasks": {},
        "task_out_channels": {},
        "task_names": [],
    }

    if result["multi_task"] and "tasks" in data_json:
        result["tasks"] = data_json["tasks"]
        # タスク名の順序を維持（shape, displacement, mapperの順）
        # data.jsonに含まれるタスクのみを抽出
        canonical_order = ["shape", "displacement", "mapper"]
        for task_name in canonical_order:
            if task_name in data_json["tasks"]:
                result["task_names"].append(task_name)
                result["task_out_channels"][task_name] = len(
                    data_json["tasks"][task_name]["labels"]
                )

    return result


def make_data_loder(
    data_json_path: str,
    real_data: bool = True,
    spatial_size: tuple = (96, 96, 96),
    batch_size: int = 1,
    multi_task: bool = False,
):
    """Create data loaders for training and validation.

    Args:
        data_json_path: Path to data.json
        real_data: If True, use CacheDataset with medical image preprocessing
        spatial_size: Size of random crops
        batch_size: Batch size for training
        multi_task: If True, load multi-task labels (3 channels: shape, displacement, mapper)

    Returns:
        train_loader, val_loader
    """
    num_samples = 4
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_transforms = [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=["image", "label"], source_key="image", allow_smaller=True
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]

    if real_data:
        base_transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            )
        )

    train_transforms = base_transforms.copy()
    train_transforms.extend(
        [
            SpatialPadd(
                keys=["image", "label"],
                spatial_size=spatial_size,
                mode="constant",
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.1,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.1,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.1,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.1,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.5,
            ),
        ]
    )
    val_transforms = base_transforms.copy()
    val_transforms.extend(
        [
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )
    train_transforms = Compose(train_transforms)
    val_transforms = Compose(val_transforms)

    datalist = load_decathlon_datalist(data_json_path, True, "training")
    val_files = load_decathlon_datalist(data_json_path, True, "validation")
    if real_data:
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=4,
        )
    else:
        train_ds = Dataset(
            data=datalist,
            transform=train_transforms,
        )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    if real_data:
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=4,
        )
    else:
        val_ds = Dataset(
            data=val_files,
            transform=val_transforms,
        )
    val_loader = ThreadDataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader


def split_multi_task_labels(labels: torch.Tensor, task_names: list) -> dict:
    """Split multi-task label tensor into per-task tensors.

    Args:
        labels: Tensor of shape (B, N, D, H, W) or (B, N, 1, D, H, W)
            where N is the number of tasks
        task_names: List of task names in order (e.g., ["shape", "displacement", "mapper"]
            or ["shape", "mapper"] for 2-task mode)

    Returns:
        dict: {task_name: (B, 1, D, H, W), ...} for each task
    """
    result = {}

    # ラベルの形状を確認・調整
    if labels.dim() == 5:
        # (B, N, D, H, W) 形式
        for i, task_name in enumerate(task_names):
            result[task_name] = labels[:, i : i + 1, :, :, :]  # (B, 1, D, H, W)
    elif labels.dim() == 6:
        # (B, N, 1, D, H, W) 形式
        for i, task_name in enumerate(task_names):
            result[task_name] = labels[:, i, :, :, :, :]  # (B, 1, D, H, W)
    else:
        raise ValueError(f"Unexpected label shape: {labels.shape}")

    return result


def adapt_input_channel_weights(
    state_dict, pretraining_in_channels, target_in_channels, model_name
):
    """事前学習モデルの入力層の重みを、異なるチャンネル数のモデルに適合させる。

    1チャンネルで学習した重みを複数チャンネルモデルに転写する場合、
    各チャンネルに同じ重みをコピーし、値を 1/target_in_channels でスケールする。
    これにより、入力の合計が元のモデルと近い値になる。

    Args:
        state_dict: 事前学習モデルのstate_dict
        pretraining_in_channels: 事前学習モデルの入力チャンネル数
        target_in_channels: 新しいモデルの入力チャンネル数
        model_name: モデル名 ("vnet", "unetr", "swin_unetr")

    Returns:
        修正されたstate_dict
    """
    if pretraining_in_channels == target_in_channels:
        return state_dict

    # モデルごとの入力層の重みキーを特定
    input_weight_keys = _find_input_weight_keys(state_dict, model_name)

    if not input_weight_keys:
        print(
            f"Warning: Could not find input layer weight keys for {model_name}. "
            "Skipping input channel adaptation."
        )
        return state_dict

    for key in input_weight_keys:
        old_weight = state_dict[key]
        # 重みの形状: (out_features, in_channels, *kernel_size)
        if old_weight.shape[1] != pretraining_in_channels:
            continue

        if pretraining_in_channels == 1 and target_in_channels > 1:
            # 1ch → Nch: 各チャンネルに同じ重みをコピーしてスケーリング
            new_weight = old_weight.repeat(
                1, target_in_channels, *([1] * (old_weight.dim() - 2))
            )
            new_weight = new_weight / target_in_channels
            state_dict[key] = new_weight
            print(
                f"  Adapted '{key}': {old_weight.shape} -> {new_weight.shape} "
                f"(copied & scaled by 1/{target_in_channels})"
            )
        elif pretraining_in_channels < target_in_channels:
            # Mch → Nch (M>1, N>M): 既存チャンネルをコピーし残りを平均で埋める
            repeats = target_in_channels // pretraining_in_channels
            remainder = target_in_channels % pretraining_in_channels
            parts = [old_weight] * repeats
            if remainder > 0:
                parts.append(old_weight[:, :remainder])
            new_weight = torch.cat(parts, dim=1)
            new_weight = new_weight * (pretraining_in_channels / target_in_channels)
            state_dict[key] = new_weight
            print(
                f"  Adapted '{key}': {old_weight.shape} -> {new_weight.shape} "
                f"(tiled & scaled)"
            )
        elif pretraining_in_channels > target_in_channels:
            # Nch → Mch (N>M): 先頭チャンネルを切り出す
            new_weight = old_weight[:, :target_in_channels]
            state_dict[key] = new_weight
            print(
                f"  Adapted '{key}': {old_weight.shape} -> {new_weight.shape} "
                f"(truncated)"
            )

    # 対応するバイアスキーは形状が変わらないのでそのまま
    return state_dict


def _find_input_weight_keys(state_dict, model_name):
    """モデルの入力層にあたる重みキーを特定する。

    入力チャンネル数に依存する全てのConv層のweightキーを返す。
    UNETR: patch_embedding + encoder1 (raw input処理パス)
    SwinUNETR: patch_embed.proj + encoder1 (raw input処理パス)
    VNet: in_tr のConv層

    Args:
        state_dict: モデルのstate_dict
        model_name: モデル名

    Returns:
        入力層の重みキーのリスト
    """
    input_keys = []

    if model_name == "vnet":
        # VNetの入力層: in_tr.conv_block.conv.weight
        for key in state_dict:
            if "in_tr" in key and "weight" in key and "conv" in key:
                input_keys.append(key)
                break
    elif model_name == "unetr":
        # UNETRの入力層:
        #   - vit.patch_embedding.patch_embeddings.weight (ViTパッチ埋め込み)
        #   - encoder1.layer.conv*.conv.weight (raw input処理パス)
        for key in state_dict:
            if "patch_embedding" in key and "weight" in key:
                input_keys.append(key)
            elif key.startswith("encoder1.") and "conv" in key and "weight" in key:
                input_keys.append(key)
    elif model_name == "swin_unetr":
        # SwinUNETRの入力層:
        #   - swinViT.patch_embed.proj.weight
        #   - encoder1 のConv層 (raw input処理パス、存在する場合)
        for key in state_dict:
            if "patch_embed" in key and "proj" in key and "weight" in key:
                input_keys.append(key)
            elif key.startswith("encoder1.") and "conv" in key and "weight" in key:
                input_keys.append(key)

    # フォールバック: キー名から入力層を推定
    if not input_keys:
        for key in state_dict:
            w = state_dict[key]
            if w.dim() >= 4 and "weight" in key:
                # 最初に見つかるConv層を入力層と仮定
                input_keys.append(key)
                break

    return input_keys


def create_model(
    model_name,
    grid_size,
    out_channel,
    feature_size,
    pretrained_path=None,
    pretraining_out_channel=14,
    pretraining_in_channels=None,
    use_checkpoint=False,
    multi_task=False,
    task_out_channels=None,
    in_channels=1,
):
    """Create segmentation model.

    Args:
        model_name: Name of the model (vnet, unetr, swin_unetr)
        grid_size: Input grid size
        out_channel: Number of output channels (for single-task)
        feature_size: Feature size for UNETR/SwinUNETR
        pretrained_path: Path to pretrained weights
        pretraining_out_channel: Output channels of pretrained model
        pretraining_in_channels: Input channels of pretrained model.
            If different from in_channels, input layer weights will be adapted
            (e.g., 1ch weights copied to each channel of multi-channel model).
            If None, defaults to in_channels (no adaptation).
        use_checkpoint: Enable gradient checkpointing for SwinUNETR
        multi_task: If True, create multi-task model
        task_out_channels: Dict of task -> num_classes for multi-task mode
            e.g., {"shape": 5, "displacement": 3, "mapper": 3}
        in_channels: Number of input channels (default: 1 for single modality)
    """
    # マルチタスクモードのバリデーション
    if multi_task:
        if model_name == "vnet":
            raise ValueError(
                "Multi-task learning is not supported for VNet. "
                "Please use 'unetr' or 'swin_unetr'."
            )
        if task_out_channels is None:
            raise ValueError("task_out_channels must be provided for multi-task mode.")

    # 入力チャンネルの適合が必要か判定
    _pretraining_in_ch = (
        pretraining_in_channels if pretraining_in_channels is not None else in_channels
    )
    need_input_adapt = pretrained_path and (_pretraining_in_ch != in_channels)
    if need_input_adapt:
        print(
            f"Input channel adaptation: pretrained={_pretraining_in_ch}ch -> target={in_channels}ch"
        )

    if model_name == "vnet":
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            if need_input_adapt:
                weights = adapt_input_channel_weights(
                    weights, _pretraining_in_ch, in_channels, model_name
                )
            model = VNet(
                in_channels=in_channels,
                out_channels=pretraining_out_channel,
                spatial_dims=3,
            )
            model.load_state_dict(weights)
            model.out_tr = OutputTransition(
                3, 32, out_channel, ("elu", {"inplace": True}), False
            )
            print(f"Model {model_name} loaded from {pretrained_path}")
        else:
            model = VNet(
                in_channels=in_channels,
                out_channels=out_channel,
                spatial_dims=3,
            )

    elif model_name == "unetr":
        fs = feature_size or 16
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            if need_input_adapt:
                weights = adapt_input_channel_weights(
                    weights, _pretraining_in_ch, in_channels, model_name
                )
            model = UNETR(
                in_channels=in_channels,
                out_channels=pretraining_out_channel,
                img_size=grid_size,
                spatial_dims=3,
                feature_size=fs,
            )
            model.load_state_dict(weights)
            if not multi_task:
                model.out = UnetOutBlock(
                    spatial_dims=3,
                    in_channels=fs,
                    out_channels=out_channel,
                )
            print(f"Model {model_name} loaded from {pretrained_path}")
        else:
            model = UNETR(
                in_channels=in_channels,
                out_channels=out_channel,
                spatial_dims=3,
                feature_size=fs,
            )
        # マルチタスクモードの場合、ラッパーで包む
        if multi_task:
            model = MultiHeadSegmentationModel(
                base_model=model,
                task_out_channels=task_out_channels,
                feature_size=fs,
            )
            print(
                f"Multi-task model created with tasks: {list(task_out_channels.keys())}"
            )
    elif model_name == "swin_unetr":
        fs = feature_size or 48
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            if need_input_adapt:
                weights = adapt_input_channel_weights(
                    weights, _pretraining_in_ch, in_channels, model_name
                )
            model = SwinUNETR(
                in_channels=in_channels,
                out_channels=pretraining_out_channel,
                spatial_dims=3,
                feature_size=fs,
                use_checkpoint=use_checkpoint,
            )
            model.load_state_dict(weights)
            if not multi_task:
                model.out = UnetOutBlock(
                    spatial_dims=3,
                    in_channels=fs,
                    out_channels=out_channel,
                )
            print(f"Model {model_name} loaded from {pretrained_path}")
            if use_checkpoint:
                print("Gradient checkpointing enabled for SwinUNETR")
        else:
            model = SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channel,
                spatial_dims=3,
                feature_size=fs,
                use_checkpoint=use_checkpoint,
            )
            if use_checkpoint:
                print("Gradient checkpointing enabled for SwinUNETR")
        # マルチタスクモードの場合、ラッパーで包む
        if multi_task:
            model = MultiHeadSegmentationModel(
                base_model=model,
                task_out_channels=task_out_channels,
                feature_size=fs,
            )
            print(
                f"Multi-task model created with tasks: {list(task_out_channels.keys())}"
            )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = torch.nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
    return model


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def validation(
    epoch_iterator_val,
    global_step,
    training_log_path,
    out_channel=14,
    use_ce_loss=False,
):
    model.eval()
    run_acc = AverageMeter()
    raw_dice_scores = []
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.autocast("cuda"):
                val_outputs = sliding_window_inference(val_inputs, grid_size, 4, model)
            val_labels_list = decollate_batch(val_labels)

            if use_ce_loss:
                # For CE loss, labels are in (B, 1, D, H, W) format with integer values
                # Convert to one-hot for Dice metric calculation only
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
            else:
                # For DiceCE loss, labels need to be one-hot
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]

            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric.reset()
            raw_dice_score = dice_metric(
                y_pred=val_output_convert, y=val_labels_convert
            )
            # Move to CPU before appending to avoid GPU memory accumulation
            raw_dice_scores.append(raw_dice_score[0].cpu())
            dice_scores, not_nans = dice_metric.aggregate()
            run_acc.update(dice_scores.cpu().numpy(), not_nans.cpu().numpy())

            # Explicitly delete variables to free GPU memory
            del val_inputs, val_labels, val_outputs
            del (
                val_labels_list,
                val_labels_convert,
                val_outputs_list,
                val_output_convert,
            )
            del raw_dice_score, dice_scores, not_nans
            torch.cuda.empty_cache()

            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )  # noqa: B038

        mean_dice_val = np.mean(run_acc.avg).item()
        class_dice_score = (
            torch.stack(raw_dice_scores, dim=0).nan_to_num().mean(dim=0).numpy()
        )  # Already on CPU

        # Log evaluation results
        with open(training_log_path, "a") as f:
            f.write(f"Step {global_step}: Validation Dice Score: {mean_dice_val:.6f}\n")
            # Log per-class dice scores
            # This assumes out_channel is the number of classes
            # and the score for background is not included
            # This is why we loop from 0 to out_channel - 1
            for class_idx in range(out_channel - 1):
                f.write(
                    f"Step {global_step}: Class {class_idx} Dice Score: {class_dice_score[class_idx].item():.6f}\n"
                )

        # Clear the dice scores list
        del raw_dice_scores
        torch.cuda.empty_cache()

    return mean_dice_val, class_dice_score


def validation_multi_task(
    epoch_iterator_val,
    global_step,
    training_log_path,
    task_out_channels,
    task_names,
    use_ce_loss=False,
):
    """マルチタスク学習用のバリデーション関数

    Args:
        epoch_iterator_val: Validation data iterator
        global_step: Current training step
        training_log_path: Path to training log file
        task_out_channels: Dict of task -> num_classes
        task_names: List of task names in order
        use_ce_loss: If True, use CE loss (not used in validation, for compatibility)

    Returns:
        mean_dice_val: Mean Dice score across all tasks
        task_dice_scores: Dict of task -> mean Dice score
    """
    model.eval()

    # タスクごとのメトリクス
    task_run_acc = {task: AverageMeter() for task in task_out_channels}
    task_raw_dice_scores = {task: [] for task in task_out_channels}

    # タスクごとのpost_label, post_pred, dice_metric
    task_post_label = {
        task: AsDiscrete(to_onehot=nc) for task, nc in task_out_channels.items()
    }
    task_post_pred = {
        task: AsDiscrete(argmax=True, to_onehot=nc)
        for task, nc in task_out_channels.items()
    }
    task_dice_metric = {
        task: DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
        for task in task_out_channels
    }

    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs = batch["image"].cuda()
            val_labels = batch["label"].cuda()

            with torch.autocast("cuda"):
                # マルチタスクモデルはdictを返す
                val_outputs = sliding_window_inference(val_inputs, grid_size, 4, model)

            # マルチタスクラベルを分離
            val_labels_dict = split_multi_task_labels(val_labels, task_names)

            # タスクごとにDiceスコアを計算
            for task in task_out_channels:
                task_labels = val_labels_dict[task]
                task_outputs = val_outputs[task]

                task_labels_list = decollate_batch(task_labels)
                task_labels_convert = [
                    task_post_label[task](lbl) for lbl in task_labels_list
                ]

                task_outputs_list = decollate_batch(task_outputs)
                task_output_convert = [
                    task_post_pred[task](pred) for pred in task_outputs_list
                ]

                task_dice_metric[task].reset()
                raw_dice_score = task_dice_metric[task](
                    y_pred=task_output_convert, y=task_labels_convert
                )
                task_raw_dice_scores[task].append(raw_dice_score[0].cpu())
                dice_scores, not_nans = task_dice_metric[task].aggregate()
                task_run_acc[task].update(
                    dice_scores.cpu().numpy(), not_nans.cpu().numpy()
                )

            # メモリ解放
            del val_inputs, val_labels, val_outputs, val_labels_dict
            torch.cuda.empty_cache()

            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )

    # タスクごとの平均Diceスコアを計算
    task_mean_dice = {}
    for task in task_out_channels:
        task_mean_dice[task] = np.mean(task_run_acc[task].avg).item()

    # 全タスクの平均Diceスコア
    mean_dice_val = np.mean(list(task_mean_dice.values()))

    # ログ出力
    with open(training_log_path, "a") as f:
        f.write(
            f"Step {global_step}: Validation Mean Dice Score: {mean_dice_val:.6f}\n"
        )
        for task, dice in task_mean_dice.items():
            f.write(f"Step {global_step}: Task '{task}' Dice Score: {dice:.6f}\n")

    # メモリ解放
    del task_raw_dice_scores
    torch.cuda.empty_cache()

    return mean_dice_val, task_mean_dice


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    scaler,
    global_step,
    dice_val_best,
    global_step_best,
    epoch_loss_values,
    metric_values,
):
    """Save training checkpoint."""
    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "global_step": global_step,
        "dice_val_best": dice_val_best,
        "global_step_best": global_step_best,
        "epoch_loss_values": epoch_loss_values,
        "metric_values": metric_values,
    }
    if torch.cuda.device_count() > 1:
        checkpoint["model_state_dict"] = model.module.state_dict()
    else:
        checkpoint["model_state_dict"] = model.state_dict()
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {global_step}: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    """Load training checkpoint and return training state."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0, 0, [], []

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if torch.cuda.device_count() > 1:
        model = model.module  # Unwrap DataParallel
        model.load_state_dict(checkpoint["model_state_dict"])
        model = torch.nn.DataParallel(model)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    global_step = checkpoint["global_step"]
    dice_val_best = checkpoint["dice_val_best"]
    global_step_best = checkpoint["global_step_best"]
    epoch_loss_values = checkpoint["epoch_loss_values"]
    metric_values = checkpoint["metric_values"]

    print(f"Resumed training from step {global_step}, best dice: {dice_val_best:.4f}")
    return (
        global_step,
        dice_val_best,
        global_step_best,
        epoch_loss_values,
        metric_values,
    )


def train(
    global_step,
    train_loader,
    dice_val_best,
    global_step_best,
    training_log_path,
    out_channel=14,
    is_real_data=True,
    use_ce_loss=False,
    multi_task=False,
    task_out_channels=None,
    task_names=None,
    gradient_accumulation_steps=1,
):
    """Training function for both single-task and multi-task learning.

    Args:
        global_step: Current training step
        train_loader: Training data loader
        dice_val_best: Best Dice score so far
        global_step_best: Step at which best Dice was achieved
        training_log_path: Path to training log file
        out_channel: Number of output channels (for single-task)
        is_real_data: Whether using real data (unused, kept for compatibility)
        use_ce_loss: If True, use CrossEntropyLoss instead of DiceCELoss
        multi_task: If True, enable multi-task learning mode
        task_out_channels: Dict of task -> num_classes (required for multi-task)
        task_names: List of task names in order (required for multi-task)
        gradient_accumulation_steps: Number of steps to accumulate gradients before updating parameters
    """
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())

        # マルチタスク時はラベルを分離
        if multi_task:
            y_dict = split_multi_task_labels(y, task_names)

        # Clear gradients at the start of accumulation cycle
        if (step - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        with torch.autocast("cuda"):
            logit_map = model(x)
            if multi_task:
                # マルチタスク損失を計算
                loss = loss_function(logit_map, y_dict)
            elif use_ce_loss:
                # CrossEntropyLoss expects labels in (B, D, H, W) format (no channel dimension)
                # and logits in (B, C, D, H, W) format
                y_squeezed = y.squeeze(
                    1
                ).long()  # Remove channel dimension and convert to long
                loss = loss_function(logit_map, y_squeezed)
            else:
                # DiceCELoss handles one-hot encoding internally
                loss = loss_function(logit_map, y)

            # Normalize loss by accumulation steps for proper gradient averaging
            loss = loss / gradient_accumulation_steps

        # Store loss value before cleanup (multiply back for logging actual loss)
        loss_value = loss.item() * gradient_accumulation_steps

        scaler.scale(loss).backward()
        epoch_loss += loss_value

        # Update parameters only at accumulation boundaries
        if step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # Explicitly delete variables to free GPU memory immediately
        if multi_task:
            del x, y, y_dict, logit_map, loss
        else:
            del x, y, logit_map, loss

        # Force garbage collection and cache clearing more frequently
        if step % 10 == 0:
            torch.cuda.empty_cache()

        epoch_iterator.set_description(  # noqa: B038
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss_value:2.5f})"
        )

        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            # Clear memory before validation
            torch.cuda.empty_cache()

            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            if multi_task:
                dice_val, dice_scores = validation_multi_task(
                    epoch_iterator_val,
                    global_step,
                    training_log_path,
                    task_out_channels,
                    task_names,
                    use_ce_loss,
                )
            else:
                dice_val, dice_scores = validation(
                    epoch_iterator_val,
                    global_step,
                    training_log_path,
                    out_channel,
                    use_ce_loss,
                )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            # Log training results
            with open(training_log_path, "a") as f:
                f.write(f"Step {global_step}: Training Loss: {epoch_loss:.6f}\n")

            # Save checkpoint after each evaluation (overwrite previous)
            checkpoint_path = os.path.join(out_dir, "training_checkpoint.pth")
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                scaler,
                global_step,
                dice_val_best,
                global_step_best,
                epoch_loss_values,
                metric_values,
            )

            # Save latest model after each validation
            last_model_path = os.path.join(out_dir, "last_model.pth")
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), last_model_path)
            else:
                torch.save(model.state_dict(), last_model_path)
            print(f"Latest model saved at step {global_step}: {last_model_path}")

            # Calculate steps correctly: [eval_num, eval_num*2, eval_num*3, ...]
            # This represents the actual global_step values when validation occurred
            # Convert to numpy array for compatibility with plot_metrics
            steps_for_plot = np.array(
                [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
            )
            plot_metrics(
                epoch_loss_values,
                metric_values,
                steps_for_plot,
                out_dir,
            )

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                if torch.cuda.device_count() > 1:
                    torch.save(
                        model.module.state_dict(),
                        os.path.join(out_dir, "best_metric_model.pth"),
                    )
                else:
                    torch.save(
                        model.state_dict(),
                        os.path.join(out_dir, "best_metric_model.pth"),
                    )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # Log detailed per-class/per-task dice scores for the best model
                with open(training_log_path, "a") as f:
                    f.write(f"*** BEST MODEL SAVED at Step {global_step} ***\n")
                    f.write(f"Best Average Dice Score: {dice_val_best:.6f}\n")
                    if multi_task:
                        f.write("Per-task Dice Scores for Best Model:\n")
                        for task, dice in dice_scores.items():
                            f.write(f"  Task '{task}': {dice:.6f}\n")
                    else:
                        f.write("Per-class Dice Scores for Best Model:\n")
                        for class_idx in range(out_channel - 1):
                            f.write(
                                f"  Class {class_idx}: {dice_scores[class_idx].item():.6f}\n"
                            )
                    f.write("=" * 50 + "\n")

            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1

        # Periodic memory cleanup
        if global_step % 100 == 0:
            torch.cuda.empty_cache()

    return global_step, dice_val_best, global_step_best


def perform_inference_and_visualize(
    model, val_loader, out_dir, device, grid_size, out_channel
):
    """Perform inference on validation data and create visualizations."""
    model.eval()

    # Get one validation sample
    val_batch = next(iter(val_loader))
    val_inputs = val_batch["image"].to(device)
    val_labels = val_batch["label"].to(device)

    with torch.no_grad():
        # Perform inference
        with torch.autocast("cuda"):
            val_outputs = sliding_window_inference(val_inputs, grid_size, 4, model)

        # Convert to predictions
        val_outputs_softmax = torch.softmax(val_outputs, 1)
        val_predictions = torch.argmax(val_outputs_softmax, dim=1)

        # Calculate Dice score for this sample
        from monai.metrics import compute_dice

        dice_scores = compute_dice(
            val_outputs_softmax, val_labels, include_background=False
        )
        mean_dice = torch.mean(dice_scores).item()

        print(f"Sample Dice Score: {mean_dice:.4f}")

        # Move to CPU and convert to numpy
        image = val_inputs[0, 0].cpu().numpy()  # First channel, first batch
        label = val_labels[0, 0].cpu().numpy()  # First channel, first batch
        prediction = val_predictions[0].cpu().numpy()  # First batch

        # Explicitly delete GPU tensors to free memory
        del (
            val_inputs,
            val_labels,
            val_outputs,
            val_outputs_softmax,
            val_predictions,
            dice_scores,
        )
        torch.cuda.empty_cache()

        # Create visualization
        create_slice_visualization(
            image, label, prediction, out_dir, mean_dice, out_channel
        )


def perform_inference_and_visualize_multi_task(
    model, val_loader, out_dir, device, grid_size, task_out_channels, task_names
):
    """マルチタスクモデルの推論と可視化を実行

    Args:
        model: Multi-task segmentation model
        val_loader: Validation data loader
        out_dir: Output directory for visualizations
        device: Device to use
        grid_size: Grid size for sliding window inference
        task_out_channels: Dict of task -> num_classes
        task_names: List of task names in order
    """
    model.eval()

    # Get one validation sample
    val_batch = next(iter(val_loader))
    val_inputs = val_batch["image"].to(device)
    val_labels = val_batch["label"].to(device)

    with torch.no_grad():
        # Perform inference
        with torch.autocast("cuda"):
            val_outputs = sliding_window_inference(val_inputs, grid_size, 4, model)

        # マルチタスクラベルを分離
        val_labels_dict = split_multi_task_labels(val_labels, task_names)

        # Move image to CPU
        image = val_inputs[0, 0].cpu().numpy()

        # タスクごとに可視化
        task_predictions = {}
        task_dice_scores = {}
        for task in task_out_channels:
            task_output = val_outputs[task]
            task_label = val_labels_dict[task]

            # Convert to predictions
            task_output_softmax = torch.softmax(task_output, 1)
            task_pred = torch.argmax(task_output_softmax, dim=1)

            # Calculate Dice score for this task
            from monai.metrics import compute_dice

            dice_scores = compute_dice(
                task_output_softmax, task_label, include_background=False
            )
            mean_dice = torch.mean(dice_scores).item()

            task_predictions[task] = task_pred[0].cpu().numpy()
            task_dice_scores[task] = mean_dice

            # Get label for this task
            label = task_label[0, 0].cpu().numpy()

            # Create visualization for this task
            create_slice_visualization_multi_task(
                image,
                label,
                task_predictions[task],
                out_dir,
                task,
                mean_dice,
                task_out_channels[task],
            )

        # Print summary
        print("Multi-task Inference Results:")
        for task, dice in task_dice_scores.items():
            print(f"  Task '{task}': Dice Score = {dice:.4f}")

        # Explicitly delete GPU tensors to free memory
        del val_inputs, val_labels, val_outputs, val_labels_dict
        torch.cuda.empty_cache()


def create_slice_visualization_multi_task(
    image, label, prediction, out_dir, task_name, dice_score=None, out_channel=14
):
    """マルチタスク用のスライス可視化を作成

    Args:
        image: Input image volume
        label: Ground truth label volume
        prediction: Predicted label volume
        out_dir: Output directory
        task_name: Name of the task (shape, displacement, mapper)
        dice_score: Dice score for this task
        out_channel: Number of output channels for this task
    """
    # Get middle slices for visualization
    depth = image.shape[2]
    middle_slice = depth // 2

    # Get slices
    image_slice = image[:, :, middle_slice]
    label_slice = label[:, :, middle_slice]
    pred_slice = prediction[:, :, middle_slice]

    # Use fixed value range for consistent color mapping
    vmin = 0
    vmax = out_channel - 1

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    title = f"Task '{task_name}' - Middle Slice"
    if dice_score is not None:
        title += f" (Dice: {dice_score:.4f})"
    fig.suptitle(title, fontsize=16)

    # Original image
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth label with consistent color range
    im1 = axes[1].imshow(label_slice, cmap="jet", alpha=0.8, vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Prediction with consistent color range
    im2 = axes[2].imshow(pred_slice, cmap="jet", alpha=0.8, vmin=vmin, vmax=vmax)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Overlay: Image + Prediction with consistent color range
    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(pred_slice, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    axes[3].set_title("Image + Prediction")
    axes[3].axis("off")

    plt.tight_layout()

    # Save the visualization
    viz_path = os.path.join(out_dir, f"inference_visualization_{task_name}.png")
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Task '{task_name}' visualization saved to: {viz_path}")


def create_slice_visualization(
    image, label, prediction, out_dir, dice_score=None, out_channel=14
):
    """Create and save slice visualizations comparing predictions and labels."""
    # Get middle slices for visualization
    depth = image.shape[2]
    middle_slice = depth // 2

    # Get slices
    image_slice = image[:, :, middle_slice]
    label_slice = label[:, :, middle_slice]
    pred_slice = prediction[:, :, middle_slice]

    # Use fixed value range for consistent color mapping across all data
    # This ensures 0 is always black, and higher values have consistent colors
    vmin = 0
    vmax = out_channel - 1  # Maximum possible class index

    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    title = "Inference Results - Middle Slice"
    if dice_score is not None:
        title += f" (Dice: {dice_score:.4f})"
    fig.suptitle(title, fontsize=16)

    # Original image
    axes[0].imshow(image_slice, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth label with consistent color range
    im1 = axes[1].imshow(label_slice, cmap="jet", alpha=0.8, vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth Label")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Prediction with consistent color range
    im2 = axes[2].imshow(pred_slice, cmap="jet", alpha=0.8, vmin=vmin, vmax=vmax)
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Overlay: Image + Prediction with consistent color range
    axes[3].imshow(image_slice, cmap="gray")
    axes[3].imshow(pred_slice, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    axes[3].set_title("Image + Prediction Overlay")
    axes[3].axis("off")

    plt.tight_layout()

    # Save the visualization
    viz_path = os.path.join(out_dir, "inference_visualization.png")
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Inference visualization saved to: {viz_path}")

    # Also create axial, coronal, and sagittal views
    create_multi_plane_visualization(
        image, label, prediction, out_dir, dice_score, out_channel
    )


def create_multi_plane_visualization(
    image, label, prediction, out_dir, dice_score=None, out_channel=14
):
    """Create visualizations across different anatomical planes."""
    # Get middle slices for each plane
    height, width, depth = image.shape

    # Axial (xy plane)
    axial_slice = depth // 2
    image_axial = image[:, :, axial_slice]
    label_axial = label[:, :, axial_slice]
    pred_axial = prediction[:, :, axial_slice]

    # Coronal (xz plane)
    coronal_slice = width // 2
    image_coronal = image[:, coronal_slice, :]
    label_coronal = label[:, coronal_slice, :]
    pred_coronal = prediction[:, coronal_slice, :]

    # Sagittal (yz plane)
    sagittal_slice = height // 2
    image_sagittal = image[sagittal_slice, :, :]
    label_sagittal = label[sagittal_slice, :, :]
    pred_sagittal = prediction[sagittal_slice, :, :]

    # Use fixed value range for consistent color mapping across all data
    # This ensures 0 is always black, and higher values have consistent colors
    vmin = 0
    vmax = out_channel - 1  # Maximum possible class index

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    title = "Multi-Plane Inference Results"
    if dice_score is not None:
        title += f" (Dice: {dice_score:.4f})"
    fig.suptitle(title, fontsize=16)

    planes = [
        ("Axial", image_axial, label_axial, pred_axial),
        ("Coronal", image_coronal, label_coronal, pred_coronal),
        ("Sagittal", image_sagittal, label_sagittal, pred_sagittal),
    ]

    for i, (plane_name, img, lbl, pred) in enumerate(planes):
        # Original image
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"{plane_name} - Image")
        axes[i, 0].axis("off")

        # Ground truth with consistent color range
        im1 = axes[i, 1].imshow(lbl, cmap="jet", alpha=0.8, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{plane_name} - Ground Truth")
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Prediction with consistent color range
        im2 = axes[i, 2].imshow(pred, cmap="jet", alpha=0.8, vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"{plane_name} - Prediction")
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Overlay with consistent color range
        axes[i, 3].imshow(img, cmap="gray")
        axes[i, 3].imshow(pred, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[i, 3].set_title(f"{plane_name} - Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()

    # Save the multi-plane visualization
    multiplane_path = os.path.join(out_dir, "inference_multiplane_visualization.png")
    plt.savefig(multiplane_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Multi-plane visualization saved to: {multiplane_path}")


if __name__ == "__main__":
    # Set PyTorch CUDA memory management settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_json_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file",
    )
    p.add_argument("--is_real_data", action="store_true", help="Use real data")
    p.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=[
            "vnet",
            "unetr",
            "swin_unetr",
        ],
        help="Name of the model to be trained",
    )
    p.add_argument("--pretrained_model", type=str, help="Path to the pretrained model")
    p.add_argument(
        "--pretraining_out_channel",
        type=int,
        default=14,
        help="Output channel size for pretrained model",
    )
    p.add_argument(
        "--grid_size",
        type=int,
        nargs="+",
        default=[96, 96, 96],
        help="Grid size for the input data",
    )
    p.add_argument("--out_channel", type=int, default=14, help="Output channel size")
    p.add_argument("--feature_size", type=int, help="Feature size for SwinUNETR")
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training and validation",
    )
    p.add_argument(
        "--max_iterations", type=int, default=30000, help="Maximum training iterations"
    )
    p.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    p.add_argument("--out_dir", type=str, help="Output directory")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint file to resume training from",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If not specified, a random seed will be generated.",
    )
    p.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Enable gradient checkpointing for SwinUNETR to reduce memory usage",
    )
    p.add_argument(
        "--use_ce_loss",
        action="store_true",
        help="Use CrossEntropyLoss instead of DiceCELoss to reduce memory usage",
    )
    p.add_argument(
        "--multi_task",
        action="store_true",
        help="Enable multi-task learning mode. Requires dataset with multi-task labels "
        "(data.json with 'multi_task': true). Only supported for UNETR and SwinUNETR.",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps. Effective batch size = batch_size * gradient_accumulation_steps. "
        "For example, batch_size=4 with gradient_accumulation_steps=2 equals batch_size=8 without accumulation.",
    )
    p.add_argument(
        "--in_channels",
        type=int,
        default=1,
        help="Number of input channels (modalities). "
        "Use 1 for single modality (e.g., CT), 2 for dual modality (e.g., DWI+ADC from ISLES). "
        "Input images must be 4D NIfTI files with shape (C, D, H, W) where C = in_channels.",
    )
    p.add_argument(
        "--pretraining_in_channels",
        type=int,
        default=None,
        help="Number of input channels of the pretrained model. "
        "When the pretrained model was trained with 1 channel and the target model uses "
        "multiple channels (--in_channels > 1), the input layer weights are copied to each "
        "channel and scaled by 1/in_channels. If not specified, defaults to --in_channels.",
    )
    args = p.parse_args()

    # マルチタスクモードのバリデーション
    if args.multi_task and args.model_name == "vnet":
        print(
            "Error: --multi_task is not supported for VNet. Please use 'unetr' or 'swin_unetr'."
        )
        exit(1)

    # Generate random seed if not specified
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        print(f"No seed specified. Generated random seed: {args.seed}")

    # Set random seed for reproducibility
    set_seed(args.seed)

    pretraining_state = "fine_tuning"
    if not args.pretrained_model:
        pretraining_state = "training_from_scratch"
    if not args.out_dir:
        args.out_dir = os.path.join(
            "training_output",
            args.model_name,
            pretraining_state,
            time.strftime("%Y%m%d_%H%M%S"),
        )
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    training_log_path = os.path.join(out_dir, "training_log.txt")
    with open(training_log_path, "w") as f:
        f.write(str(args) + "\n")
        f.write(f"Random seed: {args.seed}\n")
        if args.gradient_accumulation_steps > 1:
            effective_batch_size = args.batch_size * args.gradient_accumulation_steps
            f.write(
                f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n"
            )
            f.write(f"Effective batch size: {effective_batch_size}\n")
        f.write("=" * 50 + "\n")
    print(f"Training log will be saved to {training_log_path}")
    print(f"Output directory: {out_dir}")
    print(f"Random seed set to: {args.seed}")

    # マルチタスク情報の読み込み
    multi_task_info = None
    task_out_channels = None
    task_names = None
    if args.multi_task:
        multi_task_info = load_multi_task_info(args.data_json_path)
        if not multi_task_info["multi_task"]:
            print("Error: --multi_task specified but dataset is not multi-task format.")
            print(
                "Please generate dataset with --multi_task flag in generate_sdf_dataset.py"
            )
            exit(1)
        task_out_channels = multi_task_info["task_out_channels"]
        task_names = multi_task_info["task_names"]
        print(f"Multi-task mode enabled with {len(task_names)} tasks: {task_names}")
        for task, nc in task_out_channels.items():
            print(f"  Task '{task}': {nc} classes")
        with open(training_log_path, "a") as f:
            f.write(f"Multi-task mode: {args.multi_task}\n")
            f.write(f"Task names: {task_names}\n")
            f.write(f"Task output channels: {task_out_channels}\n")
            f.write("=" * 50 + "\n")

    grid_size = tuple(args.grid_size)
    train_loader, val_loader = make_data_loder(
        data_json_path=args.data_json_path,
        real_data=args.is_real_data,
        spatial_size=grid_size,
        batch_size=args.batch_size,
        multi_task=args.multi_task,
    )
    print(f"Training data loader created with {len(train_loader.dataset)} samples.")
    print(f"Validation data loader created with {len(val_loader.dataset)} samples.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name=args.model_name,
        grid_size=grid_size,
        out_channel=args.out_channel,
        feature_size=args.feature_size,
        pretrained_path=args.pretrained_model,
        pretraining_out_channel=args.pretraining_out_channel,
        pretraining_in_channels=args.pretraining_in_channels,
        use_checkpoint=args.use_checkpoint,
        multi_task=args.multi_task,
        task_out_channels=task_out_channels,
        in_channels=args.in_channels,
    )
    if args.multi_task:
        print(f"Multi-task model {args.model_name} created.")
    else:
        print(
            f"Model {args.model_name} created with output channels: {args.out_channel}."
        )
    print(f"Using learning rate: {args.learning_rate}")
    if args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
    else:
        print("Training from scratch, no pretrained model loaded.")
    model = model.to(device)

    # Count and log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    with open(training_log_path, "a") as f:
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write("=" * 50 + "\n")

    # Set up loss function based on user choice
    if args.multi_task:
        loss_function = MultiTaskLoss(
            task_out_channels=task_out_channels,
            use_ce_loss=args.use_ce_loss,
        )
        print(f"Using MultiTaskLoss (CE: {args.use_ce_loss})")
    elif args.use_ce_loss:
        loss_function = CrossEntropyLoss()
        print("Using CrossEntropyLoss (memory efficient)")
    else:
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        print("Using DiceCELoss")

    optimizer = torch.optim.AdamW(
        model.parameters(), args.learning_rate, weight_decay=1e-5
    )
    scaler = torch.GradScaler("cuda")
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_steps=500,
        max_steps=args.max_iterations,
    )
    max_iterations = args.max_iterations
    eval_num = 500
    post_label = AsDiscrete(to_onehot=args.out_channel)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channel)
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=True
    )
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    step_values = []  # Track steps for plotting

    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        (
            global_step,
            dice_val_best,
            global_step_best,
            epoch_loss_values,
            metric_values,
        ) = load_checkpoint(
            args.resume_from_checkpoint, model, optimizer, scheduler, scaler
        )
        with open(training_log_path, "a") as f:
            f.write(
                f"Resumed training from checkpoint: {args.resume_from_checkpoint}\n"
            )
            f.write(
                f"Resuming from step {global_step}, best dice: {dice_val_best:.4f}\n"
            )

    print(f"Starting training with learning rate: {args.learning_rate}")
    print("Starting training...")
    if args.gradient_accumulation_steps > 1:
        effective_batch_size = args.batch_size * args.gradient_accumulation_steps
        print(
            f"Gradient accumulation enabled: {args.gradient_accumulation_steps} steps"
        )
        print(
            f"Effective batch size: {effective_batch_size} (batch_size={args.batch_size} × accumulation={args.gradient_accumulation_steps})"
        )
    time_start = time.time()
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step,
            train_loader,
            dice_val_best,
            global_step_best,
            training_log_path,
            args.out_channel,
            args.is_real_data,
            args.use_ce_loss,
            multi_task=args.multi_task,
            task_out_channels=task_out_channels,
            task_names=task_names,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    time_end = time.time()
    print(
        f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}."
    )

    # Save training metrics for visualization
    # Calculate steps correctly: [eval_num, eval_num*2, eval_num*3, ...]
    # Convert to numpy array for consistency
    steps_array = np.array([eval_num * (i + 1) for i in range(len(epoch_loss_values))])
    metrics_data = {
        "training_loss": epoch_loss_values,
        "validation_dice": metric_values,
        "steps": steps_array.tolist(),  # Convert to list for JSON serialization
    }

    # Save as numpy arrays
    np.save(os.path.join(out_dir, "training_loss.npy"), np.array(epoch_loss_values))
    np.save(os.path.join(out_dir, "validation_dice.npy"), np.array(metric_values))
    np.save(os.path.join(out_dir, "steps.npy"), np.array(metrics_data["steps"]))

    # Save as JSON for easy reading
    with open(os.path.join(out_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)

    # Create training curves visualization using visualize_training_metrics.py functions
    print("Creating training curves visualization...")
    print_summary(epoch_loss_values, metric_values, steps_array)

    # Use the plot_metrics function from visualize_training_metrics.py
    # This will create and save the visualization plots without showing them
    plot_metrics(epoch_loss_values, metric_values, steps_array, out_dir)

    with open(training_log_path, "a") as f:
        f.write(
            f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}.\n"
        )
        f.write("Model files saved:\n")
        f.write("  - best_metric_model.pth (best performing model)\n")
        f.write("  - last_model.pth (latest model from final validation)\n")
        f.write("Training checkpoint saved to: training_checkpoint.pth\n")
        f.write("Training metrics saved to:\n")
        f.write("  - training_loss.npy\n")
        f.write("  - validation_dice.npy\n")
        f.write("  - steps.npy\n")
        f.write("  - training_metrics.json\n")
        f.write("Training curves visualizations saved to:\n")
        f.write("  - training_metrics_plot.png\n")
        f.write("  - training_loss_individual.png\n")
        f.write("  - validation_dice_individual.png\n")

    # Perform inference and visualize results
    print("Performing inference with best model...")
    # Load the best model
    best_model_path = os.path.join(out_dir, "best_metric_model.pth")
    if os.path.exists(best_model_path):
        if torch.cuda.device_count() > 1:
            model = model.module  # Unwrap DataParallel
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            model = torch.nn.DataParallel(model)
        else:
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using current model for inference")

    if args.multi_task:
        perform_inference_and_visualize_multi_task(
            model, val_loader, out_dir, device, grid_size, task_out_channels, task_names
        )
        with open(training_log_path, "a") as f:
            f.write("Inference visualizations saved to:\n")
            for task in task_out_channels:
                f.write(f"  - inference_visualization_{task}.png\n")
            f.write(
                "All visualizations (training curves and inference) are now complete.\n"
            )
    else:
        perform_inference_and_visualize(
            model, val_loader, out_dir, device, grid_size, args.out_channel
        )
        with open(training_log_path, "a") as f:
            f.write("Inference visualizations saved to:\n")
            f.write("  - inference_visualization.png\n")
            f.write("  - inference_multiplane_visualization.png\n")
            f.write(
                "All visualizations (training curves and inference) are now complete.\n"
            )
