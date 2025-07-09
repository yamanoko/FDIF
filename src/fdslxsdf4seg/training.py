import argparse
import os
import time

import torch
from monai.data import (
    CacheDataset,
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
)
from tqdm import tqdm


def make_data_loder(
    data_json_path: str,
    real_data: bool = True,
    spatial_size: tuple = (96, 96, 96),
    batch_size: int = 1,
):
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
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
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
    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_num=6,
        cache_rate=1.0,
        num_workers=4,
    )
    val_loader = ThreadDataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader


def create_model(
    model_name,
    grid_size,
    out_channel,
    feature_size,
    pretrained_path=None,
    pretraining_out_channel=14,
):
    if model_name == "vnet":
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            model = VNet(
                in_channels=1,
                out_channels=pretraining_out_channel,
                spatial_dims=3,
            )
            model.load_state_dict(weights=weights)
            model.out_tr = OutputTransition(
                3, 32, out_channel, ("elu", {"inplace": True}), False
            )
            print(f"Model {model_name} loaded from {pretrained_path}")
        else:
            model = VNet(
                in_channels=1,
                out_channels=out_channel,
                spatial_dims=3,
            )

    elif model_name == "unetr":
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            model = UNETR(
                in_channels=1,
                out_channels=pretraining_out_channel,
                img_size=grid_size,
                spatial_dims=3,
                feature_size=feature_size or 16,
            )
            model.load_state_dict(weights=weights)
            model.out = UnetOutBlock(
                spatial_dims=3,
                in_channels=feature_size or 16,
                out_channels=out_channel,
            )
            print(f"Model {model_name} loaded from {pretrained_path}")
        else:
            model = UNETR(
                in_channels=1,
                out_channels=out_channel,
                img_size=grid_size,
                spatial_dims=3,
                feature_size=feature_size or 16,
            )
    elif model_name == "swin_unetr":
        model = SwinUNETR(
            in_channels=1,
            out_channels=out_channel,
            img_size=grid_size,
            spatial_dims=3,
            feature_size=feature_size or 48,
        )
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            model.load_from(weights=weights)
            print(f"Model {model_name} loaded from {pretrained_path}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            with torch.autocast("cuda"):
                val_outputs = sliding_window_inference(val_inputs, grid_size, 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.autocast("cuda"):
            logit_map = model(x)
            loss = loss_function(logit_map, y)
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(out_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


if __name__ == "__main__":
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
    p.add_argument("--out_dir", type=str, help="Output directory")
    args = p.parse_args()
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
    print(f"Training log will be saved to {training_log_path}")
    print(f"Output directory: {out_dir}")

    grid_size = tuple(args.grid_size)
    train_loader, val_loader = make_data_loder(
        data_json_path=args.data_json_path,
        real_data=args.is_real_data,
        spatial_size=grid_size,
        batch_size=args.batch_size,
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
        pretraining_out_channel=args.pretraing_out_channel,
    )
    print(f"Model {args.model_name} created with output channels: {args.out_channel}.")
    if args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
    else:
        print("Training from scratch, no pretrained model loaded.")
    model = model.to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-5)
    scaler = torch.GradScaler("cuda")
    max_iterations = args.max_iterations
    eval_num = 500
    post_label = AsDiscrete(to_onehot=args.out_channel)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channel)
    dice_metric = DiceMetric(
        include_background=True, reduction="mean", get_not_nans=False
    )
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    print("Starting training...")
    time_start = time.time()
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step, train_loader, dice_val_best, global_step_best
        )
    time_end = time.time()
    print(
        f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}."
    )
    with open(training_log_path, "a") as f:
        f.write(
            f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}.\n"
        )
