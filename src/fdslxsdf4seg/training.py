import argparse
import json
import os

# Import visualization functions from visualize_training_metrics.py
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from visualize_training_metrics import (
        plot_metrics,
        print_summary,
    )
except ImportError:
    try:
        from .visualize_training_metrics import (
            plot_metrics,
            print_summary,
        )
    except ImportError:
        try:
            from fdslxsdf4seg.visualize_training_metrics import (
                plot_metrics,
                print_summary,
            )
        except ImportError:
            print(
                "Warning: Could not import visualization functions. Training will continue without visualization plots."
            )
            # Define dummy functions to prevent errors

            def plot_metrics(*args, **kwargs):
                print("Visualization function not available.")

            def print_summary(*args, **kwargs):
                print("Summary function not available.")


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
            model.load_state_dict(weights)
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
            model.load_state_dict(weights)
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


def validation(epoch_iterator_val, global_step, training_log_path, out_channel=14):
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

        # Get per-class dice scores
        dice_scores = dice_metric.aggregate()
        mean_dice_val = torch.mean(dice_scores).item()
        dice_metric.reset()

        # Log evaluation results
        with open(training_log_path, "a") as f:
            f.write(f"Step {global_step}: Validation Dice Score: {mean_dice_val:.6f}\n")
            # Log per-class dice scores
            for class_idx in range(out_channel):
                f.write(
                    f"Step {global_step}: Class {class_idx} Dice Score: {dice_scores[class_idx].item():.6f}\n"
                )

    return mean_dice_val, dice_scores


def train(
    global_step,
    train_loader,
    dice_val_best,
    global_step_best,
    training_log_path,
    out_channel=14,
):
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
            dice_val, dice_scores = validation(
                epoch_iterator_val, global_step, training_log_path, out_channel
            )
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)

            # Log training results
            with open(training_log_path, "a") as f:
                f.write(f"Step {global_step}: Training Loss: {epoch_loss:.6f}\n")

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

                # Log detailed per-class dice scores for the best model
                with open(training_log_path, "a") as f:
                    f.write(f"*** BEST MODEL SAVED at Step {global_step} ***\n")
                    f.write(f"Best Average Dice Score: {dice_val_best:.6f}\n")
                    f.write("Per-class Dice Scores for Best Model:\n")
                    for class_idx in range(out_channel):
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
            val_outputs_softmax, val_labels, include_background=True
        )
        mean_dice = torch.mean(dice_scores).item()

        print(f"Sample Dice Score: {mean_dice:.4f}")

        # Move to CPU and convert to numpy
        image = val_inputs[0, 0].cpu().numpy()  # First channel, first batch
        label = val_labels[0, 0].cpu().numpy()  # First channel, first batch
        prediction = val_predictions[0].cpu().numpy()  # First batch

        # Create visualization
        create_slice_visualization(
            image, label, prediction, out_dir, mean_dice, out_channel
        )


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
        pretraining_out_channel=args.pretraining_out_channel,
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
    step_values = []  # Track steps for plotting
    print("Starting training...")
    time_start = time.time()
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step,
            train_loader,
            dice_val_best,
            global_step_best,
            training_log_path,
            args.out_channel,
        )
    time_end = time.time()
    print(
        f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}."
    )

    # Save training metrics for visualization
    metrics_data = {
        "training_loss": epoch_loss_values,
        "validation_dice": metric_values,
        "steps": list(range(eval_num, len(epoch_loss_values) * eval_num + 1, eval_num)),
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
    print_summary(epoch_loss_values, metric_values, metrics_data["steps"])

    # Use the plot_metrics function from visualize_training_metrics.py
    # This will create and save the visualization plots without showing them
    plot_metrics(epoch_loss_values, metric_values, metrics_data["steps"], out_dir)

    with open(training_log_path, "a") as f:
        f.write(
            f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}.\n"
        )
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
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Best model not found, using current model for inference")

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
