import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fdslxsdf4seg.lr_scheduler import LinearWarmupCosineAnnealingLR
from fdslxsdf4seg.visualize_training_metrics import (
    plot_metrics,
    print_summary,
)


def setup_distributed(rank, world_size, is_multi_node=False):
    """
    Initialize the distributed environment for multi-GPU training.

    Args:
        rank: GPU ID for the current process (global rank for multi-node)
        world_size: Total number of GPUs available across all nodes
        is_multi_node: Whether this is multi-node training

    Returns:
        None
    """
    if is_multi_node:
        # For multi-node training with MPI
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12355")

        try:
            dist.init_process_group(
                "nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank,
                world_size=world_size,
            )
            print(f"Process {rank} initialized (world_size: {world_size})")
        except Exception as e:
            print(f"Failed to initialize process group: {e}")
            raise e
    else:
        # For single-node training
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # Initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """
    Clean up the distributed environment.

    Returns:
        None
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def print_only_rank0(log: str) -> None:
    """
    Print a log message only from rank 0 process.

    Args:
        log: Message to log

    Returns:
        None
    """
    try:
        if dist.is_initialized() and dist.get_rank() == 0:
            print(log)
    except Exception:
        # Fallback for when the process group isn't initialized
        # In single process mode, always print
        if not dist.is_initialized():
            print(log)


def make_data_loader_ddp(
    data_json_path: str,
    real_data: bool = True,
    spatial_size: tuple = (96, 96, 96),
    batch_size: int = 1,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Create data loaders with distributed sampling for DDP training.
    """
    num_samples = 4

    # Set device based on local rank
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        # Multi-node case
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Single-node case
        device = torch.device(f"cuda:{rank}")

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

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
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

    # Validation sampler for distributed evaluation
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    val_loader = ThreadDataLoader(
        val_ds, batch_size=1, sampler=val_sampler, num_workers=0
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
    """Create model (same as original training.py)"""
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
                spatial_dims=3,
                feature_size=feature_size or 16,
            )
    elif model_name == "swin_unetr":
        if pretrained_path:
            weights = torch.load(pretrained_path, weights_only=True)
            model = SwinUNETR(
                in_channels=1,
                out_channels=pretraining_out_channel,
                spatial_dims=3,
                feature_size=feature_size or 48,
            )
            model.load_state_dict(weights)
            model.out = UnetOutBlock(
                spatial_dims=3,
                in_channels=feature_size or 48,
                out_channels=out_channel,
            )
            print(f"Model {model_name} loaded from {pretrained_path}")
        else:
            model = SwinUNETR(
                in_channels=1,
                out_channels=out_channel,
                spatial_dims=3,
                feature_size=feature_size or 48,
            )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
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


def validation_ddp(
    epoch_iterator_val,
    global_step,
    training_log_path,
    model,
    device,
    grid_size,
    post_label,
    post_pred,
    dice_metric,
    out_channel=14,
):
    """Validation function for DDP training"""
    model.eval()
    run_acc = AverageMeter()
    raw_dice_scores = []

    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (
                batch["image"].to(device),
                batch["label"].to(device),
            )
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
            )

        mean_dice_val = np.mean(run_acc.avg).item()
        class_dice_score = torch.stack(raw_dice_scores, dim=0).mean(dim=0).numpy()

        # Gather results from all processes
        if dist.is_initialized():
            world_size = dist.get_world_size()

            # Gather dice scores from all processes
            dice_tensor = torch.tensor([mean_dice_val], device=device)
            gathered_dice = [torch.zeros_like(dice_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_dice, dice_tensor)

            # Average across all processes
            mean_dice_val = torch.stack(gathered_dice).mean().item()

        # Log evaluation results (only rank 0)
        if not dist.is_initialized() or dist.get_rank() == 0:
            with open(training_log_path, "a") as f:
                f.write(
                    f"Step {global_step}: Validation Dice Score: {mean_dice_val:.6f}\n"
                )
                # Log per-class dice scores
                for class_idx in range(out_channel - 1):
                    f.write(
                        f"Step {global_step}: Class {class_idx} Dice Score: {class_dice_score[class_idx].item():.6f}\n"
                    )

        # Clear the dice scores list
        del raw_dice_scores
        torch.cuda.empty_cache()

    return mean_dice_val, class_dice_score


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
    """Save training checkpoint (only rank 0)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        # Save the underlying model without DDP wrapper
        model_state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )

        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "global_step": global_step,
            "dice_val_best": dice_val_best,
            "global_step_best": global_step_best,
            "epoch_loss_values": epoch_loss_values,
            "metric_values": metric_values,
        }
        torch.save(checkpoint, checkpoint_path)
        print_only_rank0(f"Checkpoint saved at step {global_step}: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler):
    """Load training checkpoint and return training state."""
    if not os.path.exists(checkpoint_path):
        print_only_rank0(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0, 0, [], []

    print_only_rank0(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Load to the underlying model (handle DDP wrapper)
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
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

    print_only_rank0(
        f"Resumed training from step {global_step}, best dice: {dice_val_best:.4f}"
    )
    return (
        global_step,
        dice_val_best,
        global_step_best,
        epoch_loss_values,
        metric_values,
    )


def train_ddp(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
    is_multi_node: bool = False,
):
    """
    Main training function for DDP.

    Args:
        rank: GPU rank (global rank for multi-node)
        world_size: Total number of GPUs across all nodes
        args: Command line arguments
        is_multi_node: Whether this is multi-node training
    """
    # For multi-node, calculate local rank
    if is_multi_node:
        local_rank = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        print(f"MPI rank: {rank}, local rank: {local_rank}, world size: {world_size}")
    else:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)

    # Initialize distributed training
    setup_distributed(rank, world_size, is_multi_node)

    print_only_rank0(f"Starting DDP training on {world_size} GPUs")
    print_only_rank0(f"Arguments: {args}")

    # Determine output directory and training state
    pretraining_state = (
        "fine_tuning" if args.pretrained_model else "training_from_scratch"
    )
    if not args.out_dir:
        args.out_dir = os.path.join(
            "training_output",
            args.model_name,
            pretraining_state + "_ddp",
            time.strftime("%Y%m%d_%H%M%S"),
        )

    out_dir = args.out_dir
    if rank == 0:  # Only rank 0 creates directories and files
        os.makedirs(out_dir, exist_ok=True)
        training_log_path = os.path.join(out_dir, "training_log.txt")
        with open(training_log_path, "w") as f:
            f.write(str(args) + "\n")
            f.write(f"DDP Training with {world_size} GPUs\n")
        print_only_rank0(f"Training log will be saved to {training_log_path}")
        print_only_rank0(f"Output directory: {out_dir}")
    else:
        training_log_path = os.path.join(out_dir, "training_log.txt")

    # Wait for rank 0 to create directories
    if dist.is_initialized():
        dist.barrier()

    grid_size = tuple(args.grid_size)

    # Create data loaders
    train_loader, val_loader = make_data_loader_ddp(
        data_json_path=args.data_json_path,
        real_data=args.is_real_data,
        spatial_size=grid_size,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )

    print_only_rank0(
        f"Training data loader created with {len(train_loader.dataset)} samples."
    )
    print_only_rank0(
        f"Validation data loader created with {len(val_loader.dataset)} samples."
    )

    # Create model
    model = create_model(
        model_name=args.model_name,
        grid_size=grid_size,
        out_channel=args.out_channel,
        feature_size=args.feature_size,
        pretrained_path=args.pretrained_model,
        pretraining_out_channel=args.pretraining_out_channel,
    )

    model = model.to(device)

    # Wrap model with DDP
    if is_multi_node:
        ddp_model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,  # Reduce communication overhead
            find_unused_parameters=False,  # Performance improvement
            gradient_as_bucket_view=True,  # Handle stride issues
        )
    else:
        ddp_model = DDP(model, device_ids=[rank])

    print_only_rank0(
        f"Model {args.model_name} created with output channels: {args.out_channel}."
    )
    print_only_rank0(f"Using learning rate: {args.learning_rate}")

    # Create optimizer, loss function, etc.
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), args.learning_rate, weight_decay=1e-5
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

    # Load checkpoint if specified
    if args.resume_from_checkpoint:
        (
            global_step,
            dice_val_best,
            global_step_best,
            epoch_loss_values,
            metric_values,
        ) = load_checkpoint(
            args.resume_from_checkpoint, ddp_model, optimizer, scheduler, scaler
        )
        if rank == 0:
            with open(training_log_path, "a") as f:
                f.write(
                    f"Resumed training from checkpoint: {args.resume_from_checkpoint}\n"
                )
                f.write(
                    f"Resuming from step {global_step}, best dice: {dice_val_best:.4f}\n"
                )

    print_only_rank0("Starting training...")
    time_start = time.time()

    # Training loop
    while global_step < max_iterations:
        ddp_model.train()
        epoch_loss = 0
        step = 0

        # Set epoch for DistributedSampler
        train_loader.sampler.set_epoch(global_step // len(train_loader))

        epoch_iterator = tqdm(
            train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            dynamic_ncols=True,
            disable=(rank != 0),  # Only show progress bar on rank 0
        )

        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(device), batch["label"].to(device))

            # Clear gradients explicitly before forward pass
            optimizer.zero_grad()

            with torch.autocast("cuda"):
                logit_map = ddp_model(x)
                loss = loss_function(logit_map, y)

            # Store loss value before cleanup
            loss_value = loss.item()

            scaler.scale(loss).backward()
            epoch_loss += loss_value
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Explicitly delete variables to free GPU memory immediately
            del x, y, logit_map, loss
            # Clear gradients after step to ensure no gradient accumulation
            optimizer.zero_grad()

            # Force garbage collection and cache clearing more frequently
            if step % 10 == 0:
                torch.cuda.empty_cache()

            if rank == 0:
                epoch_iterator.set_description(
                    f"Training ({global_step} / {max_iterations} Steps) (loss={loss_value:2.5f})"
                )

            if (
                global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                # Clear memory before validation
                torch.cuda.empty_cache()

                epoch_iterator_val = tqdm(
                    val_loader,
                    desc="Validate (X / X Steps) (dice=X.X)",
                    dynamic_ncols=True,
                    disable=(rank != 0),
                )
                dice_val, dice_scores = validation_ddp(
                    epoch_iterator_val,
                    global_step,
                    training_log_path,
                    ddp_model,
                    device,
                    grid_size,
                    post_label,
                    post_pred,
                    dice_metric,
                    args.out_channel,
                )

                # Average loss across all processes
                if dist.is_initialized():
                    epoch_loss_tensor = torch.tensor([epoch_loss / step], device=device)
                    dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
                    epoch_loss = epoch_loss_tensor.item() / world_size
                else:
                    epoch_loss /= step

                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)

                # Log training results (only rank 0)
                if rank == 0:
                    with open(training_log_path, "a") as f:
                        f.write(
                            f"Step {global_step}: Training Loss: {epoch_loss:.6f}\n"
                        )

                # Save checkpoint after each evaluation (only rank 0)
                checkpoint_path = os.path.join(out_dir, "training_checkpoint.pth")
                save_checkpoint(
                    checkpoint_path,
                    ddp_model,
                    optimizer,
                    scheduler,
                    scaler,
                    global_step,
                    dice_val_best,
                    global_step_best,
                    epoch_loss_values,
                    metric_values,
                )

                # Save latest model after each validation (only rank 0)
                if rank == 0:
                    last_model_path = os.path.join(out_dir, "last_model.pth")
                    model_state_dict = (
                        ddp_model.module.state_dict()
                        if hasattr(ddp_model, "module")
                        else ddp_model.state_dict()
                    )
                    torch.save(model_state_dict, last_model_path)
                    print_only_rank0(
                        f"Latest model saved at step {global_step}: {last_model_path}"
                    )

                # Plot metrics (only rank 0)
                if rank == 0:
                    plot_metrics(
                        epoch_loss_values,
                        metric_values,
                        list(
                            range(
                                eval_num,
                                len(epoch_loss_values) * eval_num + 1,
                                eval_num,
                            )
                        ),
                        out_dir,
                    )

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step

                    # Save best model (only rank 0)
                    if rank == 0:
                        model_state_dict = (
                            ddp_model.module.state_dict()
                            if hasattr(ddp_model, "module")
                            else ddp_model.state_dict()
                        )
                        torch.save(
                            model_state_dict,
                            os.path.join(out_dir, "best_metric_model.pth"),
                        )

                        print_only_rank0(
                            "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                                dice_val_best, dice_val
                            )
                        )

                        # Log detailed per-class dice scores for the best model
                        with open(training_log_path, "a") as f:
                            f.write(f"*** BEST MODEL SAVED at Step {global_step} ***\n")
                            f.write(f"Best Average Dice Score: {dice_val_best:.6f}\n")
                            f.write("Per-class Dice Scores for Best Model:\n")
                            for class_idx in range(args.out_channel - 1):
                                f.write(
                                    f"  Class {class_idx}: {dice_scores[class_idx].item():.6f}\n"
                                )
                            f.write("=" * 50 + "\n")
                else:
                    print_only_rank0(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )

                # Reset epoch loss for next evaluation
                epoch_loss = 0

            global_step += 1

            # Periodic memory cleanup
            if global_step % 100 == 0:
                torch.cuda.empty_cache()

            # Break if we've reached max iterations
            if global_step >= max_iterations:
                break

    time_end = time.time()
    print_only_rank0(
        f"Training completed in {time_end - time_start:.2f} seconds. Best Dice: {dice_val_best:.4f} at step {global_step_best}."
    )

    # Save final metrics and create visualizations (only rank 0)
    if rank == 0:
        # Save training metrics for visualization
        metrics_data = {
            "training_loss": epoch_loss_values,
            "validation_dice": metric_values,
            "steps": list(
                range(eval_num, len(epoch_loss_values) * eval_num + 1, eval_num)
            ),
        }

        # Save as numpy arrays
        np.save(os.path.join(out_dir, "training_loss.npy"), np.array(epoch_loss_values))
        np.save(os.path.join(out_dir, "validation_dice.npy"), np.array(metric_values))
        np.save(os.path.join(out_dir, "steps.npy"), np.array(metrics_data["steps"]))

        # Save as JSON for easy reading
        with open(os.path.join(out_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Create training curves visualization
        print_only_rank0("Creating training curves visualization...")
        print_summary(epoch_loss_values, metric_values, metrics_data["steps"])
        plot_metrics(epoch_loss_values, metric_values, metrics_data["steps"], out_dir)

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

    # Clean up distributed environment
    cleanup_distributed()


def main_single_node():
    """Main function for single-node DDP training"""
    parser = argparse.ArgumentParser(
        description="DDP Training Script for Single/Multi-Node"
    )

    # Original arguments from training.py
    parser.add_argument(
        "--data_json_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file",
    )
    parser.add_argument("--is_real_data", action="store_true", help="Use real data")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["vnet", "unetr", "swin_unetr"],
        help="Name of the model to be trained",
    )
    parser.add_argument(
        "--pretrained_model", type=str, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--pretraining_out_channel",
        type=int,
        default=14,
        help="Output channel size for pretrained model",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs="+",
        default=[96, 96, 96],
        help="Grid size for the input data",
    )
    parser.add_argument(
        "--out_channel", type=int, default=14, help="Output channel size"
    )
    parser.add_argument("--feature_size", type=int, help="Feature size for SwinUNETR")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training and validation per GPU",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=30000, help="Maximum training iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Set PyTorch CUDA memory management settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Single-node DDP using mp.spawn
    world_size = torch.cuda.device_count()
    print(f"Starting single-node DDP training with {world_size} GPUs")

    mp.spawn(
        train_ddp,
        args=(world_size, args, False),  # False = single-node
        nprocs=world_size,
        join=True,
    )


def main_multi_node():
    """Main function for multi-node DDP training (called from MPI)"""
    parser = argparse.ArgumentParser(description="Multi-Node DDP Training Script")

    # Original arguments from training.py
    parser.add_argument(
        "--data_json_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file",
    )
    parser.add_argument("--is_real_data", action="store_true", help="Use real data")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["vnet", "unetr", "swin_unetr"],
        help="Name of the model to be trained",
    )
    parser.add_argument(
        "--pretrained_model", type=str, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--pretraining_out_channel",
        type=int,
        default=14,
        help="Output channel size for pretrained model",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs="+",
        default=[96, 96, 96],
        help="Grid size for the input data",
    )
    parser.add_argument(
        "--out_channel", type=int, default=14, help="Output channel size"
    )
    parser.add_argument("--feature_size", type=int, help="Feature size for SwinUNETR")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training and validation per GPU",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=30000, help="Maximum training iterations"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Set PyTorch CUDA memory management settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Multi-node DDP using MPI environment variables
    mpi_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    mpi_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))

    print(
        f"Starting multi-node DDP training: MPI rank {mpi_rank}, world size {mpi_size}"
    )

    # Call training function directly (no mp.spawn needed for multi-node)
    train_ddp(mpi_rank, mpi_size, args, True)  # True = multi-node


if __name__ == "__main__":
    # Determine if this is single-node or multi-node based on environment variables
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        # Multi-node training via MPI
        main_multi_node()
    else:
        # Single-node training
        main_single_node()
