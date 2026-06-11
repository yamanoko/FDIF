"""Visualize 2D slices of generated SDF samples (image + label mask).

Loads a MONAI-Decathlon-format dataset (the `data.json` produced by
`generate_sdf_dataset.py`) and, for each sample, renders three
orthogonal mid-plane slices (axial / coronal / sagittal) of the mapped
SDF volume `x` and the segmentation label `y`, side by side.

Usage:
    python scripts/visualize_slices.py --data_dir outputs/msn_test/data \
        --num 4 --out outputs/msn_test/slices
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _mid_slices(vol: np.ndarray):
    """Return (axial, coronal, sagittal) mid-plane 2D slices of a (D,H,W) vol."""
    d, h, w = vol.shape
    return vol[d // 2, :, :], vol[:, h // 2, :], vol[:, :, w // 2]


def visualize_sample(x: np.ndarray, y: np.ndarray, out_path: str, title: str):
    planes = ["Axial (z mid)", "Coronal (y mid)", "Sagittal (x mid)"]
    x_slices = _mid_slices(x)
    y_slices = _mid_slices(y)

    n_labels = int(y.max())
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (xs, ys, plane) in enumerate(zip(x_slices, y_slices, planes)):
        # Top row: mapped SDF volume (x).
        im0 = axes[0, col].imshow(xs, cmap="viridis", origin="lower")
        axes[0, col].set_title(f"SDF vol — {plane}")
        axes[0, col].axis("off")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Bottom row: label mask (y), discrete colors.
        im1 = axes[1, col].imshow(
            ys, cmap="tab20", origin="lower", vmin=0, vmax=max(1, n_labels)
        )
        axes[1, col].set_title(f"Label mask — {plane}")
        axes[1, col].axis("off")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="dir containing data.json")
    p.add_argument("--num", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = args.out or os.path.join(args.data_dir, "slices")
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(args.data_dir, "data.json")) as f:
        items = json.load(f)["training"]

    for i, it in enumerate(items[: args.num]):
        x = np.asarray(nib.load(it["image"]).get_fdata())
        y = np.asarray(nib.load(it["label"]).get_fdata())
        labels = sorted(int(v) for v in np.unique(y) if v != 0)
        out_path = os.path.join(out, f"sample_{i:02d}_slices.png")
        visualize_sample(
            x, y, out_path,
            title=f"{it['id']}  shape={x.shape}  labels={labels}",
        )
        print(f"[{i + 1}/{min(args.num, len(items))}] {it['id']} "
              f"x[{x.min():.0f},{x.max():.0f}] labels={labels} -> {out_path}")

    print(f"Saved slice figures to {out}")


if __name__ == "__main__":
    main()
