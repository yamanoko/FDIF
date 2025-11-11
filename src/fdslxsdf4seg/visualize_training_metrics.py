"""
Training metrics visualization script.
This script loads and visualizes the training loss and validation Dice scores saved during training.

Usage Examples:
1. Visualize a single training run:
   python visualize_training_metrics.py --output_dir path/to/training_output

2. Compare multiple training runs:
   python visualize_training_metrics.py --output_dirs path/to/run1 path/to/run2 path/to/run3

3. Compare multiple runs and save plots to specific directory:
   python visualize_training_metrics.py --output_dirs path/to/run1 path/to/run2 --save_to results/

4. Only print summary without plots:
   python visualize_training_metrics.py --output_dirs path/to/run1 path/to/run2 --no_plot

Generated Files:
- Single run: training_metrics_plot.png, training_loss_individual.png, validation_dice_individual.png
- Multiple runs: training_metrics_comparison.png, training_loss_comparison_individual.png,
  validation_dice_comparison_individual.png, convergence_analysis.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(output_dir):
    """Load training metrics from the output directory."""
    try:
        # Load numpy arrays
        training_loss = np.load(os.path.join(output_dir, "training_loss.npy"))
        validation_dice = np.load(os.path.join(output_dir, "validation_dice.npy"))
        steps = np.load(os.path.join(output_dir, "steps.npy"))

        return training_loss, validation_dice, steps
    except FileNotFoundError as e:
        print(f"Error loading metrics: {e}")
        print("Make sure the training has completed and metrics were saved.")
        return None, None, None


def plot_metrics(training_loss, validation_dice, steps, output_dir):
    """Create and save training metrics plots."""
    # Set matplotlib's maximum image size to prevent memory errors
    import matplotlib

    matplotlib.rcParams["figure.max_open_warning"] = 0

    # Validate input data
    if len(steps) == 0 or len(training_loss) == 0 or len(validation_dice) == 0:
        print("Warning: Empty data provided to plot_metrics. Skipping plot generation.")
        return

    if len(steps) != len(training_loss) or len(steps) != len(validation_dice):
        print(
            f"Warning: Data length mismatch - steps: {len(steps)}, loss: {len(training_loss)}, dice: {len(validation_dice)}"
        )
        # Truncate to shortest length
        min_len = min(len(steps), len(training_loss), len(validation_dice))
        steps = steps[:min_len]
        training_loss = training_loss[:min_len]
        validation_dice = validation_dice[:min_len]
        print(f"Truncated all arrays to length {min_len}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot training loss
    ax1.plot(steps, training_loss, "b-", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot validation Dice score
    ax2.plot(steps, validation_dice, "r-", linewidth=2, label="Validation Dice Score")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add best Dice score annotation with safe position calculation
    best_dice = np.max(validation_dice)
    best_step = steps[np.argmax(validation_dice)]
    # Use offset from data coordinates instead of multiplication
    x_range = steps[-1] - steps[0] if len(steps) > 1 else steps[0]
    y_range = np.max(validation_dice) - np.min(validation_dice)
    y_offset = max(y_range * 0.1, best_dice * 0.05)  # Ensure minimum offset
    ax2.annotate(
        f"Best: {best_dice:.4f} at step {best_step}",
        xy=(best_step, best_dice),
        xytext=(best_step - x_range * 0.2, best_dice - y_offset),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=12,
        color="red",
    )

    plt.tight_layout()

    # Save the combined plot
    plot_path = os.path.join(output_dir, "training_metrics_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training metrics plot saved to: {plot_path}")

    # Create and save individual plots
    save_individual_plots(training_loss, validation_dice, steps, output_dir)


def save_individual_plots(training_loss, validation_dice, steps, output_dir):
    """Save individual plots for training loss and validation Dice score."""

    # Validate input data
    if len(steps) == 0 or len(training_loss) == 0 or len(validation_dice) == 0:
        print(
            "Warning: Empty data provided to save_individual_plots. Skipping plot generation."
        )
        return

    if len(steps) != len(training_loss) or len(steps) != len(validation_dice):
        print("Warning: Data length mismatch in save_individual_plots")
        min_len = min(len(steps), len(training_loss), len(validation_dice))
        steps = steps[:min_len]
        training_loss = training_loss[:min_len]
        validation_dice = validation_dice[:min_len]

    # Individual Training Loss Plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(steps, training_loss, "b-", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Add lowest loss annotation with safe position calculation
    min_loss = np.min(training_loss)
    min_loss_step = steps[np.argmin(training_loss)]
    # Use offset from data coordinates instead of multiplication
    x_range = steps[-1] - steps[0] if len(steps) > 1 else steps[0]
    y_range = np.max(training_loss) - np.min(training_loss)
    y_offset = max(y_range * 0.1, min_loss * 0.2)  # Ensure minimum offset
    ax1.annotate(
        f"Lowest: {min_loss:.6f} at step {min_loss_step}",
        xy=(min_loss_step, min_loss),
        xytext=(min_loss_step + x_range * 0.1, min_loss + y_offset),
        arrowprops=dict(arrowstyle="->", color="blue"),
        fontsize=10,
        color="blue",
    )

    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "training_loss_individual.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Individual training loss plot saved to: {loss_plot_path}")

    # Individual Validation Dice Score Plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(steps, validation_dice, "r-", linewidth=2, label="Validation Dice Score")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add best Dice score annotation with safe position calculation
    best_dice = np.max(validation_dice)
    best_step = steps[np.argmax(validation_dice)]
    # Use offset from data coordinates instead of multiplication
    x_range = steps[-1] - steps[0] if len(steps) > 1 else steps[0]
    y_range = np.max(validation_dice) - np.min(validation_dice)
    y_offset = max(y_range * 0.1, best_dice * 0.05)  # Ensure minimum offset
    ax2.annotate(
        f"Best: {best_dice:.4f} at step {best_step}",
        xy=(best_step, best_dice),
        xytext=(best_step - x_range * 0.2, best_dice - y_offset),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    dice_plot_path = os.path.join(output_dir, "validation_dice_individual.png")
    plt.savefig(dice_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Individual validation Dice plot saved to: {dice_plot_path}")


def print_summary(training_loss, validation_dice, steps):
    """Print summary statistics of the training."""
    print("\n=== Training Summary ===")
    print(f"Total training steps: {steps[-1]}")
    print(f"Number of evaluations: {len(validation_dice)}")
    print(f"Final training loss: {training_loss[-1]:.6f}")
    print(f"Final validation Dice: {validation_dice[-1]:.6f}")
    print(
        f"Best validation Dice: {np.max(validation_dice):.6f} at step {steps[np.argmax(validation_dice)]}"
    )
    print(
        f"Lowest training loss: {np.min(training_loss):.6f} at step {steps[np.argmin(training_loss)]}"
    )


def load_multiple_metrics(output_dirs):
    """Load training metrics from multiple output directories."""
    all_metrics = []

    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            print(f"Warning: Directory {output_dir} does not exist, skipping...")
            continue

        try:
            # Load numpy arrays
            training_loss = np.load(os.path.join(output_dir, "training_loss.npy"))
            validation_dice = np.load(os.path.join(output_dir, "validation_dice.npy"))
            steps = np.load(os.path.join(output_dir, "steps.npy"))

            # Extract directory name for labeling
            dir_name = os.path.basename(output_dir.rstrip("/\\"))

            all_metrics.append(
                {
                    "name": dir_name,
                    "training_loss": training_loss,
                    "validation_dice": validation_dice,
                    "steps": steps,
                    "path": output_dir,
                }
            )

        except FileNotFoundError as e:
            print(f"Warning: Could not load metrics from {output_dir}: {e}")
            continue

    return all_metrics


def plot_multiple_metrics(all_metrics, output_dir):
    """Create and save comparison plots for multiple training runs."""
    if not all_metrics:
        print("No valid metrics data found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Define colors for different runs
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Plot training loss comparison
    for i, metrics in enumerate(all_metrics):
        color = colors[i % len(colors)]
        ax1.plot(
            metrics["steps"],
            metrics["training_loss"],
            linewidth=2,
            label=f"{metrics['name']} - Loss",
            color=color,
            linestyle="-",
        )

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot validation Dice score comparison
    for i, metrics in enumerate(all_metrics):
        color = colors[i % len(colors)]
        ax2.plot(
            metrics["steps"],
            metrics["validation_dice"],
            linewidth=2,
            label=f"{metrics['name']} - Dice",
            color=color,
            linestyle="-",
        )

        # Add best score annotation for each run
        best_dice = np.max(metrics["validation_dice"])
        best_step = metrics["steps"][np.argmax(metrics["validation_dice"])]
        ax2.scatter(
            best_step,
            best_dice,
            color=color,
            s=100,
            marker="*",
            zorder=5,
            edgecolors="black",
            linewidth=1,
        )
        ax2.annotate(
            f"{best_dice:.3f}",
            xy=(best_step, best_dice),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color=color,
            weight="bold",
        )

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save the comparison plot
    plot_path = os.path.join(output_dir, "training_metrics_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Training metrics comparison plot saved to: {plot_path}")

    # Show the plot
    plt.show()

    # Create and save individual comparison plots
    save_individual_comparison_plots(all_metrics, output_dir)


def save_individual_comparison_plots(all_metrics, output_dir):
    """Save individual comparison plots for training loss and validation Dice score."""
    if not all_metrics:
        return

    # Define colors for different runs
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Individual Training Loss Comparison Plot
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    for i, metrics in enumerate(all_metrics):
        color = colors[i % len(colors)]
        ax1.plot(
            metrics["steps"],
            metrics["training_loss"],
            linewidth=2,
            label=f"{metrics['name']}",
            color=color,
            linestyle="-",
        )

        # Add lowest loss annotation for each run
        min_loss = np.min(metrics["training_loss"])
        min_loss_step = metrics["steps"][np.argmin(metrics["training_loss"])]
        ax1.scatter(
            min_loss_step,
            min_loss,
            color=color,
            s=80,
            marker="v",
            zorder=5,
            edgecolors="black",
            linewidth=1,
        )

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    loss_comparison_path = os.path.join(
        output_dir, "training_loss_comparison_individual.png"
    )
    plt.savefig(loss_comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Individual training loss comparison plot saved to: {loss_comparison_path}")

    # Individual Validation Dice Score Comparison Plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))

    for i, metrics in enumerate(all_metrics):
        color = colors[i % len(colors)]
        ax2.plot(
            metrics["steps"],
            metrics["validation_dice"],
            linewidth=2,
            label=f"{metrics['name']}",
            color=color,
            linestyle="-",
        )

        # Add best score annotation for each run
        best_dice = np.max(metrics["validation_dice"])
        best_step = metrics["steps"][np.argmax(metrics["validation_dice"])]
        ax2.scatter(
            best_step,
            best_dice,
            color=color,
            s=100,
            marker="*",
            zorder=5,
            edgecolors="black",
            linewidth=1,
        )
        ax2.annotate(
            f"{best_dice:.3f}",
            xy=(best_step, best_dice),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color=color,
            weight="bold",
        )

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice Score Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    dice_comparison_path = os.path.join(
        output_dir, "validation_dice_comparison_individual.png"
    )
    plt.savefig(dice_comparison_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"Individual validation Dice comparison plot saved to: {dice_comparison_path}"
    )


def create_summary_table(all_metrics):
    """Create a summary table comparing all training runs."""
    if not all_metrics:
        return

    print("\n" + "=" * 80)
    print("TRAINING RUNS COMPARISON SUMMARY")
    print("=" * 80)

    # Header
    print(
        f"{'Run Name':<25} {'Best Dice':<12} {'Final Dice':<12} {'Final Loss':<12} {'Best Step':<10}"
    )
    print("-" * 80)

    # Data for each run
    for metrics in all_metrics:
        best_dice = np.max(metrics["validation_dice"])
        best_step = metrics["steps"][np.argmax(metrics["validation_dice"])]
        final_dice = metrics["validation_dice"][-1]
        final_loss = metrics["training_loss"][-1]

        print(
            f"{metrics['name']:<25} {best_dice:<12.6f} {final_dice:<12.6f} {final_loss:<12.6f} {best_step:<10}"
        )

    print("-" * 80)

    # Find best overall performance
    best_overall_dice = 0
    best_run_name = ""
    for metrics in all_metrics:
        best_dice = np.max(metrics["validation_dice"])
        if best_dice > best_overall_dice:
            best_overall_dice = best_dice
            best_run_name = metrics["name"]

    print(f"Best overall performance: {best_run_name} (Dice: {best_overall_dice:.6f})")
    print("=" * 80)


def plot_convergence_analysis(all_metrics, output_dir):
    """Create convergence analysis plots."""
    if not all_metrics:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "olive",
        "cyan",
    ]

    # Plot 1: Learning rate analysis (loss derivative)
    ax1.set_title("Training Loss Convergence Rate")
    for i, metrics in enumerate(all_metrics):
        color = colors[i % len(colors)]
        steps = metrics["steps"]
        loss = metrics["training_loss"]

        # Calculate moving average for smoother visualization
        if len(loss) > 5:
            window_size = min(5, len(loss))
            loss_smooth = np.convolve(
                loss, np.ones(window_size) / window_size, mode="valid"
            )
            steps_smooth = steps[: len(loss_smooth)]
            ax1.plot(
                steps_smooth,
                loss_smooth,
                linewidth=2,
                label=f"{metrics['name']}",
                color=color,
            )

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Smoothed Training Loss")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Dice score improvement rate
    ax2.set_title("Validation Dice Score Progress")
    for i, metrics in enumerate(all_metrics):
        color = colors[i % len(colors)]
        steps = metrics["steps"]
        dice = metrics["validation_dice"]

        ax2.plot(
            steps,
            dice,
            linewidth=2,
            label=f"{metrics['name']}",
            color=color,
            marker="o",
            markersize=3,
        )

    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Validation Dice Score")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save the convergence analysis plot
    plot_path = os.path.join(output_dir, "convergence_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Convergence analysis plot saved to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to a single training output directory containing metrics files",
    )
    parser.add_argument(
        "--output_dirs",
        type=str,
        nargs="+",
        help="Paths to multiple training output directories for comparison",
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="Skip plotting, only print summary"
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=".",
        help="Directory to save comparison plots (default: current directory)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.output_dir and args.output_dirs:
        print("Error: Please specify either --output_dir or --output_dirs, not both.")
        exit(1)

    if not args.output_dir and not args.output_dirs:
        print(
            "Error: Please specify either --output_dir for single run or --output_dirs for comparison."
        )
        exit(1)

    # Handle single directory case
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            print(f"Error: Output directory {args.output_dir} does not exist.")
            exit(1)

        # Load metrics
        training_loss, validation_dice, steps = load_metrics(args.output_dir)

        if training_loss is None:
            exit(1)

        # Print summary
        print_summary(training_loss, validation_dice, steps)

        # Create plots
        if not args.no_plot:
            plot_metrics(training_loss, validation_dice, steps, args.output_dir)

    # Handle multiple directories case
    elif args.output_dirs:
        print(f"Comparing {len(args.output_dirs)} training runs...")

        # Load metrics from all directories
        all_metrics = load_multiple_metrics(args.output_dirs)

        if not all_metrics:
            print("Error: No valid metrics found in any of the specified directories.")
            exit(1)

        print(f"Successfully loaded metrics from {len(all_metrics)} training runs.")

        # Create summary table
        create_summary_table(all_metrics)

        # Create comparison plots
        if not args.no_plot:
            plot_multiple_metrics(all_metrics, args.save_to)
            plot_convergence_analysis(all_metrics, args.save_to)

    # For loading and visualizing multiple metrics
    # Example usage (uncomment to use):
    """
    output_dirs = [
        "path/to/first/training/output",
        "path/to/second/training/output",
        # Add more paths as needed
    ]
    all_metrics = load_multiple_metrics(output_dirs)
    plot_multiple_metrics(all_metrics, "path/to/save/comparison/plot")
    create_summary_table(all_metrics)
    plot_convergence_analysis(all_metrics, "path/to/save/convergence/plot")
    """
