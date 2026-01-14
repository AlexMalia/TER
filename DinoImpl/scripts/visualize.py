#!/usr/bin/env python3
"""
DINO Training Visualization Script

Plot training metrics from history files.

Example usage:
    # Plot all metrics from latest history
    python scripts/visualize.py

    # Plot only loss
    python scripts/visualize.py --metric loss

    # Plot learning rate at iteration level
    python scripts/visualize.py --metric learning_rate --level iteration

    # Plot from specific history file
    python scripts/visualize.py --history checkpoints/checkpoint_epoch_0010_history.json

    # Save plot to file
    python scripts/visualize.py --metric loss --save plots/loss.png
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dino.utils.history import History


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DINO training metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--history",
        type=str,
        default="checkpoints/history_latest.json",
        help="Path to history JSON file"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["loss", "learning_rate", "momentum", "all"],
        default="all",
        help="Metric to plot"
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["iteration", "epoch"],
        default="epoch",
        help="Granularity level for plotting"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the plot (if not specified, displays interactively)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom title for the plot"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load history
    history_path = Path(args.history)
    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        sys.exit(1)

    print(f"Loading history from: {history_path}")
    history = History.load(history_path)
    print(f"Loaded {history}")

    # Import matplotlib here to allow --help without matplotlib installed
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install with: pip install matplotlib")
        sys.exit(1)

    # Plot based on selected metric
    if args.metric == "all":
        fig = history.plot_all(level=args.level, save_path=args.save)
        if args.title:
            fig.suptitle(args.title)
    elif args.metric == "loss":
        title = args.title or "Training Loss"
        history.plot_loss(level=args.level, title=title, save_path=args.save)
    elif args.metric == "learning_rate":
        title = args.title or "Learning Rate Schedule"
        history.plot_learning_rate(level=args.level, title=title, save_path=args.save)
    elif args.metric == "momentum":
        title = args.title or "Teacher Momentum Schedule"
        history.plot_momentum(level=args.level, title=title, save_path=args.save)

    # Show plot if not saving
    if args.save:
        print(f"Plot saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
