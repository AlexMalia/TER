"""Training history tracking and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class History:
    """
    Track training metrics at iteration and epoch granularity.

    Stores metrics in a structured format with JSON serialization support.
    Provides data access methods and basic matplotlib visualization.

    Core metrics tracked:
    - loss: Training loss value
    - learning_rate: Current learning rate from optimizer
    - momentum: Teacher momentum (EMA coefficient)

    Attributes:
        iteration_metrics: List of dicts, one per logged iteration
        epoch_metrics: List of dicts, one per epoch
        metadata: Dict with training run info (start_time, config, etc.)
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize History tracker.

        Args:
            metadata: Optional dict with run metadata (config, start_time, etc.)
        """
        self.iteration_metrics: List[Dict[str, Any]] = []
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.evaluation_metrics: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = metadata or {}
        self.metadata.setdefault('created_at', datetime.now().isoformat())

    def record_iteration(self, iteration: int, metrics: Dict[str, float]) -> None:
        """
        Record metrics for a single training iteration.

        Args:
            iteration: Global iteration number (0-indexed)
            metrics: Dict with metric names and values
                     Expected keys: 'loss', 'learning_rate', 'momentum'
        """
        record = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.iteration_metrics.append(record)

    def record_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Record summary metrics for a completed epoch.

        Args:
            epoch: Epoch number (1-indexed, matching trainer convention)
            metrics: Dict with aggregated metric values
                     Expected keys: 'loss', 'learning_rate', 'momentum'
        """
        record = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.epoch_metrics.append(record)

    def record_evaluation(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Record evaluation metrics at the end of an epoch.

        Args:
            epoch: Epoch number (1-indexed)
            metrics: Dict with evaluation metric values
                     Expected keys depend on evaluation setup (e.g. 'knn_top1', 'knn_top5')
        """
        record = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.evaluation_metrics.append(record)

    def get_metric(self, name: str, level: str = 'iteration') -> List[float]:
        """
        Retrieve a specific metric as a list.

        Args:
            name: Metric name ('loss', 'learning_rate', 'momentum')
            level: Either 'iteration' or 'epoch'

        Returns:
            List of metric values in chronological order

        Raises:
            ValueError: If level is invalid
        """
        if level == 'iteration':
            data = self.iteration_metrics
        elif level == 'epoch':
            data = self.epoch_metrics
        else:
            raise ValueError(f"Invalid level: {level}. Use 'iteration' or 'epoch'")

        return [record[name] for record in data if name in record]

    def get_iterations(self) -> List[int]:
        """Get list of recorded iteration numbers."""
        return [record['iteration'] for record in self.iteration_metrics]

    def get_epochs(self) -> List[int]:
        """Get list of recorded epoch numbers."""
        return [record['epoch'] for record in self.epoch_metrics]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert history to dictionary for serialization.

        Returns:
            Dict with all history data
        """
        return {
            'metadata': self.metadata,
            'iteration_metrics': self.iteration_metrics,
            'epoch_metrics': self.epoch_metrics,
            'evaluation_metrics': self.evaluation_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'History':
        """
        Create History instance from dictionary.

        Args:
            data: Dict with history data (from to_dict())

        Returns:
            History instance
        """
        history = cls(metadata=data.get('metadata', {}))
        history.iteration_metrics = data.get('iteration_metrics', [])
        history.epoch_metrics = data.get('epoch_metrics', [])
        history.evaluation_metrics = data.get('evaluation_metrics', [])
        return history

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save history to JSON file.

        Args:
            path: File path to save to (should end in .json)

        Returns:
            Path object to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"History saved to {path}")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'History':
        """
        Load history from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            History instance with loaded data

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        logger.info(f"History loaded from {path}")
        return cls.from_dict(data)

    def to_dataframe(self, level: str = 'iteration'):
        """
        Convert history to pandas DataFrame.

        Args:
            level: Either 'iteration' or 'epoch'

        Returns:
            pandas.DataFrame with metrics

        Raises:
            ImportError: If pandas is not installed
            ValueError: If level is invalid
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        if level == 'iteration':
            return pd.DataFrame(self.iteration_metrics)
        elif level == 'epoch':
            return pd.DataFrame(self.epoch_metrics)
        else:
            raise ValueError(f"Invalid level: {level}. Use 'iteration' or 'epoch'")

    def plot_loss(
        self,
        level: str = 'epoch',
        ax=None,
        title: str = 'Training Loss',
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Plot training loss curve.

        Args:
            level: 'iteration' or 'epoch'
            ax: Optional matplotlib axes to plot on
            title: Plot title
            save_path: Optional path to save figure
            **kwargs: Additional kwargs passed to plt.plot()

        Returns:
            matplotlib.axes.Axes object
        """
        return self._plot_metric(
            name='loss',
            level=level,
            ax=ax,
            title=title,
            ylabel='Loss',
            save_path=save_path,
            **kwargs
        )

    def plot_learning_rate(
        self,
        level: str = 'iteration',
        ax=None,
        title: str = 'Learning Rate Schedule',
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Plot learning rate over training.

        Args:
            level: 'iteration' or 'epoch'
            ax: Optional matplotlib axes
            title: Plot title
            save_path: Optional path to save figure
            **kwargs: Additional kwargs passed to plt.plot()

        Returns:
            matplotlib.axes.Axes object
        """
        return self._plot_metric(
            name='learning_rate',
            level=level,
            ax=ax,
            title=title,
            ylabel='Learning Rate',
            save_path=save_path,
            **kwargs
        )

    def plot_momentum(
        self,
        level: str = 'iteration',
        ax=None,
        title: str = 'Teacher Momentum Schedule',
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Plot teacher momentum over training.

        Args:
            level: 'iteration' or 'epoch'
            ax: Optional matplotlib axes
            title: Plot title
            save_path: Optional path to save figure
            **kwargs: Additional kwargs passed to plt.plot()

        Returns:
            matplotlib.axes.Axes object
        """
        return self._plot_metric(
            name='momentum',
            level=level,
            ax=ax,
            title=title,
            ylabel='Momentum',
            save_path=save_path,
            **kwargs
        )
    


    def plot_knn(self, k_values=None, ax=None, title='KNN Accuracy', save_path=None, **kwargs):
     """Plot KNN top-1 and top-5 accuracy over epochs.

     Args:
         k_values: k values to plot. If None, auto-detected from stored metrics.
         ax: Optional existing matplotlib axes.
         title: Plot title.
         save_path: Optional path to save the figure.

     Returns:
         matplotlib.axes.Axes
     """
     if not self.evaluation_metrics:
         logger.warning("No evaluation metrics recorded, skipping plot_knn")
         return None

     # Auto-detect k values from first recorded entry
     if k_values is None:
         sample = self.evaluation_metrics[0]
         k_values = sorted([
             int(key.split('_k')[1])
             for key in sample
             if key.startswith('knn_top1_k')
         ])

     epochs = [m['epoch'] for m in self.evaluation_metrics]

     created_fig = ax is None
     if created_fig:
         fig, ax = plt.subplots(figsize=(10, 6))

     for k in k_values:
         top1 = [m.get(f'knn_top1_k{k}') for m in self.evaluation_metrics]
         top5 = [m.get(f'knn_top5_k{k}') for m in self.evaluation_metrics]

         ax.plot(epochs, top1, label=f'Top-1 k={k}', **kwargs)
         if any(v is not None for v in top5):
             ax.plot(epochs, top5, label=f'Top-5 k={k}', linestyle='--', **kwargs)

     ax.set_xlabel('Epoch')
     ax.set_ylabel('Accuracy (%)')
     ax.set_title(title)
     ax.legend()
     ax.grid(True, alpha=0.3)

     if save_path and created_fig:
         plt.savefig(save_path, dpi=150, bbox_inches='tight')
         logger.info(f"Plot saved to {save_path}")

     return ax

    def _plot_metric(
        self,
        name: str,
        level: str,
        ax=None,
        title: str = '',
        ylabel: str = '',
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Internal method to plot a single metric.

        Args:
            name: Metric name
            level: 'iteration' or 'epoch'
            ax: Optional matplotlib axes
            title: Plot title
            ylabel: Y-axis label
            save_path: Optional path to save figure
            **kwargs: Additional plot kwargs

        Returns:
            matplotlib.axes.Axes object
        """
        values = self.get_metric(name, level=level)
        if level == 'iteration':
            x_values = self.get_iterations()
            xlabel = 'Iteration'
        else:
            x_values = self.get_epochs()
            xlabel = 'Epoch'

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x_values, values, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        return ax

    def plot_all(
        self,
        level: str = 'epoch',
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Create a figure with all metrics in subplots.

        Args:
            level: 'iteration' or 'epoch'
            save_path: Optional path to save figure

        Returns:
            matplotlib.figure.Figure object
        """
        has_knn = bool(self.evaluation_metrics)
        if has_knn:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        self.plot_loss(level=level, ax=axes[0])
        self.plot_learning_rate(level=level, ax=axes[1])
        self.plot_momentum(level=level, ax=axes[2])
        if has_knn:
            self.plot_knn(ax=axes[3])
            axes[3].set_title('KNN Accuracy')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Combined plot saved to {save_path}")

        return fig

    def __len__(self) -> int:
        """Return number of recorded iterations."""
        return len(self.iteration_metrics)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"History(iterations={len(self.iteration_metrics)}, "
            f"epochs={len(self.epoch_metrics)})"
        )
