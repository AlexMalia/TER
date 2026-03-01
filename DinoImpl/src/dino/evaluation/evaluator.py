from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from dino.models.dino_model import DinoModel


class Evaluator(ABC):

    @abstractmethod
    def evaluate(self, model: DinoModel, train_loader: DataLoader, test_loader: DataLoader) -> dict:
        """Run evaluation and return a dict of metric_name -> value."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, evaluate_config) -> "Evaluator":
        """Instantiate from an EvaluationConfig."""
        ...

    def plot(self, model: DinoModel, data_loader: DataLoader, save_path: str):
        """Optional method to plot evaluation results (e.g. t-SNE visualization)."""
        ...