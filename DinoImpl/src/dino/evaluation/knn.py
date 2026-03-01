import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from .evaluator import Evaluator
from dino.models.dino_model import DinoModel
from ..config import EvaluationConfig

logger = logging.getLogger(__name__)

class KNNClassifier(Evaluator):

    def __init__(self, ks: list[int], temperature: float, batch_size: int):
        self.ks = ks
        self.temperature = temperature
        self.batch_size = batch_size

    @torch.no_grad()
    def extract_features(self, model: DinoModel, data_loader: DataLoader):
        """Extract features from the model for the entire dataset."""
        model.eval()
        all_features = []
        all_labels = []

        for images, label in data_loader:
            images = images.to(model.device)
            output = model.get_backbone_features(images)

            # L2 Norm to ensure cosine similarity is just dot product
            normalized_output = F.normalize(output, dim=-1)
            all_features.append(normalized_output.cpu())
            all_labels.append(label)

        return torch.cat(all_features), torch.cat(all_labels)

    def evaluate(self, model: DinoModel, train_loader: DataLoader, test_loader: DataLoader):
        train_features, train_labels = self.extract_features(model, train_loader)
        test_features, test_labels = self.extract_features(model, test_loader)

        num_classes = int(train_labels.max().item()) + 1
        results = {}

        for k in self.ks:
            knn_results = self.run_knn(test_features, test_labels, train_features, train_labels, num_classes)
            results[f"knn_top1_k{k}"] = knn_results["top1"]
            results[f"knn_top5_k{k}"] = knn_results["top5"]
        
        return results
            
    def run_knn(self, test_features: torch.Tensor, test_labels: torch.Tensor, train_features: torch.Tensor, train_labels: torch.Tensor, num_classes: int, k: int):
        # Features are L2 normalized, dot product = cosine similarity
        similarity = test_features @ train_features.T

        # k largest similarity scores and their indices
        distances, indices = similarity.topk(k=k, dim=-1)
        
        # Apply temperature scaling to the similarity scores to get weights for neighbors
        weights = (distances / self.temperature).exp()
        neighbor_labels = train_labels[indices]

        # Creating score matrix (N_test, num_classes), +1 for each neighbor's class, then apply temperature weighting
        scores = torch.zeros(len(test_labels), num_classes)
        scores.scatter_add_(1, neighbor_labels, weights)
        
        predictions = scores.argmax(dim=1)
        top1 = (predictions == test_labels).float().mean().item() * 100

        predictions = scores.argmax(dim=1)  # (N_test,)
        top1 = (predictions == test_labels).float().mean().item() * 100

        top5 = None
        if num_classes >= 5:
            top5_preds = scores.topk(5, dim=1).indices       # (N_test, 5)
            top5 = (top5_preds == test_labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100
        
        return {"top1": top1, "top5": top5}
    
    @classmethod
    def from_config(cls, evaluate_config: EvaluationConfig):
        return cls(
            ks=evaluate_config.knn_ks,
            temperature=evaluate_config.knn_temperature,
            batch_size=evaluate_config.knn_batch_size
        )