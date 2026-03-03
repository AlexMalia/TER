import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from .evaluator import Evaluator
from dino.models.dino_model import DinoModel
from ..config import EvaluationConfig

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



logger = logging.getLogger(__name__)

class KNNClassifier(Evaluator):

    def __init__(self, ks: list[int], temperature: float, batch_size: int, device: str):
        self.ks = ks
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def extract_features(self, model: DinoModel, data_loader: DataLoader):
        """Extract and L2-normalize CLS token features for the entire dataset.

        Returns:
            features: (N, D) normalized float tensor on CPU
            labels:   (N,)           int64 tensor on CPU
        """
        model.eval()
        all_features = []
        all_labels = []

        for images, label in data_loader:
            images = images.to(self.device)
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

        similarity = test_features @ train_features.T

        # Pre-sort by descending similarity so each k just slices the top rows.
        # topk on the full matrix is done for max(k) and reused for smaller k.
        k_max = max(self.ks)
        top_distances, top_indices = similarity.topk(k=k_max, dim=-1)

        # (N_test, k_max) temperature-scaled weights, computed once
        top_weights = (top_distances / self.temperature).exp()
        top_neighbor_labels = train_labels[top_indices]

        results = {}

        for k in self.ks:
            weights = top_weights[:, :k]
            neighbor_labels = top_neighbor_labels[:, :k]

            # Creating score matrix (N_test, num_classes), +1 for each neighbor's class, then apply temperature weighting
            scores = torch.zeros(len(test_labels), num_classes)
            scores.scatter_add_(1, neighbor_labels, weights)

            # top-1
            predictions = scores.argmax(dim=1) # (N_test,)
            top1 = (predictions == test_labels).float().mean().item() * 100

            # top-5
            top5 = None
            if num_classes >= 5:
                top5_preds = scores.topk(5, dim=1).indices      # (N_test, 5)
                top5 = (top5_preds == test_labels.unsqueeze(1)).any(dim=1).float().mean().item() * 100

            results[f"knn_top1_k{k}"] = top1
            results[f"knn_top5_k{k}"] = top5
        
        return results

    
    @classmethod
    def from_config(cls, evaluate_config: EvaluationConfig):
        return cls(
            ks=evaluate_config.knn_ks,
            temperature=evaluate_config.knn_temperature,
            batch_size=evaluate_config.knn_batch_size
        )
    

    def plot(self, model: DinoModel, data_loader: DataLoader, save_path: str):
        features, labels = self.extract_features(model, data_loader)
        features = features.numpy()
        labels = labels.numpy()

        # Use t-SNE for dimensionality reduction to 2D
        tsne = TSNE(n_components=2, random_state=0)
        features_2d = tsne.fit_transform(features)

        # Plotting
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE of KNN Features")
        plt.savefig(save_path)
        plt.close()