"""DINO loss implementation with centering and sharpening."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from dino.config import LossConfig, AugmentationConfig

logger = logging.getLogger(__name__)


class DinoLoss(nn.Module):
    """
    DINO loss with cross-entropy, temperature scaling, and centering.

    The loss encourages the student to match the teacher's output distribution
    on different augmented views of the same image. Key features:
    - Temperature scaling: Different temperatures for student (sharper) and teacher (softer)
    - Centering: Running mean of teacher outputs to prevent collapse
    - Cross-entropy: Between student log-probabilities and teacher probabilities

    Args:
        out_dim: Output dimension of the projection head
        student_temp: Temperature for student (lower = sharper, typical: 0.1)
        teacher_temp: Temperature for teacher (lower = more confident, typical: 0.04)
        center_momentum: EMA momentum for centering (typical: 0.9)
        num_global_views: Number of global crops (typically 2)
        num_total_views: Total number of crops (global + local, typically 8)
    """

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        num_global_views: int = 2,
        num_total_views: int = 8
    ):
        super().__init__()

        if student_temp <= 0 or teacher_temp <= 0:
            raise ValueError("Temperatures must be positive")
        if student_temp <= teacher_temp:
            logger.warning(
                f"Student temperature ({student_temp}) should typically be "
                f"higher than teacher temperature ({teacher_temp})"
            )

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.num_global_views = num_global_views
        self.num_total_views = num_total_views
        self.center_momentum = center_momentum

        # Register a buffer to store the center for teacher output normalization
        # Buffer is not a parameter, but will be saved in state_dict
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DINO loss.

        Args:
            student_outputs: Student outputs of shape [num_total_views * batch_size, out_dim]
                            Contains outputs for all views (global + local)
            teacher_outputs: Teacher outputs of shape [num_global_views * batch_size, out_dim]
                            Contains outputs only for global views

        Returns:
            Scalar loss value

        Note:
            The loss is computed as average cross-entropy between:
            - Each teacher view and each student view (excluding same view pairs)
            This creates (num_global_views * num_total_views - num_global_views) loss terms
        """
        # Apply temperature scaling and compute log-probabilities for student
        student_logits = student_outputs / self.student_temp
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # Chunk into individual views
        student_log_probs_chunked = student_log_probs.chunk(self.num_total_views)

        # Apply centering and temperature scaling for teacher
        teacher_logits = (teacher_outputs - self.center) / self.teacher_temp
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Detach teacher probabilities (no gradients through teacher)
        teacher_probs = teacher_probs.detach()

        # Chunk into individual views
        teacher_probs_chunked = teacher_probs.chunk(self.num_global_views)

        # Compute cross-entropy loss between all view pairs
        total_loss = 0.0
        n_loss_terms = 0

        # Use the chunked probability distributions, not raw outputs
        for i, teacher_prob in enumerate(teacher_probs_chunked):
            for j, student_log_prob in enumerate(student_log_probs_chunked):
                # Skip when comparing same views
                if i == j:
                    continue

                # Cross-entropy: -sum(p * log_q)
                # Shape: [batch_size, out_dim] -> [batch_size] -> scalar
                loss = -torch.sum(teacher_prob * student_log_prob, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        # Average over all loss terms
        total_loss /= n_loss_terms

        # Update the center with EMA
        self.update_center(teacher_outputs)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor):
        """
        Update the center with exponential moving average.

        The center is the running mean of teacher outputs. Subtracting the center
        prevents the model from collapsing to a trivial solution.

        Args:
            teacher_output: Raw teacher outputs [num_global_views * batch_size, out_dim]
        """
        # Compute batch center (mean across batch dimension)
        batch_center = teacher_output.mean(dim=0, keepdim=True)  # [1, out_dim]

        # EMA update: new_center = momentum * old_center + (1 - momentum) * batch_center
        self.center = self.center * self.center_momentum + batch_center * (1.0 - self.center_momentum)


    def get_center(self) -> torch.Tensor:
        """Get the current center value."""
        return self.center.clone()

    def reset_center(self):
        """Reset the center to zeros."""
        self.center.zero_()

    @classmethod
    def from_config(
        cls,
        loss_config: LossConfig,
        augmentation_config: AugmentationConfig,
        out_dim: int
    ) -> DinoLoss:
        """
        Create a DinoLoss from configuration dataclasses.

        Args:
            loss_config: Loss configuration dataclass
            augmentation_config: Augmentation configuration dataclass
            out_dim: Output dimension of the projection head

        Returns:
            Configured DinoLoss instance
        """
        return cls(
            out_dim=out_dim,
            student_temp=loss_config.student_temp,
            teacher_temp=loss_config.teacher_temp,
            center_momentum=loss_config.center_momentum,
            num_global_views=augmentation_config.num_global_views,
            num_total_views=augmentation_config.num_total_views
        )

    def __repr__(self) -> str:
        return (
            f"DinoLoss("
            f"student_temp={self.student_temp}, "
            f"teacher_temp={self.teacher_temp}, "
            f"center_momentum={self.center_momentum}, "
            f"num_global_views={self.num_global_views}, "
            f"num_total_views={self.num_total_views})"
        )
