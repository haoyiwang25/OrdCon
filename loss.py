import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MetricComputation(nn.Module):
    """Unified module for computing distance metrics and similarities."""
    
    def __init__(self, metric_config: dict = None):
        super().__init__()
        config = metric_config or {}
        self._label_metric = config.get('label_metric', 'l1')
        self._feature_metric = config.get('feature_metric', 'l2')
    
    def compute_label_distances(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate pairwise label distances."""
        if self._label_metric != 'l1':
            raise NotImplementedError(f"Label metric {self._label_metric} not supported")
        
        # Compute pairwise differences
        expanded_y = y.unsqueeze(1)
        tiled_y = y.unsqueeze(0)
        pairwise_diff = expanded_y - tiled_y
        
        # Sum across label dimensions and compute absolute values
        signed_diff = pairwise_diff.sum(dim=-1)
        magnitude = torch.abs(signed_diff)
        
        return signed_diff, magnitude
    
    def compute_feature_affinity(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise feature similarities."""
        if self._feature_metric != 'l2':
            raise NotImplementedError(f"Feature metric {self._feature_metric} not supported")
        
        # Compute negative L2 distance as similarity
        x_expanded = x.unsqueeze(1)
        x_tiled = x.unsqueeze(0)
        euclidean_dist = torch.norm(x_expanded - x_tiled, p=2, dim=-1)
        return -euclidean_dist


class ContrastiveLoss(nn.Module):
    """Modified contrastive loss with flexible weighting schemes."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Temperature scaling
        self.scaling = kwargs.get('temperature', 2)
        
        # Weight function parameters
        self.apply_soft_weights = kwargs.get('soft', False)
        self.sigmoid_scale = kwargs.get('tau', 1.0)
        
        # Semi-hard mining factors
        self.pos_weight = kwargs.get('semi_pos_factor', 0.0)
        self.neg_weight = kwargs.get('semi_neg_factor', 1.0)
        
        # Initialize metric computer
        metric_config = {
            'label_metric': kwargs.get('label_diff', 'l1'),
            'feature_metric': kwargs.get('feature_sim', 'l2')
        }
        self.metric_computer = MetricComputation(metric_config)
    
    def _sigmoid_weighting(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid-based weighting function."""
        return torch.sigmoid(self.sigmoid_scale * distances)
    
    def _prepare_batch(self, embeddings: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare augmented batch by concatenating views."""
        # Concatenate both views
        combined_embeddings = torch.flatten(embeddings.transpose(0, 1), start_dim=0, end_dim=1)
        # Duplicate targets for both views
        combined_targets = torch.cat([targets, targets], dim=0)
        return combined_embeddings, combined_targets
    
    def _compute_masked_tensors(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Remove diagonal elements from square tensor."""
        identity = torch.eye(batch_size, device=tensor.device)
        off_diagonal_mask = (1 - identity).bool()
        return tensor.masked_select(off_diagonal_mask).view(batch_size, batch_size - 1)
    
    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the contrastive loss.
        
        Args:
            embeddings: Tensor of shape [batch_size, 2, feature_dim]
            targets: Tensor of shape [batch_size, label_dim]
        
        Returns:
            Computed loss value
        """
        # Prepare augmented batch
        flat_embeddings, flat_targets = self._prepare_batch(embeddings, targets)
        batch_size = flat_embeddings.shape[0]
        
        # Compute metrics
        signed_distances, absolute_distances = self.metric_computer.compute_label_distances(flat_targets)
        similarities = self.metric_computer.compute_feature_affinity(flat_embeddings)
        
        # Apply temperature scaling and numerical stability
        scaled_similarities = similarities / self.scaling
        max_sim = scaled_similarities.max(dim=1, keepdim=True)[0]
        stabilized_similarities = scaled_similarities - max_sim.detach()
        exponentials = torch.exp(stabilized_similarities)
        
        # Remove diagonal elements
        similarities_masked = self._compute_masked_tensors(stabilized_similarities, batch_size)
        exponentials_masked = self._compute_masked_tensors(exponentials, batch_size)
        signed_dist_masked = self._compute_masked_tensors(signed_distances, batch_size)
        absolute_dist_masked = self._compute_masked_tensors(absolute_distances, batch_size)
        
        # Compute optional soft weights
        if self.apply_soft_weights:
            distance_weights = self._sigmoid_weighting(absolute_dist_masked)
        else:
            device = absolute_dist_masked.device
            distance_weights = torch.ones_like(absolute_dist_masked, device=device)
        
        # Accumulate loss over positive pairs
        total_loss = 0.0
        num_pairs = batch_size - 1
        
        for pair_idx in range(num_pairs):
            # Extract positive pair information
            positive_similarity = similarities_masked[:, pair_idx]
            positive_distance = absolute_dist_masked[:, pair_idx]
            
            # Build masks for different sample categories
            is_closer = absolute_dist_masked < positive_distance.unsqueeze(-1)
            is_farther = absolute_dist_masked >= positive_distance.unsqueeze(-1)
            is_same_sign = signed_dist_masked < 0
            is_opposite_sign = signed_dist_masked >= 0
            
            # Semi-positive: same sign but closer
            semi_positive = (is_same_sign & is_closer).float()
            semi_positive = semi_positive * self.pos_weight
            
            # Semi-negative: opposite sign but farther
            semi_negative = (is_opposite_sign & is_farther).float()
            semi_negative = semi_negative * self.neg_weight
            
            # Hard negative: same sign but farther
            hard_negative = (is_same_sign & is_farther).float()
            
            # Combine all masks
            combined_mask = semi_positive + semi_negative + hard_negative
            if self.apply_soft_weights:
                combined_mask = combined_mask * distance_weights
            
            # Compute log probability
            denominator = (combined_mask * exponentials_masked).sum(dim=-1)
            log_probability = positive_similarity - torch.log(denominator)
            
            # Add to total loss
            total_loss -= log_probability.sum() / (batch_size * num_pairs)
        
        return total_loss