import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class AdaptiveSentenceTriplet(nn.Module):
    def __init__(self, margin=0.5, reducers="mean", beta=5):
        """
        :param margin: Triplet margin
        :param reducers: Which reducer to use: "mean", "sum", or "softmax"
        :param beta: Temperature parameter if using softmax pooling
        """
        super().__init__()
        self.margin = margin
        self.reducers = reducers
        self.beta = beta

    def _cosine_distance(self, x, y):
        # Normalize
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
        y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
        # Cosine distance = clamp(1 - cos_sim, 0..2)
        sim_matrix = torch.mm(x_norm, y_norm.T)
        return torch.clamp(1 - sim_matrix, min=0.0, max=2.0)
    
    def _mean_reducer(self, loss, valid_count):
        # Avoid division by zero if valid_count=0
        return loss.sum() / (valid_count + 1e-7)

    def _sum_reducer(self, loss, _):
        return loss.sum()
    
    def _softmax_pooling_reducer(self, loss_terms):
        """
        Softmax-pooling (a.k.a. log-sum-exp trick):
        pooled_loss = (1/β) * log( mean( exp(β * l_i) ) )
        """
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        return (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_terms)))
    
    def _apply_reducer(self, loss_terms, valid_count):
        """Select which reducer to apply."""
        if self.reducers == "mean":
            return self._mean_reducer(loss_terms, valid_count)
        elif self.reducers == "sum":
            return self._sum_reducer(loss_terms, valid_count)
        elif self.reducers == "softmax":
            return self._softmax_pooling_reducer(loss_terms)
        else:
            raise ValueError(f"Unknown reducer: {self.reducers}")

    def forward(self, og_feat, ag_feat, labels):
        """
        :param og_feat: Anchor features (batch_size, embed_dim)
        :param ag_feat: Positive features (batch_size, embed_dim)
        :param labels:  Batch labels, shape (batch_size,)
        """
        device = og_feat.device
        batch_size = og_feat.size(0)

        # Distance anchor-positive for each row i
        d_ap = self._cosine_distance(og_feat, ag_feat).diag()

        # Distance anchor-negative for each row i, all columns j
        d_an = self._cosine_distance(og_feat, og_feat)
        # Create mask: valid_neg_mask[i, j] = True if label[i] != label[j] and i != j
        labels = labels.view(-1)
        label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        valid_neg_mask = label_mask & eye_mask

        # ------------------------
        # "Adaptive" Negative Mining 
        # ------------------------
        # violation(i, j) = d_ap[i] - d_an[i, j] + margin
        violation_matrix = d_ap.unsqueeze(1) - d_an + self.margin
        # Mask out invalid negatives
        # => set them to a large negative so they won't be chosen
        violation_matrix = torch.where(
            valid_neg_mask,
            violation_matrix,
            torch.tensor(-1e9, device=device)
        )
        # For each anchor i, pick the negative j that yields the maximum violation
        max_violation, _ = violation_matrix.max(dim=1)
        # Only keep anchors where max_violation > 0 => they have a positive margin loss
        valid_mask = max_violation > 0
        loss_terms = max_violation[valid_mask]
        # Apply reducer
        return self._apply_reducer(loss_terms, valid_mask.sum().float())