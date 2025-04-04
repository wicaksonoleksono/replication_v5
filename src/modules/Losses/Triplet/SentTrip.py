import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class SentenceTriplet(nn.Module):
    def __init__(self, margin=0.5, reducers="mean", use_fallback=True,beta=5):
        """
        :param margin: Triplet margin
        :param reducers: Which reducer to use: "mean", "sum", or "adaptive"
        :param use_fallback: If True, when no semi-hard negatives are found, 
                             fallback to hard-negative mining; 
                             if False, do not fallback and return 0.
        """
        super().__init__()
        self.margin = margin
        self.reducers = reducers
        self.use_fallback = use_fallback
        self.beta=beta
    def _cosine_distance(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
        y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
        sim_matrix = torch.mm(x_norm, y_norm.T)
        # Cosine distance: clamp to [0,2]
        #  just for safety
        return torch.clamp(1 - sim_matrix, min=0.0, max=2.0)
    
    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _sum_reducer(self, loss):
        return loss.sum()
    
    def _softmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        N = loss_terms.numel()
        # Menggunakan log-sum-exp: (1/β) * log( mean(exp(β * l_i)) ) # bckground nyta apa ya
        pooled_loss = (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_terms)))
        return pooled_loss

    def _apply_reducer(self, loss_terms, valid_count):
        """
        Chooses which reducer to apply based on self.reducers setting.
        """
        if self.reducers == "mean":
            return self._mean_reducer(loss_terms, valid_count)
        elif self.reducers == "sum":
            return self._sum_reducer(loss_terms, valid_count)
        elif self.reducers == "softmax":
            return self._softmax_pooling_reducer(loss_terms)
        else:
            raise ValueError(f"Unknown reducer: {self.reducers}")

    def forward(self, og_feat, ag_feat, labels):
        if self.reducers == "focal" and (math.isnan(self.gamma) or math.isnan(self.alpha)):
            raise ValueError("Gamma and alpha kosong")
        device = og_feat.device
        batch_size = og_feat.size(0)
        # Distance between anchor and positive
        d_ap = self._cosine_distance(og_feat, ag_feat).diag()  # diagonal
        # Distance between anchor and all others
        d_an = self._cosine_distance(og_feat, og_feat)
        labels = labels.view(-1)
        # valid_neg_mask checks for different labels & not self-pair
        label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        valid_neg_mask = label_mask & eye_mask
        # Semi-hard mining
        d_ap_expanded = d_ap.unsqueeze(1)
        semi_hard_mask = (
            (d_an > d_ap_expanded) &
            (d_an < d_ap_expanded + self.margin) &
            valid_neg_mask
        )
        # Replace invalid or non-semi-hard distances with inf so min() ignores them
        d_an_semi = torch.where(semi_hard_mask, d_an, torch.full_like(d_an, float('inf')))
        min_d_an_semi, _ = torch.min(d_an_semi, dim=1)
        valid_semi = min_d_an_semi < float('inf')

        # If no semi-hard negatives are found
        if not valid_semi.any(): # using fallback Faster convergence
            if not self.use_fallback:
                # No fallback: just return 0 if no semi-hard negatives
                return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()
            else:
                # Fallback to hard negative mining
                d_an_hard = torch.where(valid_neg_mask, d_an, torch.full_like(d_an, float('inf')))
                min_d_an_hard, _ = torch.min(d_an_hard, dim=1)
                valid_hard = min_d_an_hard < float('inf')

                # If still no valid negatives, return 0
                if not valid_hard.any():
                    return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()

                loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())

        # Compute standard semi-hard negatives
        loss_terms = F.relu(d_ap[valid_semi] - min_d_an_semi[valid_semi] + self.margin)
        return self._apply_reducer(loss_terms, valid_semi.sum().float())
