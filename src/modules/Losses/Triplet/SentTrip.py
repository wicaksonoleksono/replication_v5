import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class SentenceTriplet(nn.Module):
    def __init__(self, margin, reducers, use_fallback, beta, d_fn):
        super().__init__()
        self.margin = margin
        self.reducers = reducers
        self.use_fallback = use_fallback
        self.beta = beta
        self.d_fn = d_fn

    def _cosine_sim(self, x, y):
        x_norm = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
        return torch.mm(x_norm, y_norm.T)

    def _cosine_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        return 1 - sim_matrix

    def _angular_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        with torch.no_grad():
            max_sim = sim_matrix.max().item()
            min_sim = sim_matrix.min().item()
            # Pakai scaling dinamis karena derivasi accos mendekati infinite jika mendekati -1 atau 1 .
            eps = max(1e-6, 0.001 * (max_sim - min_sim))
        safe_sim = torch.clamp(sim_matrix, -1.0 + eps, 1.0 - eps)
        return torch.acos(safe_sim)

    def _correlation_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        return torch.sqrt((1 - sim_matrix)/2)

    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _sum_reducer(self, loss):
        return loss.sum()

    def _softmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        N = loss_terms.numel()
        # Menggunakan log-sum-exp: (1/β) * log( mean(exp(β * l_i)) ) # bckground nyta apa ya
        pooled_loss = (1.0 / self.beta) * \
            torch.log(torch.mean(torch.exp(self.beta * loss_terms)))
        return pooled_loss

    def _apply_reducer(self, loss_terms, valid_count):
        if self.reducers == "mean":
            return self._mean_reducer(loss_terms, valid_count)
        elif self.reducers == "sum":
            return self._sum_reducer(loss_terms)
        elif self.reducers == "softmax":
            return self._softmax_pooling_reducer(loss_terms)

        else:
            raise ValueError(f"Unknown reducer: {self.reducers}")

    def forward(self, og_feat, ag_feat, labels):
        device = og_feat.device
        batch_size = og_feat.size(0)
        if self.d_fn == "cos":
            # Distance between anchor and positive
            d_ap = self._cosine_distance(og_feat, ag_feat).diag()
            # Distance between anchor and all others
            d_an = self._cosine_distance(og_feat, og_feat)
        elif self.d_fn == "angular":
            d_ap = self._angular_distance(og_feat, ag_feat).diag()  # diagonal
            d_an = self._angular_distance(og_feat, og_feat)
        elif self.d_fn == "correlation":
            d_ap = self._correlation_distance(
                og_feat, ag_feat).diag()  # diagonal
            d_an = self._correlation_distance(og_feat, og_feat)

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
        d_an_semi = torch.where(semi_hard_mask, d_an,
                                torch.full_like(d_an, float('inf')))
        min_d_an_semi, _ = torch.min(d_an_semi, dim=1)
        valid_semi = min_d_an_semi < float('inf')

        # If no semi-hard negatives are found
        if not valid_semi.any():  # using fallback Faster convergence
            if not self.use_fallback:
                # No fallback: just return 0 if no semi-hard negatives
                return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()
            else:
                # Fallback to hard negative mining
                d_an_hard = torch.where(
                    valid_neg_mask, d_an, torch.full_like(d_an, float('inf')))
                min_d_an_hard, _ = torch.min(d_an_hard, dim=1)
                valid_hard = min_d_an_hard < float('inf')
                # If still no valid negatives, return 0
                if not valid_hard.any():
                    return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()

                loss_terms = F.relu(
                    d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())

        # Compute standard semi-hard negatives
        loss_terms = F.relu(d_ap[valid_semi] -
                            min_d_an_semi[valid_semi] + self.margin)
        return self._apply_reducer(loss_terms, valid_semi.sum().float())
