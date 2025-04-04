import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class PASentTrip(nn.Module):
    def __init__(self, margin=0.5, reducers="mean", beta=5, temp=0.1):
        """
        :param margin: Triplet margin
        :param reducers: Which reducer to use: "mean", "sum", or "softmax"
        :param beta: Temperature (or 'beta') for the softmax-pooling reducer
        :param temp: Temperature for the adaptive-mining softmax weighting
        """
        super().__init__()
        self.margin = margin
        self.reducers = reducers
        self.beta = beta
        self.temp = temp  # for adaptive negative weighting

    def _cosine_distance(self, x, y):
        """
        Returns a (B x B) distance matrix between x and y using (clamped) 1 - cos_sim.
        """
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
        y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
        sim_matrix = torch.mm(x_norm, y_norm.T)
        # Cosine distance = clamp(1 - cos_sim) in [0, 2]
        return torch.clamp(1 - sim_matrix, min=0.0, max=2.0)

    def _mean_reducer(self, loss_values, valid_count):
        # Avoid div by zero
        return loss_values.sum() / (valid_count + 1e-7)

    def _sum_reducer(self, loss_values, _):
        return loss_values.sum()

    def _softmax_pooling_reducer(self, loss_values):
        """
        Softmax-pooling over the vector of loss terms (a 'global' pooling).
        L = (1 / beta) * log( mean( exp(beta * L_i) ) )
        """
        if loss_values.numel() == 0:
            return torch.tensor(0.0, device=loss_values.device, dtype=loss_values.dtype)
        return (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_values)))

    def _apply_reducer(self, loss_values, valid_count):
        """Select which aggregator to apply over the per-anchor losses."""
        if self.reducers == "mean":
            return self._mean_reducer(loss_values, valid_count)
        elif self.reducers == "sum":
            return self._sum_reducer(loss_values, valid_count)
        elif self.reducers == "softmax":
            return self._softmax_pooling_reducer(loss_values)
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

        # 1) Distance anchor-positive (only diagonal relevant => anchor i with positive i)
        d_ap = self._cosine_distance(og_feat, ag_feat).diag()  # (batch_size,)

        # 2) Distance anchor-negative: (batch_size, batch_size)
        d_an = self._cosine_distance(og_feat, og_feat)

        # 3) valid_neg_mask[i, j]: True if label[i] != label[j] AND i != j
        labels = labels.view(-1)
        label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))  # shape=(B, B)
        not_self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        valid_neg_mask = label_mask & not_self_mask

        # 4) Compute margin violation matrix: violation[i, j] = d_ap[i] - d_an[i, j] + margin
        violation_matrix = d_ap.unsqueeze(1) - d_an + self.margin

        # 5) Mask invalid negatives => set violation = -inf
        #    (so exp(-inf) => 0 => zero contribution in weighting)
        violation_matrix = torch.where(
            valid_neg_mask,
            violation_matrix,
            torch.tensor(float('-inf'), device=device)
        )

        # --------------------------------------------------------------------
        # ADAPTIVE MINING (soft weighting):
        #
        # For each anchor i, we define a weighting p[i, j] = softmax(violation[i, j] / temp)
        # Then the per-anchor loss L[i] = sum_j p[i, j] * ReLU( violation[i, j] ).
        # We remove any fallback or semi-hard logic entirely.
        # --------------------------------------------------------------------

        # shape: (B, B)
        relu_violation = F.relu(violation_matrix)

        # Softmax over the second dimension (negatives) with temperature = self.temp
        # We do exp(violation / temp) / sum_j exp(violation / temp)
        # Replace -inf with 0 during exponent
        scaled_violation = violation_matrix / self.temp
        weights = torch.exp(scaled_violation)
        # If everything is -inf, exp => 0 => sum=0 => handle that safely
        sum_weights = torch.sum(weights, dim=1, keepdim=True) + 1e-12
        weights = weights / sum_weights  # p[i, j]

        # Weighted sum of ReLU(violation) for each anchor i
        # L[i] = sum_j p[i, j] * ReLU(violation[i, j])
        per_anchor_loss = torch.sum(weights * relu_violation, dim=1)

        # Some anchors might have no valid negatives. Then violation_matrix = -inf => sum=0 => L[i]=0
        # valid anchor => if it has at least 1 valid negative
        valid_anchor_mask = valid_neg_mask.sum(dim=1) > 0
        final_loss_values = per_anchor_loss[valid_anchor_mask]

        # 6) Apply final reducer over anchors
        loss = self._apply_reducer(final_loss_values, valid_anchor_mask.sum().float())
        return loss
