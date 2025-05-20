import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
import math


class AngularSentence(nn.Module):
    def __init__(self, margin, reducers, use_fallback,
                 beta, d_fn, ):
        super().__init__()
        self.margin, self.reducers = margin, reducers
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn
        self._smp_l = SmoothMaxPool_learnable(init_beta=self.beta)

    def _additive_angular_distance(self, x, y, eps=1e-7):
        cos_theta = self._cosine_sim(x, y).clamp(-1.0 + eps, 1.0 - eps)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        phi = cos_theta * cos_m - sin_theta * sin_m
        phi = phi.clamp(-1.0 + eps, 1.0 - eps)
        return torch.acos(phi)

    def _softmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        return (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_terms)))

    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _softplus_pool(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        return (1.0/self.beta) * torch.log1p(torch.exp(self.beta * loss_terms).sum())

    def _apply_reducer(self, loss_terms, valid_count):
        match self.reducers:
            case "mean": reducers = self._mean_reducer(loss_terms, valid_count)
            case "softmax": reducers = self._smoothmax_pooling_reducer(loss_terms)
            case "softmax_sh": reducers = self._softplus_pool(loss_terms)
            case "sm_learnable": reducers = self._smp_l(loss_terms)
            case _:
                raise ValueError(f"Unknown reducer: {self.reducers}")
        return reducers

    def forward(self, og_feat, ag_feat, labels):
        match self.d_fn:
            case "angular_f":          dist = self._additive_angular_distance
            case _:
                raise ValueError(f"unknown d_fn {self.d_fn}")
        d_ap = dist(og_feat, ag_feat).diag()
        d_an = dist(og_feat, og_feat)
        device, B = og_feat.device, og_feat.size(0)
        labels = labels.view(-1)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        valid_neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) & eye_mask
        d_ap_exp = d_ap.unsqueeze(1)
        semi_mask = (d_an > d_ap_exp) & (d_an < d_ap_exp + self.margin) & valid_neg_mask
        d_an_semi = torch.where(semi_mask, d_an, torch.full_like(d_an, float('inf')))
        min_neg, _ = torch.min(d_an_semi, 1)
        valid = min_neg < float('inf')
        if not valid.any():
            if not self.use_fallback:
                return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()
            else:
                d_an_hard = torch.where(valid_neg_mask, d_an, torch.full_like(d_an, float('inf')))
                min_d_an_hard, _ = torch.min(d_an_hard, dim=1)
                valid_hard = min_d_an_hard < float('inf')
                if not valid_hard.any():
                    return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()
                if self.reducers.endswith("_sh"):
                    loss_terms = d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin
                else:
                    loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())
        if self.reducers.endswith("_sh"):
            loss_terms = d_ap[valid] - min_neg[valid]+self.margin
        else:
            loss_terms = F.relu(d_ap[valid] - min_neg[valid] + self.margin)
        return self._apply_reducer(loss_terms, valid.sum().float())


class SmoothMaxPool_learnable(nn.Module):
    def __init__(self, init_beta: float = 10.0, eps: float = 1e-6):
        super().__init__()
        init_beta_tensor = torch.tensor(init_beta, dtype=torch.float32)
        self.log_beta = nn.Parameter(torch.log(init_beta_tensor))
        self.eps = eps

    @property
    def beta(self):
        return torch.exp(self.log_beta) + self.eps

    def forward(self, loss_terms: torch.Tensor) -> torch.Tensor:
        if loss_terms.numel() == 0:
            return torch.tensor(0.0,
                                device=loss_terms.device,
                                dtype=loss_terms.dtype)
        b = self.beta
        return (1.0 / b) * torch.log(torch.mean(torch.exp(b * loss_terms)))
