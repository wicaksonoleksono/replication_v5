import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import math


class SentenceTriplet(nn.Module):
    def __init__(self, margin, reducers, use_fallback,
                 beta, d_fn, ang_margin):
        super().__init__()
        self.margin, self.reducers, self.ang_margin = margin, reducers, ang_margin
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn

    def _cosine_sim(self, x, y):
        return torch.mm(x, y.T)

    def _cosine_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        return 1 - sim_matrix

    def _angular_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        with torch.no_grad():
            max_sim = sim_matrix.max().item()
            min_sim = sim_matrix.min().item()
            eps = max(1e-6, 0.001 * (max_sim - min_sim))
        safe_sim = torch.clamp(sim_matrix, -1.0 + eps, 1.0 - eps)
        return torch.acos(safe_sim)

    def _additive_cosine_distance(self, x, y, eps=1e-7):
        cos_theta = x @ y.T
        cos_theta = cos_theta.clamp(-1.0 + eps, 1.0 - eps)
        phi = cos_theta - self.ang_margin
        phi = phi.clamp(-1.0 + eps, 1.0 - eps)
        return torch.acos(phi)

    def _additive_angular_distance(self, x, y, eps=1e-7):
        cos_theta = x @ y.T
        cos_theta = cos_theta.clamp(-1.0 + eps, 1.0 - eps)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))
        cos_m = math.cos(self.ang_margin)
        sin_m = math.sin(self.ang_margin)
        th = math.cos(math.pi - self.ang_margin)
        mm = math.sin(math.pi - self.ang_margin) * self.ang_margin
        phi = cos_theta * cos_m - sin_theta * sin_m
        phi = torch.where(cos_theta > th, phi, cos_theta - mm)
        return torch.acos(phi)

    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _softmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        return (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_terms)))

    def _softplus_pool(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        return (1.0/self.beta) * torch.log1p(torch.exp(self.beta * loss_terms).sum())

    def _apply_reducer(self, loss_terms, valid_count):
        match self.reducers:
            case "mean": reducers = self._mean_reducer(loss_terms, valid_count)
            case "softmax": reducers = self._softmax_pooling_reducer(loss_terms)
            case "softmax_sh": reducers = self._softplus_pool(loss_terms)
            case _:
                raise ValueError(f"Unknown reducer: {self.reducers}")
        return reducers

    def forward(self, og_feat, ag_feat, labels):
        match self.d_fn:
            case "cos":                dist = self._cosine_distance
            case "angular":            dist = self._angular_distance
            case "angular_f":          dist = self._additive_angular_distance
            case "cos_f":              dist = self._additive_cosine_distance
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
