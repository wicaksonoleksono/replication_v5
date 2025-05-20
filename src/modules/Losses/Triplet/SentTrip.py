import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


class SentenceTriplet(nn.Module):
    def __init__(self, margin, reducers, use_fallback,
                 beta, d_fn,):
        super().__init__()
        self.eps = 1e-6
        self.margin, self.reducers = margin, reducers
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn
        self._smp_l = SmoothMaxPool_learnable(init_beta=self.beta)
        self.margin_rad = torch.clamp(torch.acos(margin), -1.0 + self.eps, 1.0 - self.eps)

    def _cosine_sim(self, x, y):
        return torch.mm(x, y.T)

    def _cosine_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        return 1 - sim_matrix

    def _angular_distance(self, x, y):
        sim_matrix = self._cosine_sim(x, y)
        safe_sim = torch.clamp(sim_matrix, -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(safe_sim)

    def _additive_cosine_distance(self, x, y):
        cos = (x @ y.T).clamp(-1+self.eps, 1-self.eps)
        phi = (cos - self.margin).clamp(-1+self.eps, 1-self.eps)
        return torch.acos(phi)

    def _additive_angular_distance(self, x, y):
        cos = (x@y.T).clamp(-1+self.eps, 1-self.eps)
        sin = torch.sqrt(1-cos.pow(2))
        cos_m = torch.cos(self.margin)
        sin_m = torch.sin(self.margin)
        phi = (cos * cos_m - sin * sin_m).clamp(-1+self.eps, 1-self.eps)
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
            case "cos":                     d_p, d_n = self._cosine_distance, self._cosine_distance
            case "angular":                 d_p, d_n = self._angular_distance, self._angular_distance
            case "cos_f":               d_p, d_n = self._additive_cosine_distance, self._angular_distance
            case "ang_f":           d_p, d_n = self._additive_angular_distance, self._angular_distance
            case _:
                raise ValueError(f"unknown d_fn {self.d_fn}")

        d_ap = d_p(og_feat, ag_feat).diag()
        d_an = d_n(og_feat, og_feat)
        device, B = og_feat.device, og_feat.size(0)
        labels = labels.view(-1)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        valid_neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) & eye_mask
        margin = self.margin_rad if self.d_fn in ["angular", "ang_f", "cos_f"] else self.margin
        # semi-hard negatives
        d_ap_exp = d_ap.unsqueeze(1)
        semi_mask = (d_an > d_ap_exp) & (d_an < d_ap_exp + margin) & valid_neg_mask
        d_an_semi = torch.where(semi_mask, d_an, torch.full_like(d_an, float('inf')))
        min_neg, _ = torch.min(d_an_semi, 1)
        valid = min_neg < float('inf')
        # fallback
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
                    loss_terms = d_ap[valid_hard] - min_d_an_hard[valid_hard] + margin
                else:
                    loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + margin)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())

        if self.reducers.endswith("_sh"):
            loss_terms = d_ap[valid] - min_neg[valid]+margin
        else:
            loss_terms = F.relu(d_ap[valid] - min_neg[valid] + margin)
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

    def forward(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0,
                                device=loss_terms.device,
                                dtype=loss_terms.dtype)
        b = self.beta
        return (1.0 / b) * torch.log(torch.mean(torch.exp(b * loss_terms)))
