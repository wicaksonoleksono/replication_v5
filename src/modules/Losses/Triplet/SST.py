import torch
import torch.nn as nn
from torch.nn import functional as F


class RunningWhiten(nn.Module):

    def __init__(self, dim, ema=0.01, eps=1e-4):
        super().__init__()
        self.register_buffer("cov", torch.eye(dim))
        self.ema, self.eps = ema, eps           # eps → numerical safety

    def update(self, z):                        # z : (B, D)
        with torch.no_grad():
            mu = z.mean(0, keepdim=True)
            dz = z - mu
            batch_cov = dz.t() @ dz / max(len(z) - 1, 1)
            self.cov.mul_(1 - self.ema).add_(self.ema * batch_cov)

    def whiten(self, z):
        L = torch.linalg.cholesky(torch.linalg.inv(
            self.cov + self.eps * torch.eye(self.cov.size(0),
                                            device=z.device)))
        return z @ L.T


class SST(nn.Module):
    def __init__(self, margin, reducers, use_fallback,
                 beta, d_fn, emb_dim, ema=0.01):
        super().__init__()
        self.margin, self.reducers = margin, reducers
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn
        if d_fn == "maha":
            self.whitener = RunningWhiten(emb_dim, ema)

    def _maha_distance(self, x, y):
        xw, yw = self.whitener.whiten(x), self.whitener.whiten(y)
        return torch.cdist(xw, yw, p=2)

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
            eps = max(1e-6, 0.001 * (max_sim - min_sim))
        safe_sim = torch.clamp(sim_matrix, -1.0 + eps, 1.0 - eps)
        return torch.acos(safe_sim)

    def _chord_distance(self, x, y):
        theta = self._angular_distance(x, y)
        return 2 * torch.sin(theta / 2)

    def _scaled_chord(self, x, y):
        dot = torch.mm(x, y)
        # adding non linear profile
        scaled = torch.exp(dot / self.margin)
        return torch.sqrt(2 - 2 * scaled)

    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _sum_reducer(self, loss):
        return loss.sum()

    def _softmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        N = loss_terms.numel()
        pooled_loss = (1.0 / self.beta) * \
            torch.log(torch.mean(torch.exp(self.beta * loss_terms)))
        return pooled_loss

    def _apply_reducer(self, loss_terms, valid_count):
        match self.reducers:
            case "mean": reducers = self._mean_reducer(loss_terms, valid_count)
            case "sum": reducers = self._sum_reducer(loss_terms)
            case "softmax": reducers = self._softmax_pooling_reducer(loss_terms)
            case _:
                raise ValueError(f"Unknown reducer: {self.reducers}")
        return reducers

    def forward(self, og_feat, ag_feat):
        if self.d_fn == "maha":
            self.whitener.update(torch.cat([og_feat.detach(), ag_feat.detach()], 0))
        match self.d_fn:
            case "cos": dist = self._cosine_distance
            case "chord": dist = self._chord_distance
            case "scaled_chord": dist = self._scaled_chord
            case "maha": dist = self._maha_distance
            case _:
                raise ValueError(f"unknown d_fn {self.d_fn}")
        d_ap = dist(og_feat, ag_feat).diag()           # (B,)
        d_an = dist(og_feat, og_feat)                  # (B,B)

        device, B = og_feat.device, og_feat.size(0)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        valid_neg_mask = eye_mask
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
                loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())
        loss_terms = F.relu(d_ap[valid] - min_neg[valid] + self.margin)
        return self._apply_reducer(loss_terms, valid.sum().float())
