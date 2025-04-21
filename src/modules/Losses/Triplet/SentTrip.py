import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial


class RunningWhiten(nn.Module):
    def __init__(self, dim, ema=0.1, eps=1e-6):
        super().__init__()
        self.register_buffer("cov", torch.eye(dim))
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("L", torch.eye(dim))
        self.ema, self.eps = ema, eps

    @torch.no_grad()
    def update(self, z):
        device = z.device
        if self.mu.device != device:
            self.mu = self.mu.to(device)
            self.cov = self.cov.to(device)
            self.L = self.L.to(device)
        batch_mu = z.mean(0)
        self.mu = (1 - self.ema) * self.mu + self.ema * batch_mu
        dz = z - batch_mu
        batch_cov = dz.T @ dz / max(len(z) - 1, 1)
        self.cov = (1 - self.ema) * self.cov + self.ema * batch_cov
        scale = torch.trace(self.cov) / self.cov.size(0)
        cov_reg = self.cov + self.eps * scale * torch.eye(self.cov.size(0), device=device)
        try:
            L_cov = torch.linalg.cholesky(cov_reg)
        except RuntimeError as e:
            print("Cholesky failed, adding more regularization.")
            cov_reg += 1e-3 * scale * torch.eye(self.cov.size(0), device=device)
            L_cov = torch.linalg.cholesky(cov_reg)
        L_inv = torch.linalg.inv(L_cov.T)
        self.L = L_inv.T

    def whiten(self, z):
        return (z - self.mu) @ self.L.T


class SentenceTriplet(nn.Module):
    def __init__(self, margin, reducers, use_fallback,
                 beta, d_fn, emb_dim, ema=0.1):
        super().__init__()
        self.margin, self.reducers = margin, reducers
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn
        if d_fn in ("maha", "cos_w", "angular_w"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.whitener = RunningWhiten(emb_dim, ema).to(device)

    def _cosine_sim(self, x, y):
        x_norm = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)
        y_norm = y / (torch.norm(y, p=2, dim=1, keepdim=True) + 1e-8)
        return torch.mm(x_norm, y_norm.T)

    def _maha_distance(self, x, y):
        xw, yw = self.whitener.whiten(x), self.whitener.whiten(y)
        return torch.cdist(xw, yw, p=2)

    def _cosine_distance(self, x, y, w):
        if w == True:
            xw, yw = self.whitener.whiten(x), self.whitener.whiten(y)
            sim_matrix = self._cosine_sim(xw, yw)
        else:
            sim_matrix = self._cosine_sim(x, y)
        return 1 - sim_matrix

    def _angular_distance(self, x, y, w):
        if w == True:
            xw, yw = self.whitener.whiten(x), self.whitener.whiten(y)
            sim_matrix = self._cosine_sim(xw, yw)
        else:
            sim_matrix = self._cosine_sim(x, y)
        with torch.no_grad():
            max_sim = sim_matrix.max().item()
            min_sim = sim_matrix.min().item()
            eps = max(1e-6, 0.001 * (max_sim - min_sim))
        safe_sim = torch.clamp(sim_matrix, -1.0 + eps, 1.0 - eps)
        return torch.acos(safe_sim)

    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _sum_reducer(self, loss):
        return loss.sum()

    def _softmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        N = loss_terms.numel()
        pooled_loss = (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_terms)))
        return pooled_loss

    def _apply_reducer(self, loss_terms, valid_count):
        match self.reducers:
            case "mean": reducers = self._mean_reducer(loss_terms, valid_count)
            case "sum": reducers = self._sum_reducer(loss_terms)
            case "softmax": reducers = self._softmax_pooling_reducer(loss_terms)
            case _:
                raise ValueError(f"Unknown reducer: {self.reducers}")
        return reducers

    def forward(self, og_feat, ag_feat, labels):
        if self.d_fn in ("maha", "coswhite", "angwhite"):
            self.whitener.update(torch.cat([og_feat.detach(), ag_feat.detach()], 0))
        match self.d_fn:
            case "cos":                 dist = lambda x, y: self._cosine_distance(x, y, w=False)
            case "angular":             dist = lambda x, y: self._angular_distance(x, y, w=False)
            case "cos_w":               dist = lambda x, y: self._cosine_distance(x, y, w=True)
            case "angular_w":           dist = lambda x, y: self._angular_distance(x, y, w=True)
            case "maha":                dist = self._maha_distance
            case _:
                raise ValueError(f"unknown d_fn {self.d_fn}")
        d_ap = dist(og_feat, ag_feat).diag()           # (B,)
        d_an = dist(og_feat, og_feat)                  # (B,B)

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
                loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())
        loss_terms = F.relu(d_ap[valid] - min_neg[valid] + self.margin)
        return self._apply_reducer(loss_terms, valid.sum().float())

    # def _chord_distance(self, x, y):
    #     theta = self._angular_distance(x, y)
    #     return 2 * torch.sin(theta / 2)

    # def _scaled_chord(self, x, y):
    #     dot = torch.mm(x, y)
    #     # adding non linear profile
    #     scaled = torch.exp(dot / self.margin)
    #     return torch.sqrt(2 - 2 * scaled)
  # def angular_distance(u, v, eps=1e-7):          # 0 … π
    #     cos = (u * v).sum(-1).clamp(-1+eps, 1-eps)
    #     return torch.acos(cos)
