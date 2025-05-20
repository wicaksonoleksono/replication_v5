import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


class SentenceTriplet(nn.Module):
    def __init__(self, reducers, use_fallback,
                 beta, d_fn, mine_margin, loss_margin):
        super().__init__()
        self.eps = 1e-6
        self.loss_margin, self.reducers, self.mine_margin = loss_margin, reducers, mine_margin
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn
        lmr = torch.clamp(torch.as_tensor(loss_margin, dtype=torch.float32), -1.0 + self.eps, 1.0 - self.eps)
        mmr = torch.clamp(torch.as_tensor(loss_margin, dtype=torch.float32), -1.0 + self.eps, 1.0 - self.eps)
        self.loss_margin_rad = torch.acos(lmr)
        self.mine_margin_rad = torch.acos(mmr)

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
        cos = self._cosine_sim(x, y)
        phi = (cos - self.loss_margin).clamp(-1+self.eps, 1-self.eps)
        return torch.acos(phi)

    def _additive_angular_distance(self, x, y):
        cos = self._cosine_sim(x, y)
        sin = torch.sqrt(1 - cos.pow(2))
        cos_m, sin_m = torch.cos(self.loss_margin_rad), torch.sin(self.loss_margin_rad)
        phi = (cos * cos_m - sin * sin_m).clamp(-1+self.eps, 1-self.eps)
        return torch.acos(phi)

    def _smoothmax_pooling_reducer(self, loss_terms):
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
            case _:
                raise ValueError(f"Unknown reducer: {self.reducers}")
        return reducers

    def forward(self, og_feat, ag_feat, labels):
        match self.d_fn:
            case "cos":                     d_p, d_n = self._cosine_distance, self._cosine_distance
            case "angular":                 d_p, d_n = self._angular_distance, self._angular_distance
            case "cos_f":                   d_p, d_n = self._additive_cosine_distance, self._angular_distance
            case "ang_f":                   d_p, d_n = self._additive_angular_distance, self._angular_distance
            case _:
                raise ValueError(f"unknown d_fn {self.d_fn}")
        # shitty hyprparam
        use_rad = self.d_fn != "cos"
        margin_loss = self.loss_margin_rad if use_rad else self.loss_margin
        margin_mine = self.mine_margin_rad if use_rad else self.mine_margin
        # ach
        d_ap = d_p(og_feat, ag_feat).diag()
        d_an = d_n(og_feat, og_feat)
        device, B = og_feat.device, og_feat.size(0)
        labels = labels.view(-1)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        valid_neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) & eye_mask
        d_ap_exp = d_ap.unsqueeze(1)
        semi_mask = (d_an > d_ap_exp) & (d_an < d_ap_exp + margin_mine) & valid_neg_mask
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
                    loss_terms = d_ap[valid_hard] - min_d_an_hard[valid_hard] + margin_loss
                else:
                    loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + margin_loss)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())
        if self.reducers.endswith("_sh"):
            loss_terms = d_ap[valid] - min_neg[valid]+self.loss_margin
        else:
            loss_terms = F.relu(d_ap[valid] - min_neg[valid] + self.loss_margin)
        return self._apply_reducer(loss_terms, valid.sum().float())
