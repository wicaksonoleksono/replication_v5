import torch
from torch import nn
from torch.nn import functional as F
from functools import partial


class SentenceTriplet(nn.Module):
    def __init__(self, reducers, use_fallback,
                 beta, d_fn, margin):
        super().__init__()
        self.eps = 1e-6
        self.margin, self.reducers = margin, reducers
        self.use_fallback, self.beta, self.d_fn = use_fallback, beta, d_fn
        # m_cos = torch.tensor(1.0 - margin, dtype=torch.float32)
        # m_cos = torch.clamp(m_cos, -1.0 + self.eps, 1.0 - self.eps)
        self.margin_rad = margin

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
        phi = (cos - self.margin).clamp(-1+self.eps, 1-self.eps)
        return torch.acos(phi)

    def _additive_angular_distance(self, x, y):
        cos = self._cosine_sim(x, y)
        sin = torch.sqrt(1 - cos.pow(2))
        cos_m, sin_m = torch.cos(self.margin_rad), torch.sin(self.margin_rad)
        phi = (cos * cos_m - sin * sin_m).clamp(-1+self.eps, 1-self.eps)
        return torch.acos(phi)

    def _mean_reducer(self, loss, valid_count):
        return loss.sum() / (valid_count + 1e-7)

    def _smoothmax_pooling_reducer(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        return (1.0 / self.beta) * torch.log(torch.mean(torch.exp(self.beta * loss_terms)))

    def _sm_softmax_sh(self, loss_terms):
        if loss_terms.numel() == 0:
            return torch.tensor(0.0, device=loss_terms.device, dtype=loss_terms.dtype)
        min_val = loss_terms.min()
        shifted = loss_terms - min_val
        weights = F.softmax(shifted, dim=0)
        smooth = (weights * shifted).sum()
        return smooth

    def _sm_softmax(self, loss_terms):
        if loss_terms.numel() == 0:
            return loss_terms.new_tensor(0.0)
        weights = F.softmax(loss_terms, dim=0)
        return (weights * loss_terms).sum(dim=0)

    def _apply_reducer(self, loss_terms, valid_count):
        match self.reducers:
            case "mean": reducers = self._mean_reducer(loss_terms, valid_count)
            case "softmax": reducers = self._smoothmax_pooling_reducer(loss_terms)
            # no beta
            case "freedom_softmax_sh": reducers = self._sm_softmax_sh(loss_terms)
            case "freedom_softmax": reducers = self._sm_softmax(loss_terms)
            case _:
                raise ValueError(f"Unknown reducer: {self.reducers}")
        return reducers

    def forward(self, og_feat, ag_feat, labels):
        match self.d_fn:
            case "cos":                     d_p, d_n = self._cosine_distance, self._cosine_distance
            case "ang":                     d_p, d_n = self._angular_distance, self._angular_distance
            case "cos_f":                   d_p, d_n = self._additive_cosine_distance, self._angular_distance
            case "ang_f":                   d_p, d_n = self._additive_angular_distance, self._angular_distance
            case _:
                raise ValueError(f"unknown d_fn {self.d_fn}")
        use_rad = self.d_fn != "cos"
        mine_m = self.margin_rad if use_rad else self.margin
        if self.d_fn == "cos":
            hinge_m = self.margin
        elif self.d_fn == "ang":
            hinge_m = self.margin_rad
        else:
            hinge_m = 0.0
        d_ap = d_p(og_feat, ag_feat).diag()
        d_an = d_n(og_feat, og_feat)
        device, B = og_feat.device, og_feat.size(0)
        labels = labels.view(-1)
        eye_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        valid_neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)) & eye_mask
        d_ap_exp = d_ap.unsqueeze(1)
        semi_mask = (d_an > d_ap_exp) & (d_an < d_ap_exp + mine_m) & valid_neg_mask
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
                    loss_terms = d_ap[valid_hard] - min_d_an_hard[valid_hard] + hinge_m
                else:
                    loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + hinge_m)
                return self._apply_reducer(loss_terms, valid_hard.sum().float())
        if self.reducers.endswith("_sh"):
            loss_terms = d_ap[valid] - min_neg[valid]+hinge_m
        else:
            loss_terms = F.relu(d_ap[valid] - min_neg[valid] + hinge_m)
        return self._apply_reducer(loss_terms, valid.sum().float())
