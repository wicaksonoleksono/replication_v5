import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTripleLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_proxies: int,
        emb_dim: int,
        margin_delta: float = 0.01,
        lambda_: float = 20.0,
        gamma: float = 0.1
    ):
        super(SoftTripleLoss, self).__init__()
        self.num_classes = num_classes
        self.num_proxies = num_proxies
        self.emb_dim = emb_dim
        self.margin_delta = margin_delta
        self.lambda_ = lambda_
        self.gamma = gamma

        self.proxies = nn.Parameter(
            torch.randn(num_classes, num_proxies, emb_dim)
        )
        nn.init.xavier_uniform_(self.proxies)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        split_size = batch_size // 2  # Split into original and augmented pairs

        # Compute similarities for all features
        feat_expanded = features.unsqueeze(1).unsqueeze(2)  # (2B, 1, 1, D)
        proxy_expanded = self.proxies.unsqueeze(0)          # (1, C, K, D)
        dot_products = torch.sum(feat_expanded * proxy_expanded, dim=-1)  # (2B, C, K)
        proxy_weight = F.softmax(dot_products / self.gamma, dim=2)        # (2B, C, K)
        S_ic = torch.sum(proxy_weight * dot_products, dim=2)              # (2B, C)

        # Split into original and augmented parts
        S_ic_orig = S_ic[:split_size]
        S_ic_aug = S_ic[split_size:]
        S_ic_avg = (S_ic_orig + S_ic_aug) / 2  # Average similarities per pair

        # Use original labels (labels are duplicated, take first half)
        labels = labels[:split_size]

        # Compute loss with averaged similarities
        idx = torch.arange(split_size, device=features.device)
        S_correct = S_ic_avg[idx, labels] - self.margin_delta  # (B,)

        S_ic_scaled = self.lambda_ * S_ic_avg  # (B, C)
        S_correct_scaled = self.lambda_ * S_correct  # (B,)

        S_ic_scaled_corrected = S_ic_scaled.clone()
        S_ic_scaled_corrected[idx, labels] = S_correct_scaled
        denominator = torch.logsumexp(S_ic_scaled_corrected, dim=1)  # (B,)
        log_prob_correct = S_correct_scaled - denominator
        loss = -torch.mean(log_prob_correct)
        return loss