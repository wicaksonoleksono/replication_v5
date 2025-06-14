import torch
import torch.nn as nn
# Credits https://github.com/HobbitLong/SupContrast


class AngularCon(nn.Module):
    def __init__(self, temperature):
        super(AngularCon, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]  # 2*N
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            contrast_count = 2
            anchor_count = contrast_count
            assert batch_size % 2 == 0
            mask = torch.eye(batch_size//2, dtype=torch.float32).to(device)
            mask = mask.repeat(anchor_count, contrast_count)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            raise NotImplementedError
        contrast_feature = features
        anchor_feature = contrast_feature
        dot = torch.matmul(anchor_feature, contrast_feature.T)
        dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
        anchor_dot_contrast = -torch.acos(dot) / self.temperature
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()
        return loss
