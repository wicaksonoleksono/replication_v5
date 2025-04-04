import torch
import torch.nn as nn
import torch.nn.functional as F

class DCL(nn.Module):
    """

    Parameters:
      - temperature: scaling parameter for logits (default: 0.1)
      - weight_fn: optional function to weight the positive term (default: None)
    """
    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def forward(self, z1, z2):
        """
        Computes the one-way DCL loss.
        
        Args:
          - z1: anchor embeddings (e.g. BERT representations), shape [batch_size, dim]
          - z2: positive embeddings (augmented version of the same text), shape [batch_size, dim]
        
        Returns:
          - A scalar loss.
        """
        # Normalize the embeddings (important for cosine similarity)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        # Compute the cross-view similarity between z1 and z2
        sim_matrix = torch.mm(z1, z2.t())  # shape: [batch_size, batch_size]
        pos_sim = torch.diag(sim_matrix)     # positive similarities (each anchor with its positive)
        # Compute the positive term: -sim/temperature (optionally weighted)
        pos_loss = - pos_sim / self.temperature
        if self.weight_fn is not None:
            pos_loss = pos_loss * self.weight_fn(z1, z2)

        # Prepare negatives by concatenating:
        #   a) z1 with itself (we will mask out the diagonal)
        #   b) z1 with z2 (which includes the positive values, but they are already used)
        neg_sim_1 = torch.mm(z1, z1.t())   # shape: [batch_size, batch_size]
        neg_sim_2 = sim_matrix             # shape: [batch_size, batch_size]
        # Concatenate along the feature axis so that for each anchor we have (batch_size * 2) negatives
        neg_sim = torch.cat([neg_sim_1, neg_sim_2], dim=1) / self.temperature

        # Create a mask for the diagonal entries in the first half (self-similarity in neg_sim_1)
        batch_size = z1.size(0)
        mask = torch.zeros_like(neg_sim, dtype=torch.bool)
        mask[:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=z1.device)
        # Instead of adding a finite epsilon, directly set masked values to -infinity
        neg_sim.masked_fill_(mask, -float('inf'))

        # Compute the negative term via logsumexp over the negatives
        neg_loss = torch.logsumexp(neg_sim, dim=1)

        # Total loss (averaged over the batch)
        loss = (pos_loss + neg_loss).mean()
        return loss

# Optional: DCL variant with a weighting function (as in DCLW)
class DCLW(DCL):
    def __init__(self, sigma=0.5, temperature=0.1):
        # The weight function here is one proposed in the original paper.
        # It weights the positive term based on the cosine similarity.
        weight_fn = lambda z1, z2: 2 - z1.size(0) * F.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
        super(DCLW, self).__init__(temperature=temperature, weight_fn=weight_fn)
