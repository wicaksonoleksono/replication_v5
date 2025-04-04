# import pandas as pd
# import pickle
# import numpy as np
# import random
# import os
# from transformers import AutoTokenizer
# np.random.seed(0)
# random.seed(0)
# class preprocessor:
#     def __init__(self, 
#                  data_home='dataset/ihc_pure/',
#                  tokenizer_type='bert-base-uncased',
#                  augmentation='partial',
#                  output_dir='preprocessed_data'):
#         self.data_home = data_home
#         self.tokenizer_type = tokenizer_type
#         self.augmentation = augmentation
#         self.output_dir = output_dir
#         self.class2int = {'not_hate': 0, 'implicit_hate': 1}
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
#         os.makedirs(output_dir, exist_ok=True)
        
#     def _process_split(self, datatype):
#         """Process a single data split (train/valid/test)"""
#         datafile = os.path.join(self.data_home, f"{datatype}.tsv")
#         data = pd.read_csv(datafile, sep='\t')
#         labels = data["class"].map(self.class2int).tolist()
#         posts = data["post"].tolist()
#         if datatype == "train" and self.augmentation in ['imp', 'partial']:
#             all_posts = []
#             all_labels = []
#             for i, class_name in enumerate(data["class"]):
#                 all_posts.append(posts[i])
#                 all_labels.append(labels[i])
#                 if self.augmentation == 'full':
#                     aug_text = data["implied_statement"][i] if class_name == 'implicit_hate' \
#                         else data["aug_sent1_of_post"][i]
#                 else: 
#                     aug_text = data["implied_statement"][i] if pd.notna(data["implied_statement"][i]) else None
#                 if aug_text:  # Only add valid augmentations
#                     all_posts.append(aug_text)
#                     all_labels.append(labels[i])
#             # Tokenize all posts together
#             tokenized_posts = self.tokenizer.batch_encode_plus(all_posts).input_ids
#             return {
#                 "tokenized_post": tokenized_posts,
#                 "label": all_labels,
#                 "post": all_posts
#             }
#         else:
#             tokenized_posts = self.tokenizer.batch_encode_plus(posts).input_ids
#             return {
#                 "tokenized_post": tokenized_posts,
#                 "label": labels,
#                 "post": posts
#             }
#     def process(self):
#         data_dict = {}
#         for datatype in ["train", "valid", "test"]:
#             print(f"Processing {datatype} data...")
#             processed_data = self._process_split(datatype)
#             data_dict[datatype] = pd.DataFrame.from_dict(processed_data)
#         output_filename = "ihc_"
#         output_filename += f"{self.augmentation}" if self.augmentation in ['full', 'partial'] else ""
#         output_filename += f"_preprocessed_{self.tokenizer_type.split('-')[0]}.pkl"
#         output_path = os.path.join(self.output_dir, output_filename)
        
#         with open(output_path, 'wb') as f:
#             pickle.dump(data_dict, f)
            
#         print(f"Processing complete. Data saved to {output_path}")
# def test(test_loader, model, batch_size, ce_fn, tracker):
#     model.cuda()
#     model.eval()
#     metrics = Metrics()
#     total_loss = 0.0
#     progress_bar = tqdm(test_loader, desc="Final Testing", unit="batch")
#     with torch.no_grad():
#         for batch in progress_bar:
#             text = batch["post"].cuda()
#             attn = batch["post_attn_mask"].cuda()
#             label = torch.tensor(batch["label"]).long().cuda()
#             og_text, _ = torch.split(text, [batch_size, batch_size], dim=0)
#             og_attn, _ = torch.split(attn, [batch_size, batch_size], dim=0)
#             og_label, _ = torch.split(label, [batch_size, batch_size], dim=0)
#             og_hidden, _ = model.get_cls_features_ptrnsp(og_text, og_attn)
#             pred = model(og_hidden)
#             loss = ce_fn(pred, og_label)
#             total_loss += loss.item()
#             batch_preds = torch.argmax(pred, dim=1).detach()
#             metrics.update(batch_preds, og_label.detach())
#     computed_metrics = metrics.compute()
#     avg_loss = total_loss / len(test_loader)
#     print("\nTest Results:")
#     print(f"Loss: {avg_loss:.4f}")
#     print(f"Accuracy: {computed_metrics['accuracy']:.4f}")
#     print(f"F1 Weighted: {computed_metrics['f1_weighted']:.4f}")
#     import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from torch.nn import init


# class SoftTripleLoss(nn.Module):
#     """
#     A near-complete SoftTriple Loss implementation.

#     Args:
#         la (float):  The lambda factor on the final logits (sometimes called 'la' or 'alpha').
#         gamma (float): Inverse scale factor for sub-center softmax. 
#                        If you pass gamma=0.1, then internal scale = 1/0.1 = 10.
#         tau (float):  The coefficient for the regularizer that penalizes sub-center overlap.
#         margin (float): The margin (delta) subtracted for the ground-truth class logit.
#         dim (int):   The embedding dimension of input features.
#         cN (int):    Number of classes.
#         K (int):     Number of sub-centers per class.
#     """
#     def __init__(self, la, gamma, tau, margin, dim, cN, K):
#         super().__init__()
#         self.la = la               # \lambda in the paper
#         self.gamma = 1. / gamma    # scale factor used inside the sub-center softmax
#         self.tau = tau             # regularizer weight
#         self.margin = margin       # margin (delta)
#         self.cN = cN               # number of classes
#         self.K = K                 # sub-centers per class

#         # Learnable sub-centers: shape [dim, cN*K]
#         #   i.e. each class c has K sub-centers, each of dimension 'dim'
#         self.fc = Parameter(torch.Tensor(dim, cN * K))

#         # We'll create a mask to help compute the sub-center separation reg. 
#         # 'weight_mask' = boolean matrix of shape [cN*K, cN*K].
#         #
#         # For each class c, we want to penalize distance among that class's K sub-centers.
#         # So we set 'True' only for pairs of sub-centers (j, k) that belong to the *same class*.
#         #
#         # Then we'll do something like sqrt(2 - 2 * W^T W)[weight_mask] to get those distances.
#         self.register_buffer("weight_mask", torch.zeros(cN*K, cN*K, dtype=torch.bool))

#         # Fill in the mask
#         for i in range(cN):
#             start = i * K
#             end   = (i + 1) * K
#             # For each sub-center in [start, end), we want True for the sub-centers in the same class
#             # but not the exact same sub-center. 
#             for j in range(start, end):
#                 self.weight_mask[j, (j+1):end] = 1

#         # Initialize fc with Kaiming uniform
#         init.kaiming_uniform_(self.fc, a=math.sqrt(5))

#     def forward(self, features, labels):
#         """
#         Args:
#             features (Tensor): shape [batch_size, dim]
#             labels   (Tensor): shape [batch_size]

#         If you're using augmentations such that you have 2N features
#         (original + augmented), you should also double your labels:
#             labels = torch.cat([labels, labels])

#         Then you can feed them all at once.
#         """
#         device = features.device
#         # 1) Normalize sub-centers (so their magnitude doesn't explode).
#         centers = F.normalize(self.fc, p=2, dim=0)  # shape [dim, cN*K]

#         # 2) Compute similarity matrix: 
#         #    simInd has shape [batch_size, cN*K]
#         simInd = features.matmul(centers)  # [B, cN*K]

#         # 3) Reshape to [B, cN, K]
#         simStruc = simInd.view(-1, self.cN, self.K)

#         # 4) Sub-center softmax across the K sub-centers of each class
#         #    The 'self.gamma' is effectively "scale" for the exponent
#         #    Note that we do: exp( gamma * sim ), then softmax on K
#         prob = F.softmax(simStruc * self.gamma, dim=2)  # [B, cN, K]

#         # 5) Weighted sum across the K sub-centers => per-class logit
#         #    shape: [B, cN]
#         simClass = torch.sum(prob * simStruc, dim=2)

#         # 6) Subtract margin (delta) from the correct class's logit
#         #    We'll create a "margin mask" for each row's ground-truth label
#         marginM = torch.zeros_like(simClass, device=device)
#         marginM[torch.arange(marginM.size(0)), labels] = self.margin
#         # Subtract margin from the correct class
#         simClass = simClass - marginM
#         # 7) Multiply by \lambda (la) and do cross-entropy
#         #    shape: [B, cN]
#         logits = self.la * simClass
#         loss_ce = F.cross_entropy(logits, labels)

#         # 8) Regularizer to push sub-centers of the same class away from each other
#         #    a. compute W^T W => shape [cN*K, cN*K], i.e. pairwise dot-products between sub-centers
#         if self.tau > 0 and self.K > 1:
#             # centers has shape [dim, cN*K], so centers.t() is [cN*K, dim]
#             simCenter = centers.t().matmul(centers)  # [cN*K, cN*K]

#             # The original SoftTriple code often does something like:
#             #   sqrt( 2 + eps - 2 * simCenter[same-class pairs] )
#             # which is effectively the L2 distance between sub-centers in the same class
#             #
#             # We do a triu of self.weight_mask to avoid double-counting.
#             # Also add a small eps (1e-5) for numeric stability.
#             same_class_sims = simCenter[self.weight_mask]
#             # same_class_sims is all the dot-products among sub-centers of the same class 
#             # (above the diagonal). If sub-centers are normalized, the L2 distance^2 = 2 - 2*dot 
#             # => distance = sqrt(2 - 2*dot).
#             dist = torch.sqrt(2.0 + 1e-5 - 2.0 * same_class_sims)

#             # Average them
#             reg = dist.mean()

#             # Weighted by tau
#             loss = loss_ce + self.tau * reg
#         else:
#             loss = loss_ce

#         return loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SoftTripleLoss(nn.Module):
#     """
#     Implements the SoftTriple loss from:
#       Qian et al., "SoftTriple Loss: Revisiting Approximate Metric Learning."
    
#     l_SoftTriple = - (1/N) * sum_i log( exp( lambda*(S'_{i, y_i} - delta) ) 
#                                         / [ exp( lambda*(S'_{i, y_i} - delta) )
#                                             + sum_{j != y_i} exp( lambda*S'_{i,j} ) ] )

#     where
#       S'_{i,c} = \sum_{k in K} ( softmax_{k}( z_i^T w_c^k / gamma ) ) * ( z_i^T w_c^k ).
#     """
#     def __init__(
#         self,
#         num_classes: int,
#         num_proxies: int,
#         emb_dim: int,
#         margin_delta: float = 0.01,
#         lambda_: float = 20.0,
#         gamma: float = 0.1
#     ):
#         """
#         Args:
#             num_classes (int): number of classes
#             num_proxies (int): number of proxies per class
#             emb_dim (int): dimensionality of the input embeddings
#             margin_delta (float): minimum inter-class margin (delta)
#             lambda_ (float): scaling factor for SoftTriple (lambda in formulas)
#             gamma (float): scaling factor for entropy regularizer
#         """
#         super(SoftTripleLoss, self).__init__()
#         self.num_classes = num_classes
#         self.num_proxies = num_proxies
#         self.emb_dim = emb_dim
#         self.margin_delta = margin_delta
#         self.lambda_ = lambda_
#         self.gamma = gamma

#         # Create learnable proxies (class weights).
#         # Shape: (num_classes, num_proxies, emb_dim)
#         self.proxies = nn.Parameter(
#             torch.randn(num_classes, num_proxies, emb_dim)
#         )
#         nn.init.xavier_uniform_(self.proxies)

#     def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             features: (batch_size, emb_dim) input embeddings
#             labels: (batch_size,) ground-truth class labels in [0..num_classes-1]

#         Returns:
#             The scalar SoftTriple loss (averaged over the batch).
#         """
#         # features: (B, D)
#         # proxies: (C, K, D), where C = num_classes, K = num_proxies
#         # labels: (B,)

#         # 1) Compute dot-products z_i^T w_c^k
#         #    We'll shape it to: (B, C, K)
#         #    a) Expand features to (B, 1, 1, D)
#         #    b) Expand proxies to (1, C, K, D)
#         #    c) Then do a dot-product over the last dimension.
#         batch_size = features.shape[0]
#         feat_expanded = features.unsqueeze(1).unsqueeze(2)          # (B, 1, 1, D)
#         proxy_expanded = self.proxies.unsqueeze(0)                 # (1, C, K, D)

#         # Dot product over the last dimension => (B, C, K)
#         dot_products = torch.sum(feat_expanded * proxy_expanded, dim=-1)

#         # 2) For each class c, compute the softmax across the K proxies:
#         #    alpha_{i, c, k} = exp(dot_{i,c,k} / gamma) / sum_{k'} exp(dot_{i,c,k'} / gamma)
#         #    shape: (B, C, K)
#         proxy_weight = F.softmax(dot_products / self.gamma, dim=2)  # (B, C, K)

#         # 3) S'_{i,c} = \sum_{k in K} alpha_{i,c,k} * dot_{i,c,k}
#         #    shape: (B, C)
#         S_ic = torch.sum(proxy_weight * dot_products, dim=2)  # (B, C)

#         # 4) Apply the margin for the correct class, subtract delta from S_{i, y_i}
#         #    for numerical stability when we exponentiate, we'll do it in log-softmax form
#         #    We'll gather S_{i, y_i} and subtract margin_delta
#         #    Then compute the log-softmax across classes with scaling factor lambda_
#         #    We'll do a manual approach:
#         #      log_prob_correct_i = lambda_ * (S_{i,y_i} - delta) - log( sum(exp(lambda_*(S_{i,c'}))) )
#         #
#         #    We'll do this carefully for each sample using an index trick.
#         idx = torch.arange(batch_size, device=features.device)
#         # Extract S_{i,y_i}
#         S_correct = S_ic[idx, labels] - self.margin_delta  # (B,)

#         # Build a log-sum-exp denominator:
#         # scores = lambda * S'_{i,c} for c != y_i
#         # with c == y_i replaced by S_correct
#         S_ic_scaled = self.lambda_ * S_ic  # (B, C)
#         S_correct_scaled = self.lambda_ * S_correct  # (B,)

#         # For the denominator, we want:
#         # sum_{j != y_i} exp(lambda_ * S_{i,j}) + exp(lambda_*(S_{i,y_i} - delta))
#         # We'll replace S_{i,y_i} in S_ic_scaled with S_correct_scaled for each sample
#         S_ic_scaled_corrected = S_ic_scaled.clone()  # (B, C)
#         S_ic_scaled_corrected[idx, labels] = S_correct_scaled
#         # log( sum_c exp( S_ic_scaled_corrected[:, c] ) )
#         denominator = torch.logsumexp(S_ic_scaled_corrected, dim=1)  # (B,)
#         # numerator = S_correct_scaled
#         log_prob_correct = S_correct_scaled - denominator  # (B,)
#         # The loss is the negative average of log_prob_correct
#         loss = -torch.mean(log_prob_correct)
#         return loss


# import torch
# import torch.nn as nn
# from torch.nn import functional as F

# class SentenceTriplet(nn.Module):
#     def __init__(self, margin=0.5, reducers="mean"):
#         """
#         :param margin: Triplet margin
#         :param reducers: Which reducer to use: "mean", "sum", or "adaptive"
#         """
#         super().__init__()
#         self.margin = margin
#         self.reducers = reducers

#     def _cosine_distance(self, x, y):
#         x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
#         y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
#         sim_matrix = torch.mm(x_norm, y_norm.T)
#         # Cosine distance: clamp to [0,2]
#         return torch.clamp(1 - sim_matrix, min=0.0, max=2.0)

#     def _mean_reducer(self, loss, valid_count):
#         return loss.sum() / (valid_count + 1e-7)

#     def _sum_reducer(self, loss, valid_count):
#         return loss.sum()

#     def _adaptive_reducer(self, loss):
   
#         # dynamic=True → compute scaling factor from std(loss)
#         scaling_factor = 1.0 / (torch.std(loss) + 1e-6)  
#         # Exponential weighting
#         weights = torch.exp(scaling_factor * loss)
#         weighted_loss = weights * loss
#         return weighted_loss.mean()

#     def _apply_reducer(self, loss_terms, valid_count):
#         """
#         Chooses which reducer to apply based on self.reducers setting.
#         """
#         if self.reducers == "mean":
#             return self._mean_reducer(loss_terms, valid_count)
#         elif self.reducers == "sum":
#             return self._sum_reducer(loss_terms, valid_count)
#         elif self.reducers == "adaptive":
#             return self._adaptive_reducer(loss_terms)
#         else:
#             raise ValueError(f"Unknown reducer: {self.reducers}")

#     def forward(self, og_feat, ag_feat, labels):
#         device = og_feat.device
#         batch_size = og_feat.size(0)

#         # Distance between anchor and positive
#         d_ap = self._cosine_distance(og_feat, ag_feat).diag()  # diagonal

#         # Distance between anchor and all others
#         d_an = self._cosine_distance(og_feat, og_feat)

#         labels = labels.view(-1)
#         # valid_neg_mask checks for different labels & not self-pair
#         label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))
#         eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
#         valid_neg_mask = label_mask & eye_mask

#         # Semi-hard mining
#         d_ap_expanded = d_ap.unsqueeze(1)
#         semi_hard_mask = (d_an > d_ap_expanded) & \
#                          (d_an < d_ap_expanded + self.margin) & \
#                           valid_neg_mask

#         # Replace invalid or non-semi-hard distances with inf so min() ignores them
#         d_an_semi = torch.where(semi_hard_mask, d_an, torch.full_like(d_an, float('inf')))
#         min_d_an_semi, _ = torch.min(d_an_semi, dim=1)
#         valid_semi = min_d_an_semi < float('inf')

#         # If no semi-hard negatives are found
#         if not valid_semi.any():
#             # Hard negative mining
#             d_an_hard = torch.where(valid_neg_mask, d_an, torch.full_like(d_an, float('inf')))
#             min_d_an_hard, _ = torch.min(d_an_hard, dim=1)
#             valid_hard = min_d_an_hard < float('inf')

#             # If still no valid negatives, return 0
#             if not valid_hard.any():
#                 return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()

#             loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
#             return self._apply_reducer(loss_terms, valid_hard.sum().float())

#         # Compute standard semi-hard negatives
#         loss_terms = F.relu(d_ap[valid_semi] - min_d_an_semi[valid_semi] + self.margin)
#         return self._apply_reducer(loss_terms, valid_semi.sum().float())

# mport torch
# import torch.nn as nn
# from torch.nn import functional as F
# from tqdm import tqdm


# class SentenceTriplet(nn.Module):
#     def __init__(self, margin=0.5,reducers="mean"):
#         super().__init__()
#         self.margin = margin
#         self.reducers=reducers
#     def _cosine_distance(self, x, y):
#         x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
#         y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
#         sim_matrix = torch.mm(x_norm, y_norm.T)
#         return torch.clamp(1- sim_matrix, min=0.0, max=2.0)
#     def _adaptive_reducers(loss):
#         scaling_factor = 1.0/(torch.std(loss)+1e-6)
        
#         return 0
#     def _sum_reducers(loss):
#         return loss.sum()
#     def _mean_reducers(loss):
#         return loss.mean()
        
#     def forward(self, og_feat, ag_feat, labels):
#         device = og_feat.device
#         batch_size = og_feat.size(0)
#         d_ap = self._cosine_distance(og_feat, ag_feat).diag()
#         d_an = self._cosine_distance(og_feat, og_feat)
#         # Create masks
#         labels = labels.view(-1)
#         label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()
#         eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
#         valid_neg_mask = label_mask & eye_mask

#         # Semi-hard mining
#         d_ap_expanded = d_ap.unsqueeze(1)
#         semi_hard_mask = (d_an > d_ap_expanded) & \
#                         (d_an < d_ap_expanded + self.margin) & \
#                         valid_neg_mask
        
#         d_an_semi = torch.where(semi_hard_mask, d_an, torch.full_like(d_an, float('inf')))
#         min_d_an_semi, _ = torch.min(d_an_semi, dim=1)
#         valid_semi = min_d_an_semi < float('inf')
#         # Fallback to hard mining if no semi-hard negatives
#         if not valid_semi.any():
#             # Hard negative mining
#             d_an_hard = torch.where(valid_neg_mask, d_an, torch.full_like(d_an, float('inf')))
#             min_d_an_hard, _ = torch.min(d_an_hard, dim=1)
#             valid_hard = min_d_an_hard < float('inf')
            
#             if not valid_hard.any():
#                 return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()
            
#             loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
#             if self.reducers == "mean":
#                 return loss_terms.sum() / (valid_hard.sum().float() + 1e-7)
#             if self.reducers == "sum":
#                 return loss_terms.sum()
#         # Original semi-hard loss calculation
#         loss_terms = F.relu(d_ap[valid_semi] - min_d_an_semi[valid_semi] + self.margin)
#         if self.reducers == "mean":
#             return loss_terms.sum() / (valid_semi.sum().float() + 1e-7)
#         if self.reducers == "sum":
#             return loss_terms.sum()