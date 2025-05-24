import torch
import torch.nn as nn
import torch.nn.functional as F


class CamLoss(nn.Module):
    def __init__(self,
                 angular_margin_m,
                 num_classes,
                 embedding_size,
                 p_hyper=None,
                 lambda_a=1.0,
                 lambda_r=1.0,
                 eps=1e-6,
                 dtype=torch.float32):
        super().__init__()
        self.angular_margin_m = angular_margin_m
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.lambda_a = lambda_a
        self.lambda_r = lambda_r
        self.eps = eps
        self.dtype = dtype  # Store the desired dtype

        if p_hyper is not None:
            print(f"Note: Hyperparameter 'p' ({p_hyper}) for Minimum Norm Loss (L_N) is not used "
                  "in this angular CAM version with normalized anchors, as L_N is typically omitted.")

        # Learnable class anchors
        # Corrected tensor creation:
        self.class_anchors = nn.Parameter(torch.empty(num_classes, embedding_size, dtype=self.dtype))
        # Initialize class anchors
        nn.init.kaiming_uniform_(self.class_anchors, a=1)

    def _angular_distance(self, x1_norm, x2_norm):
        # Assuming x1_norm and x2_norm are already normalized and have matching dtypes
        cos_sim = torch.clamp(torch.matmul(x1_norm, x2_norm.T), -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(cos_sim)

    def forward(self, og_feat_norm, ag_feat_norm, labels):
        # Ensure input features are cast to the module's working dtype
        og_feat_norm = og_feat_norm.to(dtype=self.dtype, device=og_feat_norm.device)
        ag_feat_norm = ag_feat_norm.to(dtype=self.dtype, device=ag_feat_norm.device)

        # Ensure labels are on the same device as features (dtype for labels is typically long)
        labels = labels.to(device=og_feat_norm.device)

        # L2 Normalize class anchors (they are already the correct dtype)
        anchors_norm = F.normalize(self.class_anchors, p=2, dim=1)

        # --- 1. Attractor Loss (L_A_ang) ---
        cos_sim_og_all_anchors = torch.matmul(og_feat_norm, anchors_norm.T)
        cos_sim_og_correct_anchor = cos_sim_og_all_anchors.gather(1, labels.unsqueeze(1)).squeeze(1)
        cos_sim_og_correct_clamped = torch.clamp(cos_sim_og_correct_anchor, -1.0 + self.eps, 1.0 - self.eps)
        angular_dist_og_A = torch.acos(cos_sim_og_correct_clamped)
        L_A_og = angular_dist_og_A.mean()

        cos_sim_ag_all_anchors = torch.matmul(ag_feat_norm, anchors_norm.T)
        cos_sim_ag_correct_anchor = cos_sim_ag_all_anchors.gather(1, labels.unsqueeze(1)).squeeze(1)
        cos_sim_ag_correct_clamped = torch.clamp(cos_sim_ag_correct_anchor, -1.0 + self.eps, 1.0 - self.eps)
        angular_dist_ag_A = torch.acos(cos_sim_ag_correct_clamped)
        L_A_ag = angular_dist_ag_A.mean()

        L_A_total = (L_A_og + L_A_ag) / 2.0

        # --- 2. Repeller Loss (L_R_ang) ---
        cos_sim_anchor_anchor = torch.matmul(anchors_norm, anchors_norm.T)
        indices = torch.triu_indices(self.num_classes, self.num_classes, offset=1, device=anchors_norm.device)
        if indices.numel() > 0:
            diff_anchor_cos_sims = cos_sim_anchor_anchor[indices[0], indices[1]]
            diff_anchor_cos_sims_clamped = torch.clamp(diff_anchor_cos_sims, -1.0 + self.eps, 1.0 - self.eps)
            angular_dist_R_pairs = torch.acos(diff_anchor_cos_sims_clamped)
            loss_R_terms = F.relu(self.angular_margin_m - angular_dist_R_pairs)
            L_R_total = loss_R_terms.mean()
        else:
            # Ensure this tensor is created with the correct dtype and device
            L_R_total = torch.tensor(0.0, device=og_feat_norm.device, dtype=self.dtype)

        total_loss = (self.lambda_a * L_A_total) + (self.lambda_r * L_R_total)
        return total_loss
