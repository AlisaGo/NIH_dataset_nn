from torchvision.ops import sigmoid_focal_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BCEFocalComboLoss(nn.Module):
    def __init__(
        self,
        train_labels,
        device,
        bce_weight=0.75,
        focal_alpha=0.25,
        focal_gamma=2.0,
        pos_weight_type=None,   # None, "sqrt_ratio", "prefer_rarest"
        pos_weight_cap=4.0,
        gap_weight_strength=0.7,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.focal_weight = 1.0 - bce_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.pos_weight_type = pos_weight_type
        self.pos_weight_cap = pos_weight_cap
        self.gap_weight_strength = gap_weight_strength

        self.device = device

        if self.pos_weight_type is not None:
            pos_weight = self._build_pos_weight(train_labels)
        else:
            num_labels = train_labels.shape[1]
            pos_weight = torch.ones(num_labels, dtype=torch.float32, device=self.device)

        self.register_buffer("pos_weight", pos_weight)

    def _build_pos_weight(self, train_labels):
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.detach().cpu().numpy()

        pos = train_labels.sum(axis=0)
        n = train_labels.shape[0]

        pos = np.clip(pos, 1, None)
        neg = n - pos

        if self.pos_weight_type == "prefer_rarest":
            pos_weight = neg / pos
            pos_weight = np.clip(pos_weight, 1.0, None)
            pos_weight = np.log(pos_weight) + 1.0

        elif self.pos_weight_type == "sqrt_ratio":
            pos_weight = np.sqrt(neg / pos)
            pos_weight = np.clip(pos_weight, 1.0, None)

        else:
            raise ValueError(f"Unknown pos_weight_type: {self.pos_weight_type}")

        pos_weight = np.clip(pos_weight, 1.0, self.pos_weight_cap)
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
        return pos_weight

    def update_weights(self, all_probs, all_labels, momentum=0.6):
        if isinstance(all_probs, torch.Tensor):
            all_probs = all_probs.detach().cpu().numpy()
        if isinstance(all_labels, torch.Tensor):
            all_labels = all_labels.detach().cpu().numpy()

        mu_1 = []
        mu_0 = []

        for k in range(all_labels.shape[1]):
            pos_mask = all_labels[:, k] == 1
            neg_mask = all_labels[:, k] == 0

            mu1_k = all_probs[pos_mask, k].mean() if pos_mask.sum() > 0 else 0.0
            mu0_k = all_probs[neg_mask, k].mean() if neg_mask.sum() > 0 else 0.0

            mu_1.append(mu1_k)
            mu_0.append(mu0_k)

        mu_1 = torch.tensor(mu_1, dtype=torch.float32, device=self.device)
        mu_0 = torch.tensor(mu_0, dtype=torch.float32, device=self.device)

        gap = mu_1 - mu_0
        gap = torch.clamp(gap, min=0.0, max=1.0)

        target_weight = 1.0 + self.gap_weight_strength * (1.0 - gap)
        target_weight = torch.clamp(target_weight, min=1.0, max=self.pos_weight_cap)

        self.pos_weight = momentum * self.pos_weight + (1.0 - momentum) * target_weight

    def forward(self, logits, targets):
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=self.pos_weight
        )

        focal = sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none"
        )

        loss = self.bce_weight * bce + self.focal_weight * focal
        return loss.mean()