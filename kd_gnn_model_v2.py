# -*- coding: utf-8 -*-
"""
DG-PBs-KD V2: confidence-aware + class-balanced knowledge distillation

核心：
- hard loss：真实残基真实标签
- KD loss：只对真实残基
- teacher_confidence：三个教师越一致，单残基KD权重越高
- 为避免负类数量远多于正类、且负类confidence显著更高导致KD被负类主导，
  正/负残基的 confidence-weighted KD 分别求均值，再等权平均。
- DDPM合成节点仍只用正类监督，不参与KD。
"""

import torch
import torch.nn.functional as F
from improved_gnn_model import ImprovedBindingSiteGNN


class ConfidenceAwareKDBindingSiteGNN(ImprovedBindingSiteGNN):
    def __init__(
        self,
        input_dim=1280,
        hidden_dim=256,
        dropout=0.5,
        lambda_kd=0.5,
        lambda_aug=0.2,
        temperature=1.0,
        hard_pos_weight=None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_focal_loss=False,
            pos_weight=1.0,
        )

        self.lambda_kd = float(lambda_kd)
        self.lambda_aug = float(lambda_aug)
        self.temperature_kd = float(temperature)

        if self.temperature_kd <= 0:
            raise ValueError("temperature 必须 > 0")

        if hard_pos_weight is None:
            self.register_buffer("_hard_pos_weight", torch.tensor([]))
        else:
            self.register_buffer(
                "_hard_pos_weight",
                torch.tensor([float(hard_pos_weight)], dtype=torch.float32),
            )

    def _hard_bce(self, logits, targets):
        if logits.numel() == 0:
            return logits.sum() * 0.0

        pos_weight = None
        if self._hard_pos_weight.numel() > 0:
            pos_weight = self._hard_pos_weight.to(logits.device)

        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pos_weight,
        )

    @staticmethod
    def _soften_binary_probability(prob, temperature, eps=1e-6):
        prob = prob.clamp(eps, 1.0 - eps)
        if temperature == 1.0:
            return prob
        teacher_logit = torch.log(prob) - torch.log1p(-prob)
        return torch.sigmoid(teacher_logit / temperature)

    @staticmethod
    def _weighted_mean(losses, weights, eps=1e-8):
        if losses.numel() == 0:
            return None
        weights = weights.clamp(min=0.0)
        denom = weights.sum()
        if float(denom.detach().cpu()) <= eps:
            return losses.mean()
        return (losses * weights).sum() / (denom + eps)

    def compute_kd_loss(self, data, logits=None):
        if logits is None:
            logits = self(data)

        logits = logits.reshape(-1)
        y = data.y.float().reshape(-1)

        if not hasattr(data, "is_real"):
            raise AttributeError("data 缺少 is_real")
        is_real = data.is_real.bool().reshape(-1)
        synthetic_mask = ~is_real

        if logits.numel() != y.numel() or is_real.numel() != y.numel():
            raise ValueError("logits/y/is_real 长度不一致")

        # 1) 真实标签
        hard_loss = self._hard_bce(logits[is_real], y[is_real])

        # 2) confidence-aware + class-balanced KD
        kd_loss = logits.sum() * 0.0
        kd_pos = logits.sum() * 0.0
        kd_neg = logits.sum() * 0.0
        n_teacher = 0
        n_teacher_pos = 0
        n_teacher_neg = 0

        if hasattr(data, "teacher_prob") and hasattr(data, "teacher_confidence"):
            teacher_prob = data.teacher_prob.float().reshape(-1)
            teacher_conf = data.teacher_confidence.float().reshape(-1)

            if teacher_prob.numel() != y.numel():
                raise ValueError("teacher_prob长度不一致")
            if teacher_conf.numel() != y.numel():
                raise ValueError("teacher_confidence长度不一致")

            teacher_mask = (
                is_real
                & torch.isfinite(teacher_prob)
                & torch.isfinite(teacher_conf)
            )
            n_teacher = int(teacher_mask.sum().item())

            if n_teacher > 0:
                T = self.temperature_kd

                q = self._soften_binary_probability(
                    teacher_prob[teacher_mask],
                    temperature=T,
                )
                s = logits[teacher_mask] / T
                yy = y[teacher_mask]
                cc = teacher_conf[teacher_mask].clamp(0.0, 1.0)

                per_node = F.binary_cross_entropy_with_logits(
                    s, q, reduction="none"
                )
                if T != 1.0:
                    per_node = per_node * (T ** 2)

                pos = yy >= 0.5
                neg = ~pos

                n_teacher_pos = int(pos.sum().item())
                n_teacher_neg = int(neg.sum().item())

                pos_loss = self._weighted_mean(per_node[pos], cc[pos])
                neg_loss = self._weighted_mean(per_node[neg], cc[neg])

                pieces = []
                if pos_loss is not None:
                    kd_pos = pos_loss
                    pieces.append(pos_loss)
                if neg_loss is not None:
                    kd_neg = neg_loss
                    pieces.append(neg_loss)

                # 正负类等权，避免负类数量和高confidence共同主导KD
                if pieces:
                    kd_loss = torch.stack(pieces).mean()

        # 3) synthetic positive loss
        aug_loss = logits.sum() * 0.0
        n_synthetic = int(synthetic_mask.sum().item())

        if n_synthetic > 0:
            if not torch.all(y[synthetic_mask] == 1):
                raise ValueError("synthetic节点中存在非正类标签")
            aug_loss = F.binary_cross_entropy_with_logits(
                logits[synthetic_mask],
                torch.ones_like(y[synthetic_mask]),
            )

        total_loss = (
            hard_loss
            + self.lambda_kd * kd_loss
            + self.lambda_aug * aug_loss
        )

        return total_loss, {
            "total": float(total_loss.detach().cpu()),
            "hard": float(hard_loss.detach().cpu()),
            "kd": float(kd_loss.detach().cpu()),
            "kd_pos": float(kd_pos.detach().cpu()),
            "kd_neg": float(kd_neg.detach().cpu()),
            "aug": float(aug_loss.detach().cpu()),
            "n_real": int(is_real.sum().item()),
            "n_teacher": n_teacher,
            "n_teacher_pos": n_teacher_pos,
            "n_teacher_neg": n_teacher_neg,
            "n_synthetic": n_synthetic,
        }


# 兼容命名
KDBindingSiteGNNV2 = ConfidenceAwareKDBindingSiteGNN
