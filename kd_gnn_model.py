# -*- coding: utf-8 -*-
"""
DG-PBs-KD model

目标：
- 保留原 ImprovedBindingSiteGNN 的 GNN 主体结构；
- 仅在训练目标中加入多教师知识蒸馏；
- 真实残基：hard label + teacher soft label；
- DDPM 合成正节点：只使用正类监督，不参与 KD。

第一版默认：
    total_loss = hard_loss + 0.5 * kd_loss + 0.2 * aug_loss
    temperature = 1.0

注意：
- teacher_soft_prob 不做 0/1 阈值化；
- KD 只作用于 is_real=True 且 teacher_prob 有效的真实残基；
- 独立测试时不需要 teacher_prob。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from improved_gnn_model import ImprovedBindingSiteGNN


class KDBindingSiteGNN(ImprovedBindingSiteGNN):
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
        # 保留原 ImprovedBindingSiteGNN 主体架构。
        # use_focal_loss=False 是为了让硬标签损失明确使用 BCE；
        # KD 自身也使用 binary soft-target BCE。
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
            if hard_pos_weight <= 0:
                raise ValueError("hard_pos_weight 必须 > 0")
            self.register_buffer(
                "_hard_pos_weight",
                torch.tensor([float(hard_pos_weight)], dtype=torch.float32),
            )

    def _hard_bce(self, logits, targets):
        """真实残基硬标签 BCE。"""
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
        """
        二分类 soft target 的温度变换：
            q -> logit(q) -> logit(q)/T -> sigmoid
        T=1 时保持原 teacher probability 不变。
        """
        prob = prob.clamp(eps, 1.0 - eps)

        if temperature == 1.0:
            return prob

        teacher_logit = torch.log(prob) - torch.log1p(-prob)
        return torch.sigmoid(teacher_logit / temperature)

    def compute_kd_loss(self, data, logits=None):
        """
        返回：
            total_loss, loss_dict

        data 需要至少具有：
            y
            is_real
        训练集真实图还应具有：
            teacher_prob

        后续 DDPM 增强图：
            原始节点 is_real=True
            合成节点 is_real=False
            合成节点 teacher_prob 可设 NaN
        """
        if logits is None:
            logits = self(data)

        logits = logits.reshape(-1)
        y = data.y.float().reshape(-1)

        if logits.numel() != y.numel():
            raise ValueError(
                f"logits/y 长度不一致: {logits.numel()} != {y.numel()}"
            )

        if not hasattr(data, "is_real"):
            raise AttributeError(
                "data 缺少 is_real；请使用 kd_data_loader.py 或 KD 增强流程。"
            )

        is_real = data.is_real.bool().reshape(-1)

        if is_real.numel() != logits.numel():
            raise ValueError(
                f"is_real/logits 长度不一致: "
                f"{is_real.numel()} != {logits.numel()}"
            )

        synthetic_mask = ~is_real

        # 1) 真实残基硬标签损失
        hard_loss = self._hard_bce(logits[is_real], y[is_real])

        # 2) 真实残基 KD 损失
        kd_loss = logits.sum() * 0.0
        n_teacher = 0

        if hasattr(data, "teacher_prob"):
            teacher_prob = data.teacher_prob.float().reshape(-1)

            if teacher_prob.numel() != logits.numel():
                raise ValueError(
                    f"teacher_prob/logits 长度不一致: "
                    f"{teacher_prob.numel()} != {logits.numel()}"
                )

            teacher_mask = is_real & torch.isfinite(teacher_prob)
            n_teacher = int(teacher_mask.sum().item())

            if n_teacher > 0:
                T = self.temperature_kd
                q = self._soften_binary_probability(
                    teacher_prob[teacher_mask],
                    temperature=T,
                )

                # 学生也使用同一温度。
                student_logits_T = logits[teacher_mask] / T

                kd_loss = F.binary_cross_entropy_with_logits(
                    student_logits_T,
                    q,
                )

                # 与常见 KD 写法一致，温度缩放时乘 T^2。
                if T != 1.0:
                    kd_loss = kd_loss * (T ** 2)

        # 3) DDPM 合成节点：仅按正类监督
        aug_loss = logits.sum() * 0.0
        n_synthetic = int(synthetic_mask.sum().item())

        if n_synthetic > 0:
            # 这里再次检查，避免后续某次增强误把合成节点标成负类。
            if not torch.all(y[synthetic_mask] == 1):
                raise ValueError(
                    "检测到 is_real=False 的合成节点中存在非正类标签。"
                )

            aug_targets = torch.ones_like(y[synthetic_mask])
            aug_loss = F.binary_cross_entropy_with_logits(
                logits[synthetic_mask],
                aug_targets,
            )

        total_loss = (
            hard_loss
            + self.lambda_kd * kd_loss
            + self.lambda_aug * aug_loss
        )

        loss_dict = {
            "total": float(total_loss.detach().cpu()),
            "hard": float(hard_loss.detach().cpu()),
            "kd": float(kd_loss.detach().cpu()),
            "aug": float(aug_loss.detach().cpu()),
            "n_real": int(is_real.sum().item()),
            "n_teacher": n_teacher,
            "n_synthetic": n_synthetic,
        }

        return total_loss, loss_dict
