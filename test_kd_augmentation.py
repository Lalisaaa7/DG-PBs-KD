# -*- coding: utf-8 -*-
"""
只测试 KD 增强元数据，不训练真正 DDPM。

检查：
- 原始节点 is_real=True
- 合成节点 is_real=False
- 合成节点 y=1
- 合成节点 teacher_prob=NaN
- 真实节点 teacher_prob 完整保留
"""

import os
from types import SimpleNamespace

import numpy as np
import torch

from kd_data_loader import KDProteinDataset
from kd_augmentation import kd_augment_dataset


class FakeDiffusion:
    def __init__(self, dim=1280, seed=42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def generate_positive_sample(
        self,
        protein_context,
        num_samples=10,
        verbose=False,
        **kwargs,
    ):
        # 这里只验证增强后的标签/掩码逻辑，不验证DDPM质量。
        return self.rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_samples, self.dim),
        ).astype(np.float32)


ROOT = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = KDProteinDataset(
    path=os.path.join(ROOT, "Raw_data", "DNA-573_Train.txt"),
    soft_label_csv=os.path.join(
        ROOT,
        "teacher_soft_labels",
        "DNA-573_Train_teacher_soft_labels.csv",
    ),
    device=device,
    max_records=1,
    verbose=False,
)

original = ds[0]

config = SimpleNamespace(
    device=device,
    target_ratio=0.9,
    min_samples_per_protein=5,
    max_augment_ratio=2.0,
    max_nodes_per_graph=2000,
    quality_threshold=0.0,     # smoke test全部允许进入
    diversity_threshold=0.0,
    knn_k=7,
    verbose_loading=False,
)

aug = kd_augment_dataset(
    [original],
    FakeDiffusion(dim=original.x.size(1)),
    config,
)[0]

real_mask = aug.is_real
syn_mask = ~real_mask

print("\n========== DG-PBs-KD AUGMENT CHECK ==========")
print("protein            :", aug.name)
print("original nodes     :", int(real_mask.sum()))
print("synthetic nodes    :", int(syn_mask.sum()))
print("total nodes        :", aug.y.numel())
print("real teacher finite:", bool(torch.isfinite(aug.teacher_prob[real_mask]).all()))
print("syn teacher all NaN:", bool(torch.isnan(aug.teacher_prob[syn_mask]).all()))
print("syn labels all 1   :", bool(torch.all(aug.y[syn_mask] == 1)))
print("syn is_real=False  :", bool(torch.all(~aug.is_real[syn_mask])))

assert int(real_mask.sum()) == original.y.numel()
assert int(syn_mask.sum()) > 0
assert torch.isfinite(aug.teacher_prob[real_mask]).all()
assert torch.isnan(aug.teacher_prob[syn_mask]).all()
assert torch.all(aug.y[syn_mask] == 1)
assert aug.x.size(0) == aug.y.numel()
assert aug.teacher_prob.numel() == aug.y.numel()
assert aug.is_real.numel() == aug.y.numel()

print(
    "\n[PASS] DDPM合成节点标记逻辑正确："
    "synthetic不带teacher soft label，也不会参与KD。"
)
print(
    "下一步：改CV训练流程为“先划分fold，再只用train fold训练DDPM并增强”。"
)
