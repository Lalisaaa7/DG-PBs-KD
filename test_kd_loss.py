# -*- coding: utf-8 -*-
"""
DG-PBs-KD loss smoke test

只做：
1. 加载 DNA-573 第一条蛋白；
2. 运行一次 DG-PBs-KD 前向传播；
3. 计算 hard loss + KD loss；
4. 做一次 backward，检查梯度是否正常。

不会正式训练，不会生成 DDPM 样本。
"""

import os
import torch

from kd_data_loader import KDProteinDataset
from kd_gnn_model import KDBindingSiteGNN


ROOT = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"

raw_file = os.path.join(ROOT, "Raw_data", "DNA-573_Train.txt")
soft_file = os.path.join(
    ROOT,
    "teacher_soft_labels",
    "DNA-573_Train_teacher_soft_labels.csv",
)

print("device =", device)

ds = KDProteinDataset(
    path=raw_file,
    soft_label_csv=soft_file,
    device=device,
    max_records=1,
    verbose=False,
)

data = ds[0].to(device)

model = KDBindingSiteGNN(
    input_dim=1280,
    hidden_dim=256,
    dropout=0.5,
    lambda_kd=0.5,
    lambda_aug=0.2,
    temperature=1.0,
).to(device)

model.train()

logits = model(data)
loss, parts = model.compute_kd_loss(data, logits)

print("\n========== DG-PBs-KD LOSS CHECK ==========")
print("protein       :", data.name)
print("nodes         :", data.y.numel())
print("logits shape  :", tuple(logits.reshape(-1).shape))
print("loss total    :", f"{parts['total']:.6f}")
print("loss hard     :", f"{parts['hard']:.6f}")
print("loss KD       :", f"{parts['kd']:.6f}")
print("loss aug      :", f"{parts['aug']:.6f}")
print("real nodes    :", parts["n_real"])
print("teacher nodes :", parts["n_teacher"])
print("synthetic     :", parts["n_synthetic"])

assert torch.isfinite(loss)
assert parts["n_real"] == data.y.numel()
assert parts["n_teacher"] == data.y.numel()
assert parts["n_synthetic"] == 0
assert parts["kd"] > 0.0
assert parts["aug"] == 0.0

loss.backward()

grad_ok = False
for p in model.parameters():
    if p.grad is not None and torch.isfinite(p.grad).all():
        grad_ok = True
        break

assert grad_ok

print("backward      : OK")
print("\n[PASS] DG-PBs-KD 的 hard + teacher-soft KD loss 已接通。")
print("下一步：修改 DDPM 增强，使合成节点带 is_real=False，且不参与 KD。")
