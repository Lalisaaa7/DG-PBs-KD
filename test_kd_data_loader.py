# -*- coding: utf-8 -*-
"""
只加载 DNA-573 的第1条蛋白，验证 DG-PBs-KD loader 是否真正读入 teacher_prob。
不会开始训练。
"""

import os
import torch

from kd_data_loader import KDProteinDataset


ROOT = os.path.dirname(os.path.abspath(__file__))

raw_file = os.path.join(ROOT, "Raw_data", "DNA-573_Train.txt")
soft_file = os.path.join(
    ROOT,
    "teacher_soft_labels",
    "DNA-573_Train_teacher_soft_labels.csv",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device =", device)
print("raw    =", raw_file)
print("soft   =", soft_file)
print()

ds = KDProteinDataset(
    path=raw_file,
    soft_label_csv=soft_file,
    device=device,
    max_records=1,
    verbose=True,
)

assert len(ds) == 1

d = ds[0]

print("\n========== DG-PBs-KD LOADER CHECK ==========")
print("protein name      :", d.name)
print("residues          :", d.x.size(0))
print("feature dim       :", d.x.size(1))
print("label shape       :", tuple(d.y.shape))
print("teacher shape     :", tuple(d.teacher_prob.shape))
print("is_real shape     :", tuple(d.is_real.shape))
print("has_teacher       :", bool(d.has_teacher.item()))
print("teacher min/max   :", float(d.teacher_prob.min()), float(d.teacher_prob.max()))
print("all finite        :", bool(torch.isfinite(d.teacher_prob).all()))
print("all in [0,1]      :", bool(((d.teacher_prob >= 0) & (d.teacher_prob <= 1)).all()))
print("all real nodes    :", bool(d.is_real.all()))

assert d.x.size(0) == d.y.numel()
assert d.x.size(0) == d.teacher_prob.numel()
assert d.x.size(0) == d.is_real.numel()
assert torch.isfinite(d.teacher_prob).all()
assert ((d.teacher_prob >= 0) & (d.teacher_prob <= 1)).all()
assert d.is_real.all()
assert bool(d.has_teacher.item())

print("\n[PASS] KD数据加载器已成功读取 teacher_soft_prob，可进入KD loss修改。")
