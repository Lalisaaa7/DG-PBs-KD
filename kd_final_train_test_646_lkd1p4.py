# -*- coding: utf-8 -*-
"""
DNA-646 -> DNA-46 最终测试（当前先看效果）
固定：
lambda_KD = 1.4
lambda_aug = 0.2
T = 1.0
threshold = 0.2464014143
final GNN epochs = 14

运行：
cd "E:\\zhq-ds-main"
python .\\kd_final_train_test_646_lkd1p4.py --seed 42
"""

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    matthews_corrcoef, f1_score,
    precision_score, recall_score
)

from ddpm_diffusion_model import EnhancedDiffusionModel
from kd_augmentation_v2 import kd_augment_dataset
from kd_data_loader_v2 import KDProteinDataset
from kd_gnn_model_v2 import ConfidenceAwareKDBindingSiteGNN

ROOT = Path(__file__).resolve().parent

TRAIN_RAW = ROOT / "Raw_data" / "DNA-646_Train.txt"
TRAIN_SOFT = ROOT / "teacher_soft_labels_v2" / "DNA-646_Train_teacher_soft_labels_v2.csv"
TEST_RAW = ROOT / "Raw_data" / "DNA-46_Test.txt"

LAMBDA_KD = 1.4
LAMBDA_AUG = 0.2
TEMPERATURE = 1.0
LOCKED_THRESHOLD = 0.2464014143
FINAL_GNN_EPOCHS = 14


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y, p, thr):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= thr).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    spe = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auroc": float(roc_auc_score(y, p)),
        "aupr": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(spe),
        "threshold": float(thr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def train_gnn(train_aug, device, lr=5e-4, hidden_dim=256, dropout=0.5):
    model = ConfidenceAwareKDBindingSiteGNN(
        input_dim=1280,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lambda_kd=LAMBDA_KD,
        lambda_aug=LAMBDA_AUG,
        temperature=TEMPERATURE,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3, betas=(0.9, 0.999)
    )

    for epoch in range(1, FINAL_GNN_EPOCHS + 1):
        model.train()
        total = hard = kd = kd_pos = kd_neg = aug = 0.0
        n_graph = 0

        for data in train_aug:
            d = data.to(device)
            opt.zero_grad(set_to_none=True)

            logits = model(d)
            loss, parts = model.compute_kd_loss(d, logits)

            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            total += parts["total"]
            hard += parts["hard"]
            kd += parts["kd"]
            kd_pos += parts.get("kd_pos", 0.0)
            kd_neg += parts.get("kd_neg", 0.0)
            aug += parts["aug"]
            n_graph += 1

        n = max(n_graph, 1)
        print(
            f"Epoch {epoch:02d}/{FINAL_GNN_EPOCHS} | "
            f"loss={total/n:.4f} "
            f"(hard={hard/n:.4f}, kd={kd/n:.4f}, "
            f"kd_pos={kd_pos/n:.4f}, kd_neg={kd_neg/n:.4f}, "
            f"aug={aug/n:.4f})"
        )

    return model


def predict(model, dataset, device):
    model.eval()
    ys, ps, rows = [], [], []

    with torch.no_grad():
        for data in dataset:
            d = data.to(device)
            logits = model(d).reshape(-1)
            prob = torch.sigmoid(logits).cpu().numpy()
            y = d.y.cpu().numpy().astype(int)

            pid = str(getattr(data, "original_name", getattr(data, "name", "unknown")))
            seq = getattr(data, "sequence", None)

            for i, (pp, yy) in enumerate(zip(prob, y), start=1):
                aa = seq[i-1] if seq is not None and i <= len(seq) else ""
                rows.append({
                    "protein_id": pid,
                    "residue_index": i,
                    "amino_acid": aa,
                    "label": int(yy),
                    "student_prob": float(pp),
                    "student_pred": int(float(pp) >= LOCKED_THRESHOLD),
                })

            ys.extend(y.tolist())
            ps.extend(prob.tolist())

    return np.asarray(ys), np.asarray(ps), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for p in [TRAIN_RAW, TRAIN_SOFT, TEST_RAW]:
        if not p.exists():
            raise FileNotFoundError(f"找不到：{p}")

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = ROOT / "kd_results" / f"DNA-646_Train_FINAL_KD_CONFIDENCE_V2_lkd1p40_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DNA-646 -> DNA-46 FINAL DG-PBs-KD V2")
    print("=" * 80)
    print("lambda_KD =", LAMBDA_KD, "(LOCKED)")
    print("lambda_aug =", LAMBDA_AUG)
    print("T =", TEMPERATURE)
    print("GNN epochs =", FINAL_GNN_EPOCHS)
    print("threshold =", LOCKED_THRESHOLD, "(LOCKED FROM DNA-646 OOF)")
    print("DNA-46 tuning = NO")
    print("device =", device)

    print("\n[1/5] Load full DNA-646")
    train_obj = KDProteinDataset(
        path=str(TRAIN_RAW),
        soft_label_csv=str(TRAIN_SOFT),
        device=str(device),
        knn_k=12,
        verbose=False,
    )
    train_data = train_obj.proteins
    print("train proteins =", len(train_data))
    print("train residues =", sum(int(d.y.numel()) for d in train_data))

    try:
        train_obj.model = train_obj.model.cpu()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[2/5] Train DDPM on full DNA-646")
    ddpm = EnhancedDiffusionModel(input_dim=1280, T=200, device=device)
    t0 = time.time()
    ddpm.train_on_positive_samples(train_data, epochs=60, batch_size=64)
    print(f"[DDPM] finished in {(time.time()-t0)/60:.2f} min")

    print("\n[3/5] Augment full DNA-646")
    cfg = SimpleNamespace(
        device=device,
        target_ratio=0.9,
        min_samples_per_protein=5,
        max_augment_ratio=2.0,
        max_nodes_per_graph=2000,
        quality_threshold=0.7,
        diversity_threshold=0.3,
        knn_k=12,
        verbose_loading=False,
    )

    train_aug = kd_augment_dataset(train_data, ddpm, cfg)
    print("[AUG] real nodes =", sum(int(d.is_real.sum()) for d in train_aug))
    print("[AUG] synthetic nodes =", sum(int((~d.is_real).sum()) for d in train_aug))

    del ddpm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n[4/5] Train final DG-PBs-KD")
    model = train_gnn(train_aug, device)

    model_path = out_dir / "final_dg_pbs_kd_646_lkd1p40.pt"
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "seed": args.seed,
            "lambda_kd": LAMBDA_KD,
            "lambda_aug": LAMBDA_AUG,
            "temperature": TEMPERATURE,
            "gnn_epochs": FINAL_GNN_EPOCHS,
            "locked_threshold": LOCKED_THRESHOLD,
        },
        model_path,
    )
    print("[SAVE]", model_path)

    del train_aug, train_data, train_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n[5/5] DNA-46 INDEPENDENT TEST -- NO TUNING")
    test_obj = KDProteinDataset(
        path=str(TEST_RAW),
        soft_label_csv=None,
        device=str(device),
        knn_k=12,
        verbose=False,
    )
    test_data = test_obj.proteins

    try:
        test_obj.model = test_obj.model.cpu()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y, p, rows = predict(model, test_data, device)
    m = metrics(y, p, LOCKED_THRESHOLD)
    m["dataset"] = "DNA-46_Test"

    with (out_dir / "DNA-46_Test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)

    with (out_dir / "DNA-46_Test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["protein_id", "residue_index", "amino_acid", "label", "student_prob", "student_pred"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 80)
    print("FINAL DNA-46 INDEPENDENT TEST")
    print("=" * 80)
    print(f"AUROC       = {m['auroc']:.4f}")
    print(f"AUPR        = {m['aupr']:.4f}")
    print(f"MCC         = {m['mcc']:.4f}")
    print(f"F1          = {m['f1']:.4f}")
    print(f"Precision   = {m['precision']:.4f}")
    print(f"Sensitivity = {m['sensitivity']:.4f}")
    print(f"Specificity = {m['specificity']:.4f}")
    print(f"TP/TN/FP/FN = {m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}")
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
