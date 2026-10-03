# -*- coding: utf-8 -*-
"""
三教师加权 DG-PBs-KD V2：完整573训练 + 129/181独立测试

前提：
1) 三教师 weighted V2 的 lambda_KD 已锁定为 1.2
2) 已完成 seed=42/2024/3407 的正式5折
3) 已先运行 analyze_v2_multiseed_stable_threshold.py
   生成：
   kd_results/V2_lambda1p20_stable_threshold/stable_threshold_summary.json

本脚本：
- 只用 DNA-573 完整训练集训练最终 DDPM + DG-PBs-KD V2
- 使用训练OOF得到的 stable threshold
- 独立测试 DNA-129 / DNA-181
- 测试集不调参、不搜索阈值、不需要 teacher soft label

运行：
    cd "E:\\zhq-ds-main"
    python .\\kd_final_train_test_v2_triple.py --seed 42
"""

import argparse
import csv
import gc
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ddpm_diffusion_model import EnhancedDiffusionModel
from kd_augmentation_v2 import kd_augment_dataset
from kd_data_loader_v2 import KDProteinDataset
from kd_gnn_model_v2 import ConfidenceAwareKDBindingSiteGNN


ROOT = Path(__file__).resolve().parent

TRAIN_RAW = ROOT / "Raw_data" / "DNA-573_Train.txt"
TRAIN_SOFT = (
    ROOT / "teacher_soft_labels_v2"
    / "DNA-573_Train_teacher_soft_labels_v2.csv"
)

TESTS = [
    ("DNA-129_Test", ROOT / "Raw_data" / "DNA-129_Test.txt"),
    ("DNA-181_Test", ROOT / "Raw_data" / "DNA-181_Test.txt"),
]

STABLE_JSON = (
    ROOT / "kd_results"
    / "V2_lambda1p20_stable_threshold"
    / "stable_threshold_summary.json"
)

CV_DIR_TEMPLATE = (
    "DNA-573_Train_5fold_KD_CONFIDENCE_V2_"
    "lkd1p20_laug0p20_T1p00_seed{seed}"
)
CV_SEEDS = [42, 2024, 3407]

LAMBDA_KD = 1.2
LAMBDA_AUG = 0.2
TEMPERATURE = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_locked_settings():
    if not STABLE_JSON.exists():
        raise FileNotFoundError(
            f"找不到稳定阈值文件：{STABLE_JSON}\n"
            "请先运行：python .\\analyze_v2_multiseed_stable_threshold.py"
        )

    with STABLE_JSON.open("r", encoding="utf-8") as f:
        st = json.load(f)

    if abs(float(st["lambda_kd"]) - LAMBDA_KD) > 1e-9:
        raise ValueError(
            f"稳定阈值文件lambda_kd={st['lambda_kd']}，"
            f"但当前最终模型要求{LAMBDA_KD}"
        )

    threshold = float(st["stable_threshold_median"])

    epochs = []
    for seed in CV_SEEDS:
        p = (
            ROOT / "kd_results"
            / CV_DIR_TEMPLATE.format(seed=seed)
            / "cv_summary.json"
        )
        if not p.exists():
            raise FileNotFoundError(f"找不到3-seed CV结果：{p}")
        with p.open("r", encoding="utf-8") as f:
            cv = json.load(f)

        if abs(float(cv["lambda_kd"]) - LAMBDA_KD) > 1e-9:
            raise ValueError(f"{p} 中lambda_kd不一致")
        epochs.append(int(cv["median_best_epoch"]))

    final_epochs = int(statistics.median(epochs))
    return threshold, final_epochs, epochs


def calc_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auroc": float(roc_auc_score(y, p)),
        "aupr": float(average_precision_score(y, p)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_residues": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
    }


def train_final_gnn(
    train_aug,
    device,
    *,
    epochs,
    lr,
    hidden_dim,
    dropout,
):
    model = ConfidenceAwareKDBindingSiteGNN(
        input_dim=1280,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lambda_kd=LAMBDA_KD,
        lambda_aug=LAMBDA_AUG,
        temperature=TEMPERATURE,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total = hard = kd = kd_pos = kd_neg = aug = 0.0
        n_graph = 0

        for data in train_aug:
            d = data.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(d)
            loss, parts = model.compute_kd_loss(d, logits)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"出现非有限loss: {getattr(data, 'name', 'unknown')}"
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total += parts["total"]
            hard += parts["hard"]
            kd += parts["kd"]
            kd_pos += parts.get("kd_pos", 0.0)
            kd_neg += parts.get("kd_neg", 0.0)
            aug += parts["aug"]
            n_graph += 1

        dnm = max(n_graph, 1)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={total/dnm:.4f} "
            f"(hard={hard/dnm:.4f}, kd={kd/dnm:.4f}, "
            f"kd_pos={kd_pos/dnm:.4f}, kd_neg={kd_neg/dnm:.4f}, "
            f"aug={aug/dnm:.4f})"
        )

    return model


def predict_dataset(model, dataset, device):
    model.eval()
    ys, ps, rows = [], [], []

    with torch.no_grad():
        for data in dataset:
            d = data.to(device)
            logits = model(d).reshape(-1)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            label = d.y.detach().cpu().numpy().astype(int)

            pid = str(getattr(data, "original_name", getattr(data, "name", "unknown")))
            seq = getattr(data, "sequence", None)

            for i, (p, y) in enumerate(zip(prob, label), start=1):
                aa = ""
                if seq is not None and i <= len(seq):
                    aa = seq[i - 1]
                rows.append({
                    "protein_id": pid,
                    "residue_index": i,
                    "amino_acid": aa,
                    "label": int(y),
                    "student_prob": float(p),
                })

            ys.extend(label.tolist())
            ps.extend(prob.tolist())

    return np.asarray(ys), np.asarray(ps), rows


def save_predictions(path, rows, threshold):
    fields = [
        "protein_id", "residue_index", "amino_acid",
        "label", "student_prob", "student_pred"
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["student_pred"] = int(float(rr["student_prob"]) >= threshold)
            w.writerow(rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diffusion-epochs", type=int, default=60)
    ap.add_argument("--diffusion-batch-size", type=int, default=64)
    ap.add_argument("--diffusion-T", type=int, default=200)
    ap.add_argument("--gnn-lr", type=float, default=5e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--target-ratio", type=float, default=0.9)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--max-augment-ratio", type=float, default=2.0)
    ap.add_argument("--quality-threshold", type=float, default=0.7)
    ap.add_argument("--diversity-threshold", type=float, default=0.3)
    ap.add_argument("--knn-k", type=int, default=12)
    ap.add_argument("--max-nodes", type=int, default=2000)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    locked_threshold, final_epochs, cv_epochs = load_locked_settings()

    output_dir = (
        ROOT / "kd_results"
        / f"DNA-573_Train_FINAL_KD_CONFIDENCE_V2_TRIPLE_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("FINAL TRIPLE-TEACHER DG-PBs-KD V2")
    print("=" * 84)
    print("teacher             : CLAPE + ESM-NBR + TransBind (weighted V2)")
    print("lambda_KD           :", LAMBDA_KD, "(LOCKED)")
    print("lambda_aug          :", LAMBDA_AUG)
    print("temperature         :", TEMPERATURE)
    print("CV median epochs    :", cv_epochs)
    print("final GNN epochs    :", final_epochs)
    print("stable threshold    :", f"{locked_threshold:.6f}", "(LOCKED)")
    print("threshold source    : DNA-573 OOF only")
    print("independent tuning  : NO")
    print("device              :", device)
    print("=" * 84)

    print("\n[1/5] Load FULL DNA-573 training set")
    train_obj = KDProteinDataset(
        path=str(TRAIN_RAW),
        soft_label_csv=str(TRAIN_SOFT),
        device=str(device),
        knn_k=args.knn_k,
        verbose=False,
    )
    train_data = train_obj.proteins
    print(
        f"[TRAIN] proteins={len(train_data)}, "
        f"residues={sum(int(d.y.numel()) for d in train_data)}"
    )

    # ESM embeddings已经生成；释放loader中的ESM模型显存
    try:
        train_obj.model = train_obj.model.cpu()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[2/5] Train FINAL DDPM on FULL DNA-573")
    ddpm = EnhancedDiffusionModel(
        input_dim=1280,
        T=args.diffusion_T,
        device=device,
    )
    t0 = time.time()
    ddpm.train_on_positive_samples(
        train_data,
        epochs=args.diffusion_epochs,
        batch_size=args.diffusion_batch_size,
    )
    print(f"[DDPM] finished in {(time.time()-t0)/60:.2f} min")

    print("\n[3/5] Augment FULL DNA-573")
    aug_cfg = SimpleNamespace(
        device=device,
        target_ratio=args.target_ratio,
        min_samples_per_protein=args.min_samples,
        max_augment_ratio=args.max_augment_ratio,
        max_nodes_per_graph=args.max_nodes,
        quality_threshold=args.quality_threshold,
        diversity_threshold=args.diversity_threshold,
        knn_k=args.knn_k,
        verbose_loading=False,
    )
    train_aug = kd_augment_dataset(train_data, ddpm, aug_cfg)

    real_nodes = int(sum(int(d.is_real.sum()) for d in train_aug))
    syn_nodes = int(sum(int((~d.is_real).sum()) for d in train_aug))
    print(f"[AUG] real nodes={real_nodes}, synthetic nodes={syn_nodes}")
    print("[AUG] synthetic nodes do NOT participate in KD ✅")

    del ddpm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n[4/5] Train FINAL DG-PBs-KD V2")
    model = train_final_gnn(
        train_aug,
        device,
        epochs=final_epochs,
        lr=args.gnn_lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    model_path = output_dir / "final_dg_pbs_kd_v2_triple.pt"
    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "seed": args.seed,
            "lambda_kd": LAMBDA_KD,
            "lambda_aug": LAMBDA_AUG,
            "temperature": TEMPERATURE,
            "gnn_epochs": final_epochs,
            "locked_threshold": locked_threshold,
            "threshold_source": "3-seed, 15 cross-fitted DNA-573 OOF thresholds median",
            "teacher": "CLAPE + ESM-NBR + TransBind weighted V2",
        },
        model_path,
    )
    print("[SAVE]", model_path)

    del train_aug, train_data, train_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("\n[5/5] INDEPENDENT TEST — NO TUNING")
    all_metrics = []

    for test_name, test_raw in TESTS:
        print("\n" + "-" * 76)
        print("[TEST]", test_name)
        print("threshold:", f"{locked_threshold:.6f}", "(LOCKED)")
        print("-" * 76)

        test_obj = KDProteinDataset(
            path=str(test_raw),
            soft_label_csv=None,
            device=str(device),
            knn_k=args.knn_k,
            verbose=False,
        )
        test_data = test_obj.proteins

        try:
            test_obj.model = test_obj.model.cpu()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        y, p, rows = predict_dataset(model, test_data, device)
        m = calc_metrics(y, p, locked_threshold)
        m["dataset"] = test_name
        all_metrics.append(m)

        save_predictions(
            output_dir / f"{test_name}_predictions.csv",
            rows,
            locked_threshold,
        )
        with (output_dir / f"{test_name}_metrics.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(m, f, indent=2, ensure_ascii=False)

        print(f"AUROC       = {m['auroc']:.4f}")
        print(f"AUPR        = {m['aupr']:.4f}")
        print(f"MCC         = {m['mcc']:.4f}")
        print(f"F1          = {m['f1']:.4f}")
        print(f"Precision   = {m['precision']:.4f}")
        print(f"Sensitivity = {m['sensitivity']:.4f}")
        print(f"Specificity = {m['specificity']:.4f}")
        print(
            f"TP/TN/FP/FN = "
            f"{m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}"
        )

        del test_data, test_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    summary_path = output_dir / "independent_test_summary.csv"
    fields = [
        "dataset", "n_residues", "n_positive", "n_negative",
        "auroc", "aupr", "mcc", "f1", "precision",
        "sensitivity", "specificity", "threshold",
        "tp", "tn", "fp", "fn",
    ]
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in all_metrics:
            w.writerow({k: m[k] for k in fields})

    print("\n" + "=" * 84)
    print("FINAL INDEPENDENT TEST COMPLETE")
    print("=" * 84)
    for m in all_metrics:
        print(
            f"{m['dataset']}: "
            f"AUROC={m['auroc']:.4f}, "
            f"AUPR={m['aupr']:.4f}, "
            f"MCC={m['mcc']:.4f}, "
            f"F1={m['f1']:.4f}"
        )
    print("Saved to:", output_dir)
    print(
        "\n[IMPORTANT] 129/181只用于最终评价，"
        "阈值和训练参数均来自DNA-573内部。"
    )


if __name__ == "__main__":
    main()
