# -*- coding: utf-8 -*-
"""
DG-PBs-KD final training + independent test

用途
----
完成训练集内部5折以后：
1) 从 cv_summary.json 读取已经锁定的：
   - lambda_kd
   - lambda_aug
   - temperature
   - median_best_epoch
   - OOF MCC最优阈值
2) 在全部训练集上重新训练最终 DDPM；
3) 只增强完整训练集；
4) 用固定 epoch 训练最终 DG-PBs-KD；
5) 保存最终模型；
6) 使用锁定阈值直接测试独立测试集。
   测试集绝不调阈值、绝不参与DDPM、绝不需要teacher soft label。

573组：
    DNA-573_Train -> DNA-129_Test + DNA-181_Test

646组：
    DNA-646_Train -> DNA-46_Test

运行：
    python .\kd_final_train_test.py --dataset 573

以后完成646五折后：
    python .\kd_final_train_test.py --dataset 646
"""

import argparse
import csv
import gc
import json
import math
import os
import random
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
from kd_augmentation import kd_augment_dataset
from kd_data_loader import KDProteinDataset
from kd_gnn_model import KDBindingSiteGNN


ROOT = Path(__file__).resolve().parent

GROUPS = {
    "573": {
        "train_name": "DNA-573_Train",
        "train_raw": ROOT / "Raw_data" / "DNA-573_Train.txt",
        "train_soft": ROOT / "teacher_soft_labels" / "DNA-573_Train_teacher_soft_labels.csv",
        "cv_dir": ROOT / "kd_results" / "DNA-573_Train_5fold_KD",
        "tests": [
            ("DNA-129_Test", ROOT / "Raw_data" / "DNA-129_Test.txt"),
            ("DNA-181_Test", ROOT / "Raw_data" / "DNA-181_Test.txt"),
        ],
    },
    "646": {
        "train_name": "DNA-646_Train",
        "train_raw": ROOT / "Raw_data" / "DNA-646_Train.txt",
        "train_soft": ROOT / "teacher_soft_labels" / "DNA-646_Train_teacher_soft_labels.csv",
        "cv_dir": ROOT / "kd_results" / "DNA-646_Train_5fold_KD",
        "tests": [
            ("DNA-46_Test", ROOT / "Raw_data" / "DNA-46_Test.txt"),
        ],
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    auroc = float("nan")
    aupr = float("nan")
    if len(np.unique(y)) >= 2:
        auroc = float(roc_auc_score(y, p))
        aupr = float(average_precision_score(y, p))

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "auroc": auroc,
        "aupr": aupr,
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


def predict_dataset(model, dataset, device):
    model.eval()

    all_prob = []
    all_label = []
    rows = []

    with torch.no_grad():
        for data in dataset:
            d = data.to(device)
            logits = model(d).reshape(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = d.y.detach().cpu().numpy().astype(int)

            name = str(getattr(data, "name", "unknown"))
            seq = getattr(data, "sequence", None)

            for i, (p, y) in enumerate(zip(probs, labels), start=1):
                aa = ""
                if seq is not None and i <= len(seq):
                    aa = seq[i - 1]

                rows.append({
                    "protein_id": name,
                    "residue_index": i,
                    "amino_acid": aa,
                    "label": int(y),
                    "student_prob": f"{float(p):.10f}",
                })

            all_prob.extend(probs.tolist())
            all_label.extend(labels.tolist())

    return (
        np.asarray(all_label, dtype=np.int64),
        np.asarray(all_prob, dtype=np.float64),
        rows,
    )


def train_final_gnn(
    train_aug,
    device,
    *,
    epochs,
    lr,
    hidden_dim,
    dropout,
    lambda_kd,
    lambda_aug,
    temperature,
):
    model = KDBindingSiteGNN(
        input_dim=1280,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lambda_kd=lambda_kd,
        lambda_aug=lambda_aug,
        temperature=temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
    )

    for epoch in range(1, epochs + 1):
        model.train()

        total = hard = kd = aug = 0.0
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
            aug += parts["aug"]
            n_graph += 1

        denom = max(n_graph, 1)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={total/denom:.4f} "
            f"(hard={hard/denom:.4f}, "
            f"kd={kd/denom:.4f}, "
            f"aug={aug/denom:.4f})"
        )

    return model


def save_predictions(path, rows, threshold):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "protein_id",
            "residue_index",
            "amino_acid",
            "label",
            "student_prob",
            "student_pred",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            r = dict(row)
            r["student_pred"] = int(float(r["student_prob"]) >= threshold)
            writer.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["573", "646"], required=True)
    ap.add_argument("--seed", type=int, default=42)

    # 与当前 clean-CV 保持一致
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
    cfg = GROUPS[args.dataset]

    cv_summary_path = cfg["cv_dir"] / "cv_summary.json"
    if not cv_summary_path.exists():
        raise FileNotFoundError(
            f"找不到CV结果：{cv_summary_path}\n"
            f"请先完成：python .\\kd_cv_pipeline.py --dataset {args.dataset}"
        )

    with open(cv_summary_path, "r", encoding="utf-8") as f:
        cv = json.load(f)

    lambda_kd = float(cv["lambda_kd"])
    lambda_aug = float(cv["lambda_aug"])
    temperature = float(cv["temperature"])
    locked_threshold = float(cv["oof_threshold_mcc"])
    final_epochs = int(cv["median_best_epoch"])

    if final_epochs <= 0:
        raise ValueError(f"median_best_epoch非法: {final_epochs}")

    output_dir = ROOT / "kd_results" / f"{cfg['train_name']}_FINAL_KD"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DG-PBs-KD FINAL TRAINING")
    print("=" * 80)
    print("train dataset       :", cfg["train_name"])
    print("device              :", device)
    print("lambda_kd           :", lambda_kd)
    print("lambda_aug          :", lambda_aug)
    print("temperature         :", temperature)
    print("fixed GNN epochs    :", final_epochs)
    print("LOCKED threshold    :", f"{locked_threshold:.6f}")
    print("threshold source    : TRAIN OOF ONLY")
    print("output              :", output_dir)
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. 完整训练集读取；有 teacher soft label
    # ------------------------------------------------------------------
    print("\n[1/5] Load FULL TRAINING SET")
    train_obj = KDProteinDataset(
        path=str(cfg["train_raw"]),
        soft_label_csv=str(cfg["train_soft"]),
        device=str(device),
        knn_k=args.knn_k,
        verbose=False,
    )
    train_data = train_obj.proteins

    n_train_res = sum(int(d.y.numel()) for d in train_data)
    print(
        f"[TRAIN] proteins={len(train_data)}, residues={n_train_res}"
    )

    # ------------------------------------------------------------------
    # 2. DDPM在完整训练集上训练
    # ------------------------------------------------------------------
    print("\n[2/5] Train FINAL DDPM on FULL TRAINING SET")
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

    # ------------------------------------------------------------------
    # 3. 只增强完整训练集
    # ------------------------------------------------------------------
    print("\n[3/5] Augment FULL TRAINING SET")
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

    train_aug = kd_augment_dataset(
        train_data,
        ddpm,
        aug_cfg,
    )

    real_nodes = int(sum(int(d.is_real.sum()) for d in train_aug))
    syn_nodes = int(sum(int((~d.is_real).sum()) for d in train_aug))

    print(f"[AUG] real nodes      = {real_nodes}")
    print(f"[AUG] synthetic nodes = {syn_nodes}")
    print("[AUG] synthetic teacher_prob = NaN, no KD ✅")

    # DDPM不再需要
    del ddpm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # 4. 固定epoch训练最终DG-PBs-KD
    # ------------------------------------------------------------------
    print("\n[4/5] Train FINAL DG-PBs-KD")
    print(
        f"[LOCKED] epochs={final_epochs}, "
        f"lambda_kd={lambda_kd}, lambda_aug={lambda_aug}"
    )

    model = train_final_gnn(
        train_aug,
        device,
        epochs=final_epochs,
        lr=args.gnn_lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lambda_kd=lambda_kd,
        lambda_aug=lambda_aug,
        temperature=temperature,
    )

    model_path = output_dir / "final_dg_pbs_kd.pt"
    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "train_dataset": cfg["train_name"],
            "seed": args.seed,
            "lambda_kd": lambda_kd,
            "lambda_aug": lambda_aug,
            "temperature": temperature,
            "gnn_epochs": final_epochs,
            "locked_threshold": locked_threshold,
            "threshold_source": "training OOF only",
            "diffusion_epochs": args.diffusion_epochs,
            "diffusion_T": args.diffusion_T,
            "knn_k": args.knn_k,
        },
        model_path,
    )
    print("[SAVE] final model:", model_path)

    # 训练增强数据可释放
    del train_aug, train_data, train_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # 5. 独立测试；不传soft_label_csv
    # ------------------------------------------------------------------
    print("\n[5/5] INDEPENDENT TEST — NO TUNING")
    all_metrics = []

    for test_name, test_raw in cfg["tests"]:
        print("\n" + "-" * 72)
        print("[TEST]", test_name)
        print("raw      :", test_raw)
        print("threshold:", f"{locked_threshold:.6f}", "(LOCKED)")
        print("-" * 72)

        if not test_raw.exists():
            print(f"[MISSING] 跳过，找不到 {test_raw}")
            continue

        test_obj = KDProteinDataset(
            path=str(test_raw),
            soft_label_csv=None,     # 独立测试绝不需要教师
            device=str(device),
            knn_k=args.knn_k,
            verbose=False,
        )
        test_data = test_obj.proteins

        y, p, rows = predict_dataset(
            model,
            test_data,
            device,
        )

        metrics = calc_metrics(
            y,
            p,
            locked_threshold,
        )
        metrics["dataset"] = test_name
        all_metrics.append(metrics)

        pred_path = output_dir / f"{test_name}_predictions.csv"
        save_predictions(
            pred_path,
            rows,
            locked_threshold,
        )

        with open(
            output_dir / f"{test_name}_metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"proteins    = {len(test_data)}")
        print(f"residues    = {metrics['n_residues']}")
        print(f"positives   = {metrics['n_positive']}")
        print(f"AUROC       = {metrics['auroc']:.4f}")
        print(f"AUPR        = {metrics['aupr']:.4f}")
        print(f"MCC         = {metrics['mcc']:.4f}")
        print(f"F1          = {metrics['f1']:.4f}")
        print(f"Precision   = {metrics['precision']:.4f}")
        print(f"Sensitivity = {metrics['sensitivity']:.4f}")
        print(f"Specificity = {metrics['specificity']:.4f}")
        print(
            f"TP/TN/FP/FN = "
            f"{metrics['tp']}/{metrics['tn']}/"
            f"{metrics['fp']}/{metrics['fn']}"
        )

        del test_obj, test_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 汇总CSV
    if all_metrics:
        summary_csv = output_dir / "independent_test_summary.csv"
        fields = [
            "dataset",
            "n_residues",
            "n_positive",
            "n_negative",
            "auroc",
            "aupr",
            "mcc",
            "f1",
            "precision",
            "sensitivity",
            "specificity",
            "threshold",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        with open(
            summary_csv,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for m in all_metrics:
                writer.writerow({k: m[k] for k in fields})

        print("\n" + "=" * 80)
        print("FINAL INDEPENDENT TEST COMPLETE")
        print("=" * 80)
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
            "\n[IMPORTANT] 上述独立测试指标全部使用训练OOF锁定阈值，"
            "没有在测试集上重新搜索阈值。"
        )
    else:
        print("\n没有找到可测试的独立测试集文件。")


if __name__ == "__main__":
    main()
