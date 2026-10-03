# -*- coding: utf-8 -*-
"""
DG-PBs-KD clean 5-fold CV pipeline v2

核心原则
--------
1. 先按蛋白划分 fold；
2. 每一折 DDPM 只看当前 train fold；
3. 只增强当前 train fold；
4. val fold 永远保持原始真实残基，不做增强；
5. KD 只作用于真实残基，并按 teacher_confidence 加权、正负类平衡；
6. early stopping 使用 val AUPR，不在独立测试集调任何参数；
7. 5折结束后用 OOF 概率在训练集内部确定最终分类阈值。

运行（先做小规模冒烟测试）：
    python .\kd_cv_confidence_v2.py --dataset 573 --smoke

正式 DNA-573 5折：
    python .\kd_cv_confidence_v2.py --dataset 573

正式 DNA-646 5折：
    python .\kd_cv_confidence_v2.py --dataset 646
"""

import argparse
import copy
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
from sklearn.model_selection import KFold

from ddpm_diffusion_model import EnhancedDiffusionModel
from kd_augmentation_v2 import kd_augment_dataset
from kd_data_loader_v2 import KDProteinDataset
from kd_gnn_model_v2 import ConfidenceAwareKDBindingSiteGNN


ROOT = Path(__file__).resolve().parent

DATASET_CFG = {
    "573": {
        "name": "DNA-573_Train",
        "raw": ROOT / "Raw_data" / "DNA-573_Train.txt",
        "soft": ROOT / "teacher_soft_labels_v2" / "DNA-573_Train_teacher_soft_labels_v2.csv",
    },
    "646": {
        "name": "DNA-646_Train",
        "raw": ROOT / "Raw_data" / "DNA-646_Train.txt",
        "soft": ROOT / "teacher_soft_labels_v2" / "DNA-646_Train_teacher_soft_labels_v2.csv",
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_probs(model, dataset, device):
    model.eval()
    probs, labels = [], []
    protein_ids, residue_indices = [], []

    with torch.no_grad():
        for data in dataset:
            d = data.to(device)
            logits = model(d).reshape(-1)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            y = d.y.detach().cpu().numpy().astype(int)

            probs.extend(p.tolist())
            labels.extend(y.tolist())

            name = getattr(data, "original_name", getattr(data, "name", "unknown"))
            protein_ids.extend([str(name)] * len(y))
            residue_indices.extend(list(range(1, len(y) + 1)))

    return {
        "prob": np.asarray(probs, dtype=np.float64),
        "label": np.asarray(labels, dtype=np.int64),
        "protein_id": protein_ids,
        "residue_index": residue_indices,
    }


def ranking_metrics(y, p):
    out = {"auroc": float("nan"), "aupr": float("nan")}
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["aupr"] = float(average_precision_score(y, p))
    return out


def find_best_mcc_threshold(y, p):
    """
    在训练集 OOF 预测上精确寻找 MCC 最优阈值。
    预测规则：p >= threshold -> 1
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]

    total_pos = int(ys.sum())
    total_neg = int(len(ys) - total_pos)

    tp = fp = 0
    best_mcc = -1.0
    best_thr = 0.5

    i = 0
    n = len(ys)
    while i < n:
        score = ps[i]
        j = i
        add_tp = add_fp = 0

        while j < n and ps[j] == score:
            if ys[j] == 1:
                add_tp += 1
            else:
                add_fp += 1
            j += 1

        tp += add_tp
        fp += add_fp
        fn = total_pos - tp
        tn = total_neg - fp

        denom = math.sqrt(
            max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0)
        )
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(score)

        i = j

    return best_thr, float(best_mcc)


def hard_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(threshold),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_kd_model(
    train_aug,
    val_fold,
    device,
    *,
    hidden_dim,
    dropout,
    lr,
    epochs,
    patience,
    lambda_kd,
    lambda_aug,
    temperature,
):
    model = ConfidenceAwareKDBindingSiteGNN(
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.7,
        patience=3,
    )

    best_state = None
    best_val_aupr = -1.0
    best_epoch = 0
    no_improve = 0

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
                    f"训练出现非有限loss: protein={getattr(d, 'name', 'unknown')}"
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

        val = collect_probs(model, val_fold, device)
        rank = ranking_metrics(val["label"], val["prob"])
        val_aupr = rank["aupr"]

        scheduler.step(val_aupr if np.isfinite(val_aupr) else 0.0)

        denom = max(n_graph, 1)
        print(
            f"    Epoch {epoch:02d}/{epochs} | "
            f"loss={total/denom:.4f} "
            f"(hard={hard/denom:.4f}, kd={kd/denom:.4f}, "
            f"kd_pos={kd_pos/denom:.4f}, kd_neg={kd_neg/denom:.4f}, "
            f"aug={aug/denom:.4f}) | "
            f"Val AUPR={val_aupr:.4f} AUROC={rank['auroc']:.4f}"
        )

        if np.isfinite(val_aupr) and val_aupr > best_val_aupr + 1e-8:
            best_val_aupr = float(val_aupr)
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"    Early stopping: epoch={epoch}")
            break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }
        best_epoch = epoch

    model.load_state_dict(best_state)
    model.to(device)

    return model, best_epoch, best_val_aupr


def write_oof_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "fold",
            "protein_id",
            "residue_index",
            "label",
            "student_prob",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["573", "646"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--soft-label-csv", type=str, default=None)
    ap.add_argument("--variant-name", type=str, default="triple_weighted")

    # 正式训练参数：尽量沿用原 DG-PBs 设置
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--diffusion-epochs", type=int, default=60)
    ap.add_argument("--diffusion-batch-size", type=int, default=64)
    ap.add_argument("--diffusion-T", type=int, default=200)

    ap.add_argument("--gnn-epochs", type=int, default=30)
    ap.add_argument("--gnn-lr", type=float, default=5e-4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)

    ap.add_argument("--lambda-kd", type=float, default=0.5)
    ap.add_argument("--lambda-aug", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=1.0)

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
    cfg = DATASET_CFG[args.dataset].copy()
    if args.soft_label_csv:
        cfg["soft"] = Path(args.soft_label_csv)
    if not cfg["soft"].exists():
        raise FileNotFoundError(f"找不到soft label文件：{cfg['soft']}")

    def _tag_float(x):
        # 0.10 -> 0p10，避免文件夹名中小数点过多
        return f"{float(x):.2f}".replace(".", "p")

    run_tag = (
        f"lkd{_tag_float(args.lambda_kd)}"
        f"_laug{_tag_float(args.lambda_aug)}"
        f"_T{_tag_float(args.temperature)}"
        f"_seed{args.seed}"
    )

    if args.smoke:
        n_splits = 2
        max_records = 12
        diffusion_epochs = 1
        gnn_epochs = 1
        patience = 1
        out_name = f"{cfg['name']}_SMOKE_KD_ABLATION_{args.variant_name}_{run_tag}"
        print("\n[SMOKE MODE] 只跑前12条蛋白、2折、DDPM 1 epoch、GNN 1 epoch。\n")
    else:
        n_splits = args.folds
        max_records = None
        diffusion_epochs = args.diffusion_epochs
        gnn_epochs = args.gnn_epochs
        patience = args.patience
        out_name = f"{cfg['name']}_5fold_KD_ABLATION_{args.variant_name}_{run_tag}"

    output_dir = ROOT / "kd_results" / out_name

    # 默认禁止覆盖已有完整实验结果，保护之前跑好的结果
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"结果目录已存在且非空：{output_dir}\n"
            "为避免覆盖实验，请更换参数/seed，或手动确认后删除该目录再重跑。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DG-PBs-KD TEACHER-COMPOSITION ABLATION CV")
    print("dataset :", cfg["name"])
    print("device  :", device)
    print("raw     :", cfg["raw"])
    print("soft    :", cfg["soft"])
    print("output  :", output_dir)
    print("lambda_kd :", args.lambda_kd)
    print("lambda_aug:", args.lambda_aug)
    print("temperature:", args.temperature)
    print("seed      :", args.seed)
    print("=" * 78)

    # 立即保存本次运行配置，便于后续论文复现
    run_config = vars(args).copy()
    run_config["dataset_name"] = cfg["name"]
    run_config["method"] = "teacher-composition ablation: confidence-aware + class-balanced KD"
    run_config["variant_name"] = args.variant_name
    run_config["soft_label_csv_resolved"] = str(cfg["soft"])
    run_config["output_dir"] = str(output_dir)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    # 只加载一次 ESM-2 / 构图；之后按蛋白索引划 fold
    dataset_obj = KDProteinDataset(
        path=str(cfg["raw"]),
        soft_label_csv=str(cfg["soft"]),
        device=str(device),
        knn_k=args.knn_k,
        max_records=max_records,
        verbose=False,
    )
    original_data = dataset_obj.proteins

    print(f"\n[DATA] proteins = {len(original_data)}")
    print(f"[DATA] residues = {sum(int(d.y.numel()) for d in original_data)}")

    # V2 sanity check
    all_conf = torch.cat([d.teacher_confidence.float() for d in original_data])
    if not torch.isfinite(all_conf).all():
        raise ValueError("teacher_confidence 存在 NaN/Inf")
    if not ((all_conf >= 0) & (all_conf <= 1)).all():
        raise ValueError("teacher_confidence 超出 [0,1]")
    print(
        f"[DATA] teacher_confidence mean={float(all_conf.mean()):.4f}, "
        f"min={float(all_conf.min()):.4f}, max={float(all_conf.max()):.4f} ✅"
    )

    if len(original_data) < n_splits:
        raise ValueError("蛋白数小于折数")

    indices = np.arange(len(original_data))
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_assignments = []
    fold_summaries = []
    oof_rows = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print("\n" + "#" * 78)
        print(f"FOLD {fold}/{n_splits}")
        print("#" * 78)

        # 每折独立设置seed，使复现稳定
        set_seed(args.seed + fold)

        train_fold = [original_data[i] for i in train_idx]
        val_fold = [original_data[i] for i in val_idx]

        train_names = {str(d.name) for d in train_fold}
        val_names = {str(d.name) for d in val_fold}
        overlap = train_names & val_names
        if overlap:
            raise RuntimeError(f"train/val protein 泄漏: {list(overlap)[:5]}")

        print(f"[SPLIT] train proteins={len(train_fold)}")
        print(f"[SPLIT] val   proteins={len(val_fold)}")
        print("[SPLIT] overlap=0  ✅")

        for i in train_idx:
            fold_assignments.append(
                {"protein_id": str(original_data[i].name), "fold": fold, "role": "train"}
            )
        for i in val_idx:
            fold_assignments.append(
                {"protein_id": str(original_data[i].name), "fold": fold, "role": "val"}
            )

        # ---------- DDPM：只使用 train fold ----------
        print("\n[1/3] Train DDPM on TRAIN FOLD ONLY")
        ddpm = EnhancedDiffusionModel(
            input_dim=1280,
            T=args.diffusion_T,
            device=device,
        )

        t0 = time.time()
        ddpm.train_on_positive_samples(
            train_fold,
            epochs=diffusion_epochs,
            batch_size=args.diffusion_batch_size,
        )
        print(f"[DDPM] finished in {(time.time()-t0)/60:.2f} min")

        # ---------- 只增强 train fold ----------
        print("\n[2/3] Augment TRAIN FOLD ONLY")
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

        train_aug = kd_augment_dataset(train_fold, ddpm, aug_cfg)

        real_nodes = int(sum(int(d.is_real.sum()) for d in train_aug))
        syn_nodes = int(sum(int((~d.is_real).sum()) for d in train_aug))
        print(f"[AUG] real nodes={real_nodes}, synthetic nodes={syn_nodes}")
        print("[AUG] validation fold remains untouched ✅")

        # DDPM不再需要，释放显存
        del ddpm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # ---------- 训练 KD GNN ----------
        print("\n[3/3] Train DG-PBs-KD")
        model, best_epoch, best_val_aupr = train_kd_model(
            train_aug,
            val_fold,
            device,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            lr=args.gnn_lr,
            epochs=gnn_epochs,
            patience=patience,
            lambda_kd=args.lambda_kd,
            lambda_aug=args.lambda_aug,
            temperature=args.temperature,
        )

        val = collect_probs(model, val_fold, device)
        rank = ranking_metrics(val["label"], val["prob"])

        print(
            f"[FOLD {fold}] best_epoch={best_epoch}, "
            f"AUROC={rank['auroc']:.4f}, AUPR={rank['aupr']:.4f}"
        )

        for pid, ridx, y, p in zip(
            val["protein_id"],
            val["residue_index"],
            val["label"],
            val["prob"],
        ):
            oof_rows.append(
                {
                    "fold": fold,
                    "protein_id": pid,
                    "residue_index": ridx,
                    "label": int(y),
                    "student_prob": f"{float(p):.10f}",
                }
            )

        fold_summaries.append(
            {
                "fold": fold,
                "train_proteins": len(train_fold),
                "val_proteins": len(val_fold),
                "best_epoch": best_epoch,
                "val_auroc": rank["auroc"],
                "val_aupr": rank["aupr"],
                "real_nodes_after_aug": real_nodes,
                "synthetic_nodes": syn_nodes,
            }
        )

        # 可选保存小GNN checkpoint；不保存每折DDPM
        torch.save(
            {
                "model_state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "best_epoch": best_epoch,
                "fold": fold,
                "dataset": cfg["name"],
                "lambda_kd": args.lambda_kd,
                "lambda_aug": args.lambda_aug,
                "temperature": args.temperature,
                "method": "teacher-composition ablation: confidence-aware + class-balanced KD",
            },
            output_dir / f"fold_{fold}_best_gnn.pt",
        )

        del model, train_aug
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ---------- OOF汇总 ----------
    oof_path = output_dir / "oof_predictions.csv"
    write_oof_csv(oof_path, oof_rows)

    y_oof = np.asarray([int(r["label"]) for r in oof_rows], dtype=np.int64)
    p_oof = np.asarray([float(r["student_prob"]) for r in oof_rows], dtype=np.float64)

    rank = ranking_metrics(y_oof, p_oof)
    best_thr, _ = find_best_mcc_threshold(y_oof, p_oof)
    hard = hard_metrics(y_oof, p_oof, best_thr)

    # fold summary
    fold_csv = output_dir / "cv_fold_summary.csv"
    with open(fold_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(fold_summaries)

    assign_csv = output_dir / "fold_assignments.csv"
    with open(assign_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["protein_id", "fold", "role"]
        )
        writer.writeheader()
        writer.writerows(fold_assignments)

    median_best_epoch = int(
        round(np.median([x["best_epoch"] for x in fold_summaries]))
    )

    summary = {
        "dataset": cfg["name"],
        "seed": args.seed,
        "n_splits": n_splits,
        "smoke": bool(args.smoke),
        "lambda_kd": args.lambda_kd,
        "lambda_aug": args.lambda_aug,
        "temperature": args.temperature,
        "oof_auroc": rank["auroc"],
        "oof_aupr": rank["aupr"],
        "oof_mcc": hard["mcc"],
        "oof_f1": hard["f1"],
        "oof_precision": hard["precision"],
        "oof_sensitivity": hard["sensitivity"],
        "oof_specificity": hard["specificity"],
        "oof_threshold_mcc": best_thr,
        "median_best_epoch": median_best_epoch,
        "fold_mean_aupr": float(np.nanmean([x["val_aupr"] for x in fold_summaries])),
        "fold_std_aupr": float(np.nanstd([x["val_aupr"] for x in fold_summaries])),
        "fold_mean_auroc": float(np.nanmean([x["val_auroc"] for x in fold_summaries])),
        "fold_std_auroc": float(np.nanstd([x["val_auroc"] for x in fold_summaries])),
        "run_tag": run_tag,
        "output_dir": str(output_dir),
        "method": "teacher-composition ablation: confidence-aware + class-balanced KD",
    }

    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("CV COMPLETE")
    print("=" * 78)
    print(f"OOF AUROC = {summary['oof_auroc']:.4f}")
    print(f"OOF AUPR  = {summary['oof_aupr']:.4f}")
    print(f"OOF MCC   = {summary['oof_mcc']:.4f}")
    print(f"OOF F1    = {summary['oof_f1']:.4f}")
    print(f"OOF threshold (MCC, TRAIN ONLY) = {summary['oof_threshold_mcc']:.6f}")
    print(f"Median best epoch = {summary['median_best_epoch']}")
    print("Saved to:", output_dir)

    if args.smoke:
        print(
            "\n[SMOKE PASS] 如果上面完整结束且没有报错，"
            "下一步可以去掉 --smoke 正式跑5折。"
        )
    else:
        print(
            "\n[NEXT] 这一步只完成训练集内部CV。"
            "不要直接在独立测试集调参；下一步将根据CV结果固定参数，"
            "在全部训练集上训练最终DG-PBs-KD，再测试独立集。"
        )


if __name__ == "__main__":
    main()
