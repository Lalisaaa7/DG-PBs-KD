# -*- coding: utf-8 -*-
"""
DNA-573 OOF 稳定阈值分析（不使用 DNA-129 / DNA-181 标签）

目标
----
对已经锁定的 lambda_KD=0.5 的 DNA-573 五折 OOF 预测做阈值稳定性分析：

A. pooled OOF threshold
   把5折OOF拼起来，直接找MCC最优阈值（你之前得到约0.1478）

B. cross-fitted threshold
   对每个fold：
       用另外4个fold的OOF标签+概率寻找MCC最优阈值
       再把这个阈值应用到当前held-out fold
   这样每个fold的阈值都没有使用该fold自己的标签。

C. stable threshold
   取5个 cross-fitted threshold 的中位数，作为更稳健的候选最终阈值。

整个过程只使用 DNA-573 训练集内部OOF，不读取独立测试集。

运行：
    python .\analyze_stable_threshold.py

也可以显式指定：
    python .\analyze_stable_threshold.py --input "E:\\zhq-ds-main\\kd_results\\DNA-573_Train_5fold_KD\\oof_predictions.csv"
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent


def find_default_input():
    candidates = [
        ROOT / "kd_results" / "DNA-573_Train_5fold_KD_lkd0p50_laug0p20_T1p00_seed42" / "oof_predictions.csv",
        ROOT / "kd_results" / "DNA-573_Train_5fold_KD" / "oof_predictions.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[-1]


def find_best_mcc_threshold(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)

    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]

    total_pos = int(ys.sum())
    total_neg = int(len(ys) - total_pos)

    tp = fp = 0
    best_mcc = -np.inf
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

    return float(best_thr), float(best_mcc)


def metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= float(threshold)).astype(np.int64)

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
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "predicted_positive_rate": float(pred.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_oof(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"fold", "label", "student_prob"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"OOF文件缺少列: {sorted(missing)}")

        for r in reader:
            rows.append(
                {
                    "fold": int(r["fold"]),
                    "label": int(float(r["label"])),
                    "prob": float(r["student_prob"]),
                    "protein_id": r.get("protein_id", ""),
                    "residue_index": r.get("residue_index", ""),
                }
            )
    if not rows:
        raise ValueError("OOF文件为空")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="lambda_KD=0.5 的 oof_predictions.csv",
    )
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else find_default_input()
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到 OOF 文件：{input_path}\n"
            "请用 --input 显式指定 lambda_KD=0.5 的 oof_predictions.csv"
        )

    rows = load_oof(input_path)
    folds = sorted(set(r["fold"] for r in rows))

    y_all = np.asarray([r["label"] for r in rows], dtype=np.int64)
    p_all = np.asarray([r["prob"] for r in rows], dtype=np.float64)
    fold_all = np.asarray([r["fold"] for r in rows], dtype=np.int64)

    print("=" * 80)
    print("DNA-573 OOF STABLE THRESHOLD ANALYSIS")
    print("=" * 80)
    print("input    :", input_path)
    print("residues :", len(rows))
    print("folds    :", folds)
    print("positive :", int(y_all.sum()))
    print()

    pooled_thr, _ = find_best_mcc_threshold(y_all, p_all)
    pooled_metrics = metrics(y_all, p_all, pooled_thr)

    print("[A] POOLED OOF THRESHOLD")
    print(f"threshold = {pooled_thr:.6f}")
    print(
        f"MCC={pooled_metrics['mcc']:.4f}, "
        f"F1={pooled_metrics['f1']:.4f}, "
        f"Precision={pooled_metrics['precision']:.4f}, "
        f"Recall={pooled_metrics['recall']:.4f}, "
        f"Specificity={pooled_metrics['specificity']:.4f}"
    )
    print()

    fold_results = []
    crossfit_pred = np.zeros(len(rows), dtype=np.int64)

    print("[B] CROSS-FITTED THRESHOLDS")
    for fold in folds:
        train_mask = fold_all != fold
        held_mask = fold_all == fold

        thr, train_mcc = find_best_mcc_threshold(
            y_all[train_mask],
            p_all[train_mask],
        )
        held_metrics = metrics(y_all[held_mask], p_all[held_mask], thr)
        crossfit_pred[held_mask] = (p_all[held_mask] >= thr).astype(np.int64)

        result = {
            "held_out_fold": int(fold),
            "threshold_fit_on_other_4_folds": float(thr),
            "train4_mcc_at_selected_threshold": float(train_mcc),
            "heldout_mcc": held_metrics["mcc"],
            "heldout_f1": held_metrics["f1"],
            "heldout_precision": held_metrics["precision"],
            "heldout_recall": held_metrics["recall"],
            "heldout_specificity": held_metrics["specificity"],
            "heldout_predicted_positive_rate": held_metrics["predicted_positive_rate"],
        }
        fold_results.append(result)

        print(
            f"Fold {fold}: threshold={thr:.6f} | "
            f"held MCC={held_metrics['mcc']:.4f}, "
            f"F1={held_metrics['f1']:.4f}, "
            f"P={held_metrics['precision']:.4f}, "
            f"R={held_metrics['recall']:.4f}"
        )

    thresholds = np.asarray(
        [r["threshold_fit_on_other_4_folds"] for r in fold_results],
        dtype=np.float64,
    )

    stable_thr = float(np.median(thresholds))
    mean_thr = float(np.mean(thresholds))
    std_thr = float(np.std(thresholds))

    crossfit_mcc = float(matthews_corrcoef(y_all, crossfit_pred))
    crossfit_f1 = float(f1_score(y_all, crossfit_pred, zero_division=0))
    crossfit_precision = float(
        precision_score(y_all, crossfit_pred, zero_division=0)
    )
    crossfit_recall = float(
        recall_score(y_all, crossfit_pred, zero_division=0)
    )
    tn = int(((crossfit_pred == 0) & (y_all == 0)).sum())
    fp = int(((crossfit_pred == 1) & (y_all == 0)).sum())
    crossfit_specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print()
    print("[C] STABLE THRESHOLD")
    print("5 thresholds :", ", ".join(f"{x:.6f}" for x in thresholds))
    print(f"median       : {stable_thr:.6f}")
    print(f"mean         : {mean_thr:.6f}")
    print(f"std          : {std_thr:.6f}")
    print()

    stable_metrics = metrics(y_all, p_all, stable_thr)
    default05_metrics = metrics(y_all, p_all, 0.5)

    print("[D] TRAIN-ONLY COMPARISON")
    print(
        f"Pooled-optimal {pooled_thr:.6f}: "
        f"MCC={pooled_metrics['mcc']:.4f}, "
        f"F1={pooled_metrics['f1']:.4f}"
    )
    print(
        f"Stable-median {stable_thr:.6f}: "
        f"MCC={stable_metrics['mcc']:.4f}, "
        f"F1={stable_metrics['f1']:.4f}, "
        f"P={stable_metrics['precision']:.4f}, "
        f"R={stable_metrics['recall']:.4f}, "
        f"Spe={stable_metrics['specificity']:.4f}"
    )
    print(
        f"Default 0.500000: "
        f"MCC={default05_metrics['mcc']:.4f}, "
        f"F1={default05_metrics['f1']:.4f}"
    )
    print(
        f"Cross-fitted held-out hard metrics: "
        f"MCC={crossfit_mcc:.4f}, "
        f"F1={crossfit_f1:.4f}, "
        f"P={crossfit_precision:.4f}, "
        f"R={crossfit_recall:.4f}, "
        f"Spe={crossfit_specificity:.4f}"
    )

    out_dir = input_path.parent / "stable_threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "crossfit_thresholds.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_results[0].keys()))
        writer.writeheader()
        writer.writerows(fold_results)

    summary = {
        "input": str(input_path),
        "pooled_oof_threshold": pooled_thr,
        "pooled_oof_metrics": pooled_metrics,
        "crossfit_thresholds": thresholds.tolist(),
        "stable_threshold_median": stable_thr,
        "threshold_mean": mean_thr,
        "threshold_std": std_thr,
        "stable_threshold_oof_metrics": stable_metrics,
        "default_0p5_oof_metrics": default05_metrics,
        "crossfit_heldout_metrics": {
            "mcc": crossfit_mcc,
            "f1": crossfit_f1,
            "precision": crossfit_precision,
            "recall": crossfit_recall,
            "specificity": crossfit_specificity,
        },
        "rule": (
            "Final candidate threshold = median of 5 thresholds; "
            "each threshold was fitted using the other four OOF folds only."
        ),
        "independent_test_used": False,
    }

    with (out_dir / "stable_threshold_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("[SAVE]", out_dir)
    print(
        "\n[NEXT] 把这里的5个threshold、median和cross-fitted MCC/F1发给我。"
        "\n暂时不要重新跑DNA-129/DNA-181。"
    )


if __name__ == "__main__":
    main()
