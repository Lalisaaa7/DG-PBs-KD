# -*- coding: utf-8 -*-
"""
DG-PBs-KD V2：lambda_KD=1.2 的多随机种子稳定阈值分析

只使用 DNA-573 的 OOF 预测：
    seed = 42, 2024, 3407
不读取 DNA-129 / DNA-181 标签。

每个 seed：
1) 计算 pooled OOF MCC 最优阈值；
2) 对5折做 cross-fitted threshold：
   用另外4折OOF选择阈值，再应用到当前held-out fold；
3) 收集三个seed共15个“4折训练得到”的阈值。

最终：
- stable_threshold = 15个 cross-fitted thresholds 的中位数
- 在每个seed的完整OOF上用同一个 stable_threshold 计算 MCC/F1/P/R/Spe
- 保存汇总结果

运行：
    cd "E:\\zhq-ds-main"
    python .\\analyze_v2_multiseed_stable_threshold.py
"""

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
SEEDS = [42, 2024, 3407]
LAMBDA_TAG = "1p20"
LAMBDA_KD = 1.2


def oof_path(seed):
    return (
        ROOT / "kd_results"
        / f"DNA-573_Train_5fold_KD_CONFIDENCE_V2_lkd{LAMBDA_TAG}_laug0p20_T1p00_seed{seed}"
        / "oof_predictions.csv"
    )


def load_oof(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"fold", "label", "student_prob"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
        for r in reader:
            rows.append({
                "fold": int(r["fold"]),
                "label": int(float(r["label"])),
                "prob": float(r["student_prob"]),
            })
    return rows


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

        denom = math.sqrt(max(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0
        ))
        mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(score)

        i = j

    return float(best_thr), float(best_mcc)


def hard_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    spe = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(spe),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    print("=" * 88)
    print("DG-PBs-KD V2 MULTI-SEED STABLE THRESHOLD")
    print("lambda_KD = 1.2")
    print("independent test labels used: NO")
    print("=" * 88)

    per_seed_data = {}
    all_crossfit_thresholds = []
    threshold_rows = []

    for seed in SEEDS:
        path = oof_path(seed)
        if not path.exists():
            raise FileNotFoundError(f"找不到：{path}")

        rows = load_oof(path)
        y = np.array([r["label"] for r in rows], dtype=int)
        p = np.array([r["prob"] for r in rows], dtype=float)
        folds = np.array([r["fold"] for r in rows], dtype=int)

        pooled_thr, pooled_mcc = find_best_mcc_threshold(y, p)

        print(f"\n[seed={seed}] pooled threshold={pooled_thr:.6f}, MCC={pooled_mcc:.4f}")

        seed_thresholds = []
        for fold in sorted(set(folds.tolist())):
            train_mask = folds != fold
            held_mask = folds == fold

            thr, train4_mcc = find_best_mcc_threshold(
                y[train_mask], p[train_mask]
            )
            hm = hard_metrics(y[held_mask], p[held_mask], thr)

            seed_thresholds.append(thr)
            all_crossfit_thresholds.append(thr)

            threshold_rows.append({
                "seed": seed,
                "held_out_fold": int(fold),
                "threshold_fit_on_other_4_folds": thr,
                "train4_mcc": train4_mcc,
                "heldout_mcc": hm["mcc"],
                "heldout_f1": hm["f1"],
            })

            print(
                f"  Fold {fold}: threshold={thr:.6f} | "
                f"held MCC={hm['mcc']:.4f}, F1={hm['f1']:.4f}"
            )

        print(
            f"  seed median={np.median(seed_thresholds):.6f}, "
            f"mean={np.mean(seed_thresholds):.6f}, "
            f"std={np.std(seed_thresholds):.6f}"
        )

        per_seed_data[seed] = {
            "y": y,
            "p": p,
            "pooled_threshold": pooled_thr,
            "crossfit_thresholds": seed_thresholds,
        }

    stable_threshold = float(np.median(all_crossfit_thresholds))

    print("\n" + "=" * 88)
    print("FINAL STABLE THRESHOLD")
    print("=" * 88)
    print("15 cross-fitted thresholds:")
    print(", ".join(f"{x:.6f}" for x in all_crossfit_thresholds))
    print(f"\nmedian = {stable_threshold:.6f}")
    print(f"mean   = {np.mean(all_crossfit_thresholds):.6f}")
    print(f"std    = {np.std(all_crossfit_thresholds):.6f}")

    metric_rows = []
    for seed in SEEDS:
        y = per_seed_data[seed]["y"]
        p = per_seed_data[seed]["p"]
        m = hard_metrics(y, p, stable_threshold)
        metric_rows.append({"seed": seed, **m})

        print(
            f"\nseed={seed} @ stable threshold {stable_threshold:.6f}: "
            f"MCC={m['mcc']:.4f}, F1={m['f1']:.4f}, "
            f"P={m['precision']:.4f}, R={m['recall']:.4f}, "
            f"Spe={m['specificity']:.4f}"
        )

    print("\n3-seed hard-metric mean ± std:")
    for key in ["mcc", "f1", "precision", "recall", "specificity"]:
        arr = np.array([r[key] for r in metric_rows], dtype=float)
        print(f"{key:12s} = {arr.mean():.4f} ± {arr.std():.4f}")

    out_dir = ROOT / "kd_results" / "V2_lambda1p20_stable_threshold"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "crossfit_thresholds_15.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_rows)

    with (out_dir / "stable_threshold_per_seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    payload = {
        "lambda_kd": LAMBDA_KD,
        "seeds": SEEDS,
        "selection_source": "DNA-573 OOF only",
        "independent_test_labels_used": False,
        "crossfit_thresholds_15": all_crossfit_thresholds,
        "stable_threshold_median": stable_threshold,
        "threshold_mean": float(np.mean(all_crossfit_thresholds)),
        "threshold_std": float(np.std(all_crossfit_thresholds)),
        "per_seed_metrics_at_stable_threshold": metric_rows,
    }

    (out_dir / "stable_threshold_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n[SAVE]", out_dir)
    print("\n[NEXT] 把 FINAL STABLE THRESHOLD 和 3-seed hard-metric mean±std 发给我。")
    print("暂时不要测试 DNA-129 / DNA-181。")


if __name__ == "__main__":
    main()
