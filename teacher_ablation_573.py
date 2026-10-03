# -*- coding: utf-8 -*-
"""
DNA-573 teacher ablation (fair MCC/F1-first evaluation)

Compares 7 teacher combinations using equal soft voting:
- CLAPE
- ESM-NBR
- TransBind
- CLAPE+ESM-NBR
- CLAPE+TransBind
- ESM-NBR+TransBind
- CLAPE+ESM-NBR+TransBind

For hard metrics, does NOT use default thresholds. Instead, for every combination,
performs protein-level 5-fold cross-fitted threshold selection:
other 4 folds -> choose MCC-optimal threshold; held-out fold -> evaluate only.
Thus ESM-NBR is not unfairly penalized by its original default 0.8 threshold.

Ranking rule: OOF MCC first, OOF F1 second, AUPR third.
Independent-test labels are never used.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT / "ThreeTeacher_DNA_Predictor_ReadyCode" / "ThreeTeacher_DNA_Predictor"
    / "outputs" / "DNA_573_Train" / "residue_predictions.csv"
)

COMBOS = {
    "CLAPE": ["clape_prob"],
    "ESM-NBR": ["esm_nbr_prob"],
    "TransBind": ["transbind_prob"],
    "CLAPE+ESM-NBR": ["clape_prob", "esm_nbr_prob"],
    "CLAPE+TransBind": ["clape_prob", "transbind_prob"],
    "ESM-NBR+TransBind": ["esm_nbr_prob", "transbind_prob"],
    "CLAPE+ESM-NBR+TransBind": ["clape_prob", "esm_nbr_prob", "transbind_prob"],
}


def find_best_mcc_threshold(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
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
        denom = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 0))
        mcc = ((tp*tn - fp*fn) / denom) if denom > 0 else 0.0
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(score)
        i = j
    return float(best_thr), float(best_mcc)


def hard_metrics(y, pred):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "MCC": float(matthews_corrcoef(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "Specificity": float(spe),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def load_csv(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"protein_id", "label", "clape_prob", "esm_nbr_prob", "transbind_prob"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"输入CSV缺少列: {sorted(missing)}")
        for r in reader:
            rows.append({
                "protein_id": r["protein_id"],
                "label": int(float(r["label"])),
                "clape_prob": float(r["clape_prob"]),
                "esm_nbr_prob": float(r["esm_nbr_prob"]),
                "transbind_prob": float(r["transbind_prob"]),
            })
    if not rows:
        raise ValueError("输入CSV为空")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else DEFAULT_INPUT
    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到: {input_path}\n请用 --input 指定 DNA-573_Train 的 residue_predictions.csv"
        )

    rows = load_csv(input_path)
    protein_ids = np.asarray([r["protein_id"] for r in rows], dtype=object)
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    probs = {
        "clape_prob": np.asarray([r["clape_prob"] for r in rows], dtype=np.float64),
        "esm_nbr_prob": np.asarray([r["esm_nbr_prob"] for r in rows], dtype=np.float64),
        "transbind_prob": np.asarray([r["transbind_prob"] for r in rows], dtype=np.float64),
    }

    unique_proteins = np.asarray(sorted(set(protein_ids.tolist())), dtype=object)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    protein_fold = {}
    for fold, (_, val_idx) in enumerate(kf.split(unique_proteins), start=1):
        for pid in unique_proteins[val_idx]:
            protein_fold[pid] = fold
    fold_arr = np.asarray([protein_fold[pid] for pid in protein_ids], dtype=int)

    print("=" * 88)
    print("DNA-573 TEACHER ABLATION — MCC/F1 FIRST")
    print("=" * 88)
    print("input:", input_path)
    print("proteins:", len(unique_proteins), "residues:", len(labels), "positives:", int(labels.sum()))
    print("fusion: equal soft voting")
    print("hard threshold: protein-level 5-fold cross-fitted, MCC-optimal on other 4 folds")
    print("selection: OOF MCC first, OOF F1 second, AUPR third")
    print("independent-test labels used: NO")

    summary_rows = []
    fold_rows = []

    for name, cols in COMBOS.items():
        combo_prob = np.mean(np.vstack([probs[c] for c in cols]), axis=0)
        auroc = float(roc_auc_score(labels, combo_prob))
        aupr = float(average_precision_score(labels, combo_prob))

        oof_pred = np.zeros_like(labels)
        fold_thresholds = []

        for fold in range(1, 6):
            train_mask = fold_arr != fold
            val_mask = fold_arr == fold
            thr, train_mcc = find_best_mcc_threshold(labels[train_mask], combo_prob[train_mask])
            pred_val = (combo_prob[val_mask] >= thr).astype(np.int64)
            oof_pred[val_mask] = pred_val
            hm = hard_metrics(labels[val_mask], pred_val)
            fold_thresholds.append(thr)
            fold_rows.append({
                "model": name,
                "fold": fold,
                "threshold_fit_on_other_4_folds": thr,
                "train4_MCC_at_selected_threshold": train_mcc,
                "heldout_MCC": hm["MCC"],
                "heldout_F1": hm["F1"],
                "heldout_Precision": hm["Precision"],
                "heldout_Recall": hm["Recall"],
                "heldout_Specificity": hm["Specificity"],
            })

        hm_all = hard_metrics(labels, oof_pred)
        full_thr, full_mcc = find_best_mcc_threshold(labels, combo_prob)
        summary_rows.append({
            "model": name,
            "n_teachers": len(cols),
            "AUROC": auroc,
            "AUPR": aupr,
            "OOF_MCC_crossfitted": hm_all["MCC"],
            "OOF_F1_crossfitted": hm_all["F1"],
            "OOF_Precision_crossfitted": hm_all["Precision"],
            "OOF_Recall_crossfitted": hm_all["Recall"],
            "OOF_Specificity_crossfitted": hm_all["Specificity"],
            "median_crossfit_threshold": float(np.median(fold_thresholds)),
            "mean_crossfit_threshold": float(np.mean(fold_thresholds)),
            "std_crossfit_threshold": float(np.std(fold_thresholds)),
            "full573_fitted_threshold_for_reference": full_thr,
            "full573_MCC_at_fitted_threshold": full_mcc,
        })

    summary_rows.sort(
        key=lambda r: (r["OOF_MCC_crossfitted"], r["OOF_F1_crossfitted"], r["AUPR"]),
        reverse=True,
    )

    out_dir = ROOT / "teacher_ablation_results" / "DNA_573"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "teacher_ablation_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader(); writer.writerows(summary_rows)

    fold_csv = out_dir / "teacher_ablation_fold_details.csv"
    with fold_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader(); writer.writerows(fold_rows)

    json_path = out_dir / "teacher_ablation_summary.json"
    json_path.write_text(json.dumps({
        "dataset": "DNA-573_Train",
        "seed": args.seed,
        "selection_rule": "OOF MCC first, OOF F1 second, AUPR third",
        "fusion_rule": "equal soft voting within each teacher combination",
        "threshold_rule": "protein-level 5-fold cross-fitted threshold",
        "independent_test_labels_used": False,
        "ranking": summary_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 110)
    print("FINAL ABLATION RANKING — MCC PRIMARY")
    print("=" * 110)
    print(f"{'Model':<30}{'AUROC':>9}{'AUPR':>9}{'OOF MCC':>10}{'OOF F1':>10}{'Prec':>9}{'Recall':>9}{'Thr-med':>10}")
    print("-" * 110)
    for i, r in enumerate(summary_rows, start=1):
        mark = "  <-- BEST" if i == 1 else ""
        print(
            f"{r['model']:<30}{r['AUROC']:>9.4f}{r['AUPR']:>9.4f}"
            f"{r['OOF_MCC_crossfitted']:>10.4f}{r['OOF_F1_crossfitted']:>10.4f}"
            f"{r['OOF_Precision_crossfitted']:>9.4f}{r['OOF_Recall_crossfitted']:>9.4f}"
            f"{r['median_crossfit_threshold']:>10.4f}{mark}"
        )

    print("\n[SAVE]")
    print(summary_csv)
    print(fold_csv)
    print(json_path)
    print("\n[NEXT] 把 FINAL ABLATION RANKING 截图或 summary.csv 发给我。")


if __name__ == "__main__":
    main()
