# -*- coding: utf-8 -*-
"""
ESM-NBR + TransBind 双教师 DG-PBs-KD 一键完整流程
================================================

一次顺序完成：

A) DNA-573_Train
   -> 生成 ESM-NBR + TransBind 双教师 soft label（等权）
   -> 在 DNA-573 内部搜索 lambda_KD
   -> 按 MCC > F1 > AUPR > AUROC 选择最佳 lambda
   -> 用该 lambda 的 OOF 最优阈值和 median best epoch
   -> 完整 DNA-573 训练最终 DG-PBs-KD
   -> DNA-129_Test + DNA-181_Test 独立测试

B) DNA-646_Train
   -> 同样生成双教师 soft label
   -> 内部搜索 lambda_KD
   -> 选择最佳 lambda
   -> 完整 DNA-646 训练最终 DG-PBs-KD
   -> DNA-46_Test 独立测试

注意：
- 只使用 ESM-NBR + TransBind，不使用 CLAPE。
- 双教师 soft label = 0.5 * ESM-NBR + 0.5 * TransBind。
- teacher_confidence 根据两教师分歧自动计算。
- lambda / threshold / epoch 都只从对应训练集内部 CV 得到。
- 129 / 181 / 46 不参与调参。
- 默认 seed=42，属于“先看最终效果”的快速正式流程。
- 已经完成的双教师 CV 结果会自动跳过。
- 顺序运行，不并行抢 GPU。

运行：
    conda activate three_teacher_dna
    cd "E:\\zhq-ds-main"
    python .\\run_dual_teacher_full_pipeline.py

如果只想先跑573：
    python .\\run_dual_teacher_full_pipeline.py --only 573

如果只想先跑646：
    python .\\run_dual_teacher_full_pipeline.py --only 646
"""

import argparse
import csv
import gc
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)

from ddpm_diffusion_model import EnhancedDiffusionModel
from kd_augmentation_v2 import kd_augment_dataset
from kd_data_loader_v2 import KDProteinDataset
from kd_gnn_model_v2 import ConfidenceAwareKDBindingSiteGNN


ROOT = Path(__file__).resolve().parent
CV_SCRIPT = ROOT / "kd_cv_teacher_ablation.py"

SEED = 42
LAMBDA_AUG = 0.2
TEMPERATURE = 1.0

# 与前面搜索保持一致
LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

CONFIGS = {
    "573": {
        "train_name": "DNA-573_Train",
        "train_raw": ROOT / "Raw_data" / "DNA-573_Train.txt",
        "teacher_pred": (
            ROOT / "ThreeTeacher_DNA_Predictor_ReadyCode"
            / "ThreeTeacher_DNA_Predictor"
            / "outputs" / "DNA_573_Train" / "residue_predictions.csv"
        ),
        "dual_soft": (
            ROOT / "teacher_ablation_soft_labels"
            / "DNA-573_Train_esm_trans_equal.csv"
        ),
        "tests": [
            ("DNA-129_Test", ROOT / "Raw_data" / "DNA-129_Test.txt"),
            ("DNA-181_Test", ROOT / "Raw_data" / "DNA-181_Test.txt"),
        ],
    },
    "646": {
        "train_name": "DNA-646_Train",
        "train_raw": ROOT / "Raw_data" / "DNA-646_Train.txt",
        "teacher_pred": (
            ROOT / "ThreeTeacher_DNA_Predictor_ReadyCode"
            / "ThreeTeacher_DNA_Predictor"
            / "outputs" / "DNA_646_Train" / "residue_predictions.csv"
        ),
        "dual_soft": (
            ROOT / "teacher_ablation_soft_labels"
            / "DNA-646_Train_esm_trans_equal.csv"
        ),
        "tests": [
            ("DNA-46_Test", ROOT / "Raw_data" / "DNA-46_Test.txt"),
        ],
    },
}


def tag_float(x):
    return f"{float(x):.2f}".replace(".", "p")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Step 1: 生成 ESM-NBR + TransBind 双教师 soft labels
# ---------------------------------------------------------------------
def prepare_dual_soft_labels(cfg):
    src = cfg["teacher_pred"]
    out = cfg["dual_soft"]

    if not src.exists():
        raise FileNotFoundError(
            f"找不到教师预测文件：\n{src}\n"
            "请先确认三教师工程已经完成该训练集的 residue_predictions.csv。"
        )

    # 已存在也重新生成，确保与当前 residue_predictions.csv 一致
    out.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {
        "protein_id",
        "residue_index",
        "amino_acid",
        "label",
        "esm_nbr_prob",
        "transbind_prob",
    }
    missing = required - fields
    if missing:
        raise ValueError(f"{src} 缺少列：{sorted(missing)}")

    fieldnames = [
        "protein_id",
        "residue_index",
        "amino_acid",
        "label",
        "teacher_soft_prob",
        "teacher_confidence",
        "teacher_disagreement_std",
    ]

    confs = []

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in rows:
            e = float(r["esm_nbr_prob"])
            t = float(r["transbind_prob"])

            q = 0.5 * e + 0.5 * t

            # 两个值的总体标准差
            std = math.sqrt(((e - q) ** 2 + (t - q) ** 2) / 2.0)

            # 与之前学生端消融保持相同定义
            conf = max(0.0, min(1.0, 1.0 - 2.0 * std))
            confs.append(conf)

            writer.writerow({
                "protein_id": r["protein_id"],
                "residue_index": r["residue_index"],
                "amino_acid": r["amino_acid"],
                "label": r["label"],
                "teacher_soft_prob": f"{q:.10f}",
                "teacher_confidence": f"{conf:.10f}",
                "teacher_disagreement_std": f"{std:.10f}",
            })

    print(
        f"[SOFT OK] {cfg['train_name']} | "
        f"residues={len(rows)} | "
        f"confidence mean={np.mean(confs):.4f} | "
        f"output={out}"
    )


# ---------------------------------------------------------------------
# Step 2: 双教师 lambda CV
# ---------------------------------------------------------------------
def cv_result_dir(dataset, cfg, lam):
    run_tag = (
        f"lkd{tag_float(lam)}"
        f"_laug{tag_float(LAMBDA_AUG)}"
        f"_T{tag_float(TEMPERATURE)}"
        f"_seed{SEED}"
    )

    return (
        ROOT / "kd_results"
        / f"{cfg['train_name']}_5fold_KD_ABLATION_esm_trans_equal_{run_tag}"
    )


def cv_summary_path(dataset, cfg, lam):
    return cv_result_dir(dataset, cfg, lam) / "cv_summary.json"


def cv_complete(dataset, cfg, lam):
    p = cv_summary_path(dataset, cfg, lam)
    if not p.exists():
        return False

    try:
        j = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        j.get("smoke") is False
        and int(j.get("n_splits", 0)) == 5
        and int(j.get("seed", -1)) == SEED
        and abs(float(j.get("lambda_kd", -999)) - lam) < 1e-9
    )


def run_lambda_sweep(dataset, cfg):
    if not CV_SCRIPT.exists():
        raise FileNotFoundError(
            f"找不到：{CV_SCRIPT}\n"
            "请把 kd_cv_teacher_ablation.py 放在 E:\\zhq-ds-main\\"
        )

    print("\n" + "=" * 96)
    print(f"{cfg['train_name']} | ESM-NBR + TransBind lambda_KD SEARCH")
    print("=" * 96)

    for lam in LAMBDAS:
        if cv_complete(dataset, cfg, lam):
            print(f"[SKIP] lambda={lam:.1f} 已完成")
            continue

        out = cv_result_dir(dataset, cfg, lam)

        if out.exists() and any(out.iterdir()):
            raise RuntimeError(
                f"发现非空但不完整结果目录：\n{out}\n"
                "请确认是否为断电残留，手动改名/删除后再重新运行。"
            )

        cmd = [
            sys.executable,
            "-u",
            str(CV_SCRIPT),
            "--dataset", dataset,
            "--lambda-kd", str(lam),
            "--lambda-aug", str(LAMBDA_AUG),
            "--temperature", str(TEMPERATURE),
            "--seed", str(SEED),
            "--variant-name", "esm_trans_equal",
            "--soft-label-csv", str(cfg["dual_soft"]),
        ]

        print("\n[RUN]", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=str(ROOT))

        if ret.returncode != 0:
            raise RuntimeError(f"lambda={lam:.1f} CV运行失败")

        if not cv_complete(dataset, cfg, lam):
            raise RuntimeError(
                f"lambda={lam:.1f} 进程结束，但未找到有效正式5折结果"
            )

    # 选最佳 lambda
    rows = []

    for lam in LAMBDAS:
        p = cv_summary_path(dataset, cfg, lam)
        j = json.loads(p.read_text(encoding="utf-8"))

        rows.append({
            "lambda_kd": float(j["lambda_kd"]),
            "auroc": float(j["oof_auroc"]),
            "aupr": float(j["oof_aupr"]),
            "mcc": float(j["oof_mcc"]),
            "f1": float(j["oof_f1"]),
            "precision": float(j.get("oof_precision", float("nan"))),
            "recall": float(j.get("oof_sensitivity", float("nan"))),
            "specificity": float(j.get("oof_specificity", float("nan"))),
            "threshold": float(j["oof_threshold_mcc"]),
            "epoch": int(j["median_best_epoch"]),
        })

    best = max(
        rows,
        key=lambda r: (
            r["mcc"],
            r["f1"],
            r["aupr"],
            r["auroc"],
        ),
    )

    print("\n" + "-" * 94)
    print(f"{cfg['train_name']} DOUBLE-TEACHER LAMBDA SUMMARY")
    print("-" * 94)
    print(
        f"{'lambda':>7} {'AUROC':>8} {'AUPR':>8} "
        f"{'MCC':>8} {'F1':>8} {'thr':>10} {'epoch':>7}"
    )

    for r in rows:
        mark = "  <-- BEST" if r["lambda_kd"] == best["lambda_kd"] else ""
        print(
            f"{r['lambda_kd']:>7.1f} "
            f"{r['auroc']:>8.4f} "
            f"{r['aupr']:>8.4f} "
            f"{r['mcc']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['threshold']:>10.6f} "
            f"{r['epoch']:>7d}"
            f"{mark}"
        )

    print(
        f"\n[LOCK FOR THIS RUN] lambda_KD={best['lambda_kd']:.1f}, "
        f"threshold={best['threshold']:.6f}, "
        f"epochs={best['epoch']}"
    )

    sweep_dir = (
        ROOT / "kd_results"
        / "DUAL_ESM_TRANS_FULL"
        / cfg["train_name"]
        / "lambda_sweep"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)

    with (sweep_dir / "lambda_sweep_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (sweep_dir / "best_lambda.json").write_text(
        json.dumps(
            {
                "dataset": cfg["train_name"],
                "teacher": "ESM-NBR + TransBind",
                "fusion": "equal 0.5 / 0.5",
                "seed": SEED,
                "selection_rule": "MCC > F1 > AUPR > AUROC",
                "best": best,
                "all": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return best


# ---------------------------------------------------------------------
# Final full-train model
# ---------------------------------------------------------------------
def train_final_gnn(
    train_aug,
    device,
    lambda_kd,
    epochs,
    lr=5e-4,
    hidden_dim=256,
    dropout=0.5,
):
    model = ConfidenceAwareKDBindingSiteGNN(
        input_dim=1280,
        hidden_dim=hidden_dim,
        dropout=dropout,
        lambda_kd=lambda_kd,
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
                raise RuntimeError("出现 non-finite loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total += float(parts["total"])
            hard += float(parts["hard"])
            kd += float(parts["kd"])
            kd_pos += float(parts.get("kd_pos", 0.0))
            kd_neg += float(parts.get("kd_neg", 0.0))
            aug += float(parts["aug"])
            n_graph += 1

        n = max(n_graph, 1)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss={total/n:.4f} "
            f"(hard={hard/n:.4f}, kd={kd/n:.4f}, "
            f"kd_pos={kd_pos/n:.4f}, kd_neg={kd_neg/n:.4f}, "
            f"aug={aug/n:.4f})"
        )

    return model


def calc_metrics(y, p, threshold):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= threshold).astype(np.int64)

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
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_residues": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
    }


def predict_dataset(model, dataset, device, threshold):
    model.eval()

    ys, ps, rows = [], [], []

    with torch.no_grad():
        for data in dataset:
            d = data.to(device)
            logits = model(d).reshape(-1)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = d.y.detach().cpu().numpy().astype(int)

            pid = str(
                getattr(
                    data,
                    "original_name",
                    getattr(data, "name", "unknown")
                )
            )
            seq = getattr(data, "sequence", None)

            for idx, (pp, yy) in enumerate(zip(probs, labels), start=1):
                aa = ""
                if seq is not None and idx <= len(seq):
                    aa = seq[idx - 1]

                rows.append({
                    "protein_id": pid,
                    "residue_index": idx,
                    "amino_acid": aa,
                    "label": int(yy),
                    "student_prob": float(pp),
                    "student_pred": int(float(pp) >= threshold),
                })

            ys.extend(labels.tolist())
            ps.extend(probs.tolist())

    return np.asarray(ys), np.asarray(ps), rows


def run_final(dataset, cfg, best):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    out_dir = (
        ROOT / "kd_results"
        / "DUAL_ESM_TRANS_FULL"
        / f"{cfg['train_name']}_FINAL_seed{SEED}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 96)
    print(f"FINAL DOUBLE-TEACHER: {cfg['train_name']}")
    print("#" * 96)
    print("Teacher      : ESM-NBR + TransBind")
    print("Fusion       : 0.5 / 0.5")
    print("lambda_KD    :", best["lambda_kd"])
    print("lambda_aug   :", LAMBDA_AUG)
    print("T            :", TEMPERATURE)
    print("threshold    :", best["threshold"], "(TRAIN CV ONLY)")
    print("GNN epochs   :", best["epoch"])
    print("test tuning  : NO")
    print("device       :", device)

    # 1) load full training
    print("\n[1/5] Load FULL training set")

    train_obj = KDProteinDataset(
        path=str(cfg["train_raw"]),
        soft_label_csv=str(cfg["dual_soft"]),
        device=str(device),
        knn_k=12,
        verbose=False,
    )
    train_data = train_obj.proteins

    print("proteins =", len(train_data))
    print("residues =", sum(int(d.y.numel()) for d in train_data))

    try:
        train_obj.model = train_obj.model.cpu()
    except Exception:
        pass
    clear_cuda()

    # 2) final DDPM
    print("\n[2/5] Train DDPM on FULL training set")
    ddpm = EnhancedDiffusionModel(
        input_dim=1280,
        T=200,
        device=device,
    )

    t0 = time.time()
    ddpm.train_on_positive_samples(
        train_data,
        epochs=60,
        batch_size=64,
    )
    print(f"[DDPM] finished in {(time.time()-t0)/60:.2f} min")

    # 3) augment
    print("\n[3/5] Augment FULL training set")

    aug_cfg = SimpleNamespace(
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

    train_aug = kd_augment_dataset(
        train_data,
        ddpm,
        aug_cfg,
    )

    real_nodes = sum(int(d.is_real.sum()) for d in train_aug)
    syn_nodes = sum(int((~d.is_real).sum()) for d in train_aug)

    print("[AUG] real nodes =", real_nodes)
    print("[AUG] synthetic nodes =", syn_nodes)
    print("[AUG] synthetic nodes do NOT participate in KD")

    del ddpm
    clear_cuda()

    # 4) train final GNN
    print("\n[4/5] Train FINAL DG-PBs-KD")

    model = train_final_gnn(
        train_aug,
        device,
        lambda_kd=best["lambda_kd"],
        epochs=best["epoch"],
    )

    model_path = out_dir / "final_dg_pbs_kd_dual_esm_trans.pt"

    torch.save(
        {
            "model_state_dict": {
                k: v.detach().cpu()
                for k, v in model.state_dict().items()
            },
            "dataset": cfg["train_name"],
            "seed": SEED,
            "teacher": "ESM-NBR + TransBind",
            "fusion": "equal 0.5 / 0.5",
            "lambda_kd": best["lambda_kd"],
            "lambda_aug": LAMBDA_AUG,
            "temperature": TEMPERATURE,
            "threshold": best["threshold"],
            "gnn_epochs": best["epoch"],
        },
        model_path,
    )

    print("[SAVE MODEL]", model_path)

    del train_aug, train_data, train_obj
    clear_cuda()

    # 5) independent tests
    print("\n[5/5] INDEPENDENT TEST -- NO TUNING")

    summary = []

    for test_name, test_raw in cfg["tests"]:
        if not test_raw.exists():
            raise FileNotFoundError(f"找不到：{test_raw}")

        print("\n" + "-" * 80)
        print("[TEST]", test_name)
        print("threshold =", best["threshold"], "(LOCKED)")
        print("-" * 80)

        test_obj = KDProteinDataset(
            path=str(test_raw),
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
        clear_cuda()

        y, p, rows = predict_dataset(
            model,
            test_data,
            device,
            best["threshold"],
        )

        m = calc_metrics(
            y,
            p,
            best["threshold"],
        )
        m["dataset"] = test_name
        summary.append(m)

        with (out_dir / f"{test_name}_metrics.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(m, f, ensure_ascii=False, indent=2)

        with (out_dir / f"{test_name}_predictions.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
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
            writer.writerows(rows)

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
        clear_cuda()

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

    with (out_dir / "independent_test_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in summary:
            writer.writerow({k: m[k] for k in fields})

    print("\n" + "=" * 96)
    print(f"{cfg['train_name']} DOUBLE-TEACHER FINAL COMPLETE")
    print("=" * 96)

    for m in summary:
        print(
            f"{m['dataset']}: "
            f"AUROC={m['auroc']:.4f}, "
            f"AUPR={m['aupr']:.4f}, "
            f"MCC={m['mcc']:.4f}, "
            f"F1={m['f1']:.4f}"
        )

    print("Saved to:", out_dir)

    del model
    clear_cuda()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["573", "646"],
        default=None,
        help="只跑指定训练组；默认两组都跑",
    )
    args = parser.parse_args()

    datasets = [args.only] if args.only else ["573", "646"]

    print("=" * 100)
    print("ESM-NBR + TransBind DOUBLE-TEACHER DG-PBs-KD FULL PIPELINE")
    print("=" * 100)
    print("Teacher = ESM-NBR + TransBind")
    print("Fusion  = equal 0.5 / 0.5")
    print("Lambda selection = MCC > F1 > AUPR > AUROC")
    print("Seed = 42")
    print("Independent-test tuning = NO")
    print("=" * 100)

    all_rows = []

    for dataset in datasets:
        cfg = CONFIGS[dataset]

        for p in [cfg["train_raw"], cfg["teacher_pred"]]:
            if not p.exists():
                raise FileNotFoundError(f"找不到：{p}")

        # Step 1
        print("\n" + "=" * 100)
        print(f"STEP 1 | PREPARE DUAL SOFT LABEL | {cfg['train_name']}")
        print("=" * 100)
        prepare_dual_soft_labels(cfg)

        # Step 2
        print("\n" + "=" * 100)
        print(f"STEP 2 | LAMBDA SEARCH | {cfg['train_name']}")
        print("=" * 100)
        best = run_lambda_sweep(dataset, cfg)

        # Step 3
        print("\n" + "=" * 100)
        print(f"STEP 3 | FINAL TRAIN + INDEPENDENT TEST | {cfg['train_name']}")
        print("=" * 100)
        summary = run_final(dataset, cfg, best)

        for m in summary:
            all_rows.append({
                "train": cfg["train_name"],
                "test": m["dataset"],
                "lambda_kd": best["lambda_kd"],
                "threshold": best["threshold"],
                "epochs": best["epoch"],
                "auroc": m["auroc"],
                "aupr": m["aupr"],
                "mcc": m["mcc"],
                "f1": m["f1"],
                "precision": m["precision"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            })

    out_root = ROOT / "kd_results" / "DUAL_ESM_TRANS_FULL"
    out_root.mkdir(parents=True, exist_ok=True)

    if all_rows:
        out_csv = out_root / "all_dual_teacher_independent_results.csv"

        with out_csv.open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(all_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(all_rows)

        print("\n" + "#" * 100)
        print("ALL DOUBLE-TEACHER EXPERIMENTS COMPLETE")
        print("#" * 100)

        print(
            f"{'Train':<18} {'Test':<15} "
            f"{'lambda':>7} {'thr':>9} "
            f"{'AUROC':>8} {'AUPR':>8} "
            f"{'MCC':>8} {'F1':>8}"
        )
        print("-" * 100)

        for r in all_rows:
            print(
                f"{r['train']:<18} {r['test']:<15} "
                f"{r['lambda_kd']:>7.1f} "
                f"{r['threshold']:>9.4f} "
                f"{r['auroc']:>8.4f} "
                f"{r['aupr']:>8.4f} "
                f"{r['mcc']:>8.4f} "
                f"{r['f1']:>8.4f}"
            )

        print("\n[SAVE]", out_csv)

    print(
        "\nIMPORTANT: 一旦看到129/181/46结果，"
        "不要再根据这些独立测试集反向修改lambda或threshold。"
    )


if __name__ == "__main__":
    main()
