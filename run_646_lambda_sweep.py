# -*- coding: utf-8 -*-
"""
DNA-646 三教师 DG-PBs-KD V2：lambda_KD 一次性搜索

固定：
dataset=646, lambda_aug=0.2, temperature=1.0, seed=42
搜索：
lambda_KD = 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6

选择规则：
MCC 第一，F1 第二，AUPR 第三，AUROC 第四。

运行：
cd "E:\\zhq-ds-main"
python .\\run_646_lambda_sweep.py
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "kd_cv_confidence_v2.py"

DATASET = "646"
SEED = 42
LAMBDA_AUG = 0.2
TEMPERATURE = 1.0
LAMBDAS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]


def tag_float(x):
    return f"{float(x):.2f}".replace(".", "p")


def result_dir(lambda_kd):
    run_tag = (
        f"lkd{tag_float(lambda_kd)}"
        f"_laug{tag_float(LAMBDA_AUG)}"
        f"_T{tag_float(TEMPERATURE)}"
        f"_seed{SEED}"
    )
    return ROOT / "kd_results" / f"DNA-646_Train_5fold_KD_CONFIDENCE_V2_{run_tag}"


def summary_path(lambda_kd):
    return result_dir(lambda_kd) / "cv_summary.json"


def is_complete(lambda_kd):
    p = summary_path(lambda_kd)
    if not p.exists():
        return False
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False

    return (
        obj.get("dataset") == "DNA-646_Train"
        and obj.get("smoke") is False
        and int(obj.get("n_splits", 0)) == 5
        and abs(float(obj.get("lambda_kd", -999)) - lambda_kd) < 1e-9
        and int(obj.get("seed", -1)) == SEED
    )


def run_one(lambda_kd):
    out = result_dir(lambda_kd)
    print("\n" + "=" * 90, flush=True)
    print(f"DNA-646 | lambda_KD={lambda_kd:.1f}", flush=True)
    print("=" * 90, flush=True)

    if is_complete(lambda_kd):
        print(f"[SKIP] 已完成：{out}", flush=True)
        return True

    if out.exists() and any(out.iterdir()):
        print(
            f"[STOP] 发现非空但不完整目录：\n{out}\n"
            "请确认是中断残留后手动改名或删除，再重新运行本脚本。",
            flush=True,
        )
        return False

    cmd = [
        sys.executable,
        str(PIPELINE),
        "--dataset", DATASET,
        "--lambda-kd", str(lambda_kd),
        "--lambda-aug", str(LAMBDA_AUG),
        "--temperature", str(TEMPERATURE),
        "--seed", str(SEED),
    ]

    print("[RUN]", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = (time.time() - t0) / 60.0

    if proc.returncode != 0:
        print(f"[FAIL] lambda={lambda_kd:.1f}, returncode={proc.returncode}", flush=True)
        return False

    if not is_complete(lambda_kd):
        print("[FAIL] 运行结束但没有找到有效 cv_summary.json", flush=True)
        return False

    print(f"[OK] lambda={lambda_kd:.1f} 完成，耗时约 {elapsed:.1f} min", flush=True)
    return True


def load_result(lambda_kd):
    if not is_complete(lambda_kd):
        return None

    obj = json.loads(summary_path(lambda_kd).read_text(encoding="utf-8"))
    return {
        "lambda_kd": float(obj["lambda_kd"]),
        "oof_auroc": float(obj["oof_auroc"]),
        "oof_aupr": float(obj["oof_aupr"]),
        "oof_mcc": float(obj["oof_mcc"]),
        "oof_f1": float(obj["oof_f1"]),
        "oof_precision": float(obj.get("oof_precision", float("nan"))),
        "oof_sensitivity": float(obj.get("oof_sensitivity", float("nan"))),
        "oof_specificity": float(obj.get("oof_specificity", float("nan"))),
        "threshold": float(obj["oof_threshold_mcc"]),
        "median_best_epoch": int(obj["median_best_epoch"]),
        "result_dir": str(result_dir(lambda_kd)),
    }


def summarize():
    rows = [r for lam in LAMBDAS if (r := load_result(lam)) is not None]
    if not rows:
        print("[WARN] 没有可汇总结果")
        return

    best = max(
        rows,
        key=lambda r: (
            r["oof_mcc"],
            r["oof_f1"],
            r["oof_aupr"],
            r["oof_auroc"],
        ),
    )

    out_dir = ROOT / "kd_results" / "DNA-646_lambda_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "lambda_sweep_summary.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["lambda_kd"]))

    json_path = out_dir / "best_lambda.json"
    json_path.write_text(
        json.dumps(
            {
                "dataset": "DNA-646_Train",
                "seed": SEED,
                "selection_rule": "OOF MCC first; OOF F1 second; OOF AUPR third; OOF AUROC fourth",
                "best": best,
                "all_completed_results": sorted(rows, key=lambda x: x["lambda_kd"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("DNA-646 lambda_KD SEARCH SUMMARY")
    print("=" * 100)
    print(
        f"{'lambda':>8} {'AUROC':>9} {'AUPR':>9} {'MCC':>9} "
        f"{'F1':>9} {'Precision':>10} {'Recall':>9} {'threshold':>11} {'epoch':>7}"
    )
    print("-" * 100)

    for r in sorted(rows, key=lambda x: x["lambda_kd"]):
        mark = "  <-- BEST" if r["lambda_kd"] == best["lambda_kd"] else ""
        print(
            f"{r['lambda_kd']:>8.1f} "
            f"{r['oof_auroc']:>9.4f} "
            f"{r['oof_aupr']:>9.4f} "
            f"{r['oof_mcc']:>9.4f} "
            f"{r['oof_f1']:>9.4f} "
            f"{r['oof_precision']:>10.4f} "
            f"{r['oof_sensitivity']:>9.4f} "
            f"{r['threshold']:>11.6f} "
            f"{r['median_best_epoch']:>7d}"
            f"{mark}"
        )

    print("\n选择规则：MCC 第一，F1 第二，AUPR 第三，AUROC 第四。")
    print(
        f"当前最佳 lambda_KD={best['lambda_kd']:.1f} | "
        f"MCC={best['oof_mcc']:.4f}, F1={best['oof_f1']:.4f}, "
        f"AUPR={best['oof_aupr']:.4f}, AUROC={best['oof_auroc']:.4f}"
    )

    print("\n[SAVE]")
    print(csv_path)
    print(json_path)


def main():
    if not PIPELINE.exists():
        raise FileNotFoundError(
            f"找不到 {PIPELINE}，请把本脚本与 kd_cv_confidence_v2.py 都放在 E:\\zhq-ds-main\\"
        )

    print("=" * 90)
    print("DNA-646 三教师 DG-PBs-KD V2 lambda_KD 一次性搜索")
    print("=" * 90)
    print("搜索:", LAMBDAS)
    print("固定: lambda_aug=0.2, T=1.0, seed=42")
    print("已完成的lambda会自动SKIP；顺序运行，不并行抢GPU。")
    print("=" * 90)

    for lam in LAMBDAS:
        if not run_one(lam):
            print("\n[BATCH STOP] 处理问题后重新运行同一个脚本即可继续。")
            summarize()
            sys.exit(1)

    summarize()
    print("\n[ALL DONE] DNA-646 lambda_KD 搜索全部完成。")


if __name__ == "__main__":
    main()
