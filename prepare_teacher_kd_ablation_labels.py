# -*- coding: utf-8 -*-
"""
Prepare DNA-573 teacher-composition ablation soft labels.

Creates:
1) TransBind-only
2) ESM-NBR + TransBind (equal soft voting)
3) CLAPE + ESM-NBR + TransBind (equal soft voting)

All are built from the same current residue_predictions.csv.
No independent-test labels are used.
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = (
    ROOT / "ThreeTeacher_DNA_Predictor_ReadyCode"
    / "ThreeTeacher_DNA_Predictor"
    / "outputs" / "DNA_573_Train"
    / "residue_predictions.csv"
)
OUT_DIR = ROOT / "teacher_ablation_soft_labels"

VARIANTS = {
    "transbind_only": ["transbind_prob"],
    "clape_trans_equal": ["clape_prob", "transbind_prob"],
    "esm_trans_equal": ["esm_nbr_prob", "transbind_prob"],
    "triple_equal": ["clape_prob", "esm_nbr_prob", "transbind_prob"],
}


def main():
    if not SRC.exists():
        raise FileNotFoundError(f"找不到：{SRC}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with SRC.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {
        "protein_id", "residue_index", "amino_acid", "label",
        "clape_prob", "esm_nbr_prob", "transbind_prob"
    }
    missing = required - fields
    if missing:
        raise ValueError(f"源CSV缺少列：{sorted(missing)}")

    for variant, cols in VARIANTS.items():
        out = OUT_DIR / f"DNA-573_Train_{variant}.csv"

        fieldnames = [
            "protein_id", "residue_index", "amino_acid", "label",
            "teacher_soft_prob", "teacher_confidence",
            "teacher_disagreement_std",
        ]

        confs = []
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in rows:
                ps = [float(r[c]) for c in cols]
                q = sum(ps) / len(ps)

                if len(ps) == 1:
                    std = 0.0
                    conf = 1.0
                else:
                    var = sum((p - q) ** 2 for p in ps) / len(ps)
                    std = math.sqrt(max(var, 0.0))
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
            f"[OK] {variant:16s} -> {out}\n"
            f"     residues={len(rows)}, "
            f"confidence mean={sum(confs)/len(confs):.4f}, "
            f"min={min(confs):.4f}, max={max(confs):.4f}"
        )

    print("\n[ALL DONE] teacher-composition soft labels generated.")
    print("Independent test labels used: NO")


if __name__ == "__main__":
    main()
