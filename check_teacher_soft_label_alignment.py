#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
严格检查 DG-PBs 原始训练数据 与 teacher soft-label CSV 是否逐残基对齐。

默认目录结构：
E:\zhq-ds-main\
├─ Raw_data\
│  ├─ DNA-573_Train.txt
│  └─ DNA-646_Train.txt
├─ teacher_soft_labels\
│  ├─ DNA-573_Train_teacher_soft_labels.csv
│  └─ DNA-646_Train_teacher_soft_labels.csv
└─ check_teacher_soft_label_alignment.py

运行：
    python .\check_teacher_soft_label_alignment.py

说明：
- DNA-573 每条记录可能有两行 0/1 标注；本脚本与原 DG-PBs 协议一致，只使用第一行标签。
- DNA-646 为常规 ID / sequence / label 格式。
"""

import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATASETS = [
    {
        "name": "DNA-573_Train",
        "raw": ROOT / "Raw_data" / "DNA-573_Train.txt",
        "soft": ROOT / "teacher_soft_labels" / "DNA-573_Train_teacher_soft_labels.csv",
        "expected_proteins": 573,
        "expected_residues": 159883,
    },
    {
        "name": "DNA-646_Train",
        "raw": ROOT / "Raw_data" / "DNA-646_Train.txt",
        "soft": ROOT / "teacher_soft_labels" / "DNA-646_Train_teacher_soft_labels.csv",
        "expected_proteins": 646,
        "expected_residues": 314139,
    },
]


def is_binary_line(s: str) -> bool:
    s = s.strip()
    return bool(s) and set(s) <= {"0", "1"}


def parse_raw(path: Path):
    """
    按 >protein_id 分块。
    每个块：
      第1个非空非二进制行 = sequence
      第1个二进制行 = DG-PBs 使用的 label
      后续二进制行（DNA-573 的第二标签）忽略
    """
    lines = [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    records = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        if not line.startswith(">"):
            raise ValueError(f"{path.name}: 第 {i+1} 行预期为 >protein_id，实际为: {line[:60]}")

        protein_id = line[1:].strip()
        if not protein_id:
            raise ValueError(f"{path.name}: 第 {i+1} 行 protein_id 为空")

        i += 1
        block = []
        while i < len(lines) and not lines[i].startswith(">"):
            if lines[i]:
                block.append(lines[i])
            i += 1

        if len(block) < 2:
            raise ValueError(f"{path.name}: {protein_id} 记录内容不足")

        sequence = None
        label = None

        for item in block:
            if sequence is None and not is_binary_line(item):
                sequence = item.strip().upper()
            elif label is None and is_binary_line(item):
                label = item.strip()

        if sequence is None:
            raise ValueError(f"{path.name}: {protein_id} 未找到氨基酸序列")
        if label is None:
            raise ValueError(f"{path.name}: {protein_id} 未找到第一行0/1标签")
        if len(sequence) != len(label):
            raise ValueError(
                f"{path.name}: {protein_id} 序列/标签长度不一致: "
                f"sequence={len(sequence)}, label={len(label)}"
            )

        records.append((protein_id, sequence, label))

    # 检查重复 protein id
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        seen = set()
        dup = []
        for x in ids:
            if x in seen and x not in dup:
                dup.append(x)
            seen.add(x)
        raise ValueError(f"{path.name}: 原始数据存在重复 protein_id: {dup[:10]}")

    return records


def normalize_pid(x: str) -> str:
    x = str(x).strip()
    if x.startswith(">"):
        x = x[1:]
    return x


def check_one(cfg):
    name = cfg["name"]
    raw_path = cfg["raw"]
    soft_path = cfg["soft"]

    print("=" * 72)
    print(f"[CHECK] {name}")
    print(f"raw : {raw_path}")
    print(f"soft: {soft_path}")

    if not raw_path.exists():
        print(f"[FAIL] 找不到原始文件: {raw_path}")
        return False
    if not soft_path.exists():
        print(f"[FAIL] 找不到 soft-label 文件: {soft_path}")
        return False

    # 1. 解析原始数据
    records = parse_raw(raw_path)
    raw_proteins = len(records)
    raw_residues = sum(len(seq) for _, seq, _ in records)

    # 建立逐残基期望字典
    expected = {}
    for pid, seq, labels in records:
        for idx, (aa, y) in enumerate(zip(seq, labels), start=1):
            expected[(pid, idx)] = (aa, int(y))

    # 2. 读 soft-label CSV
    with soft_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "protein_id", "residue_index", "amino_acid",
            "label", "teacher_soft_prob"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"[FAIL] CSV缺少必要列: {sorted(missing)}")
            return False

        csv_rows = 0
        csv_pids = set()
        seen_keys = set()
        min_prob = float("inf")
        max_prob = float("-inf")

        errors = []
        per_pid_last_idx = {}

        for line_no, row in enumerate(reader, start=2):
            csv_rows += 1

            pid = normalize_pid(row["protein_id"])
            try:
                idx = int(float(row["residue_index"]))
                csv_label = int(float(row["label"]))
                prob = float(row["teacher_soft_prob"])
            except Exception as e:
                errors.append(f"CSV第{line_no}行数值解析失败: {e}")
                if len(errors) >= 20:
                    break
                continue

            aa = str(row["amino_acid"]).strip().upper()

            key = (pid, idx)

            if key in seen_keys:
                errors.append(f"CSV第{line_no}行重复残基键: {key}")
            seen_keys.add(key)
            csv_pids.add(pid)

            # 概率合法性
            if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
                errors.append(f"CSV第{line_no}行 teacher_soft_prob 非法: {prob}")
            else:
                min_prob = min(min_prob, prob)
                max_prob = max(max_prob, prob)

            # 每条蛋白 residue_index 必须从1连续递增
            expected_idx = per_pid_last_idx.get(pid, 0) + 1
            if idx != expected_idx:
                errors.append(
                    f"CSV第{line_no}行索引不连续: {pid}, "
                    f"expected={expected_idx}, got={idx}"
                )
            per_pid_last_idx[pid] = idx

            # 与原始数据逐残基核对
            if key not in expected:
                errors.append(f"CSV第{line_no}行在原始数据中找不到: {key}")
            else:
                raw_aa, raw_label = expected[key]
                if aa != raw_aa:
                    errors.append(
                        f"氨基酸不匹配 {key}: raw={raw_aa}, csv={aa}"
                    )
                if csv_label != raw_label:
                    errors.append(
                        f"标签不匹配 {key}: raw={raw_label}, csv={csv_label}"
                    )

            if len(errors) >= 20:
                break

    # 如果前20条没出错，再检查是否有原始残基缺失
    if not errors:
        missing_keys = set(expected.keys()) - seen_keys
        extra_keys = seen_keys - set(expected.keys())
        if missing_keys:
            sample = list(sorted(missing_keys))[:10]
            errors.append(f"CSV缺少 {len(missing_keys)} 个原始残基，例如: {sample}")
        if extra_keys:
            sample = list(sorted(extra_keys))[:10]
            errors.append(f"CSV多出 {len(extra_keys)} 个残基，例如: {sample}")

    # 汇总
    print(f"原始蛋白数  : {raw_proteins}")
    print(f"原始残基数  : {raw_residues}")
    print(f"CSV蛋白数   : {len(csv_pids)}")
    print(f"CSV残基数   : {csv_rows}")

    if min_prob != float("inf"):
        print(f"soft概率范围: [{min_prob:.6f}, {max_prob:.6f}]")

    # 已知数量提示（不是唯一判定标准）
    if raw_proteins != cfg["expected_proteins"]:
        print(
            f"[WARN] 蛋白数与预期不同: expected={cfg['expected_proteins']}, "
            f"actual={raw_proteins}"
        )
    if raw_residues != cfg["expected_residues"]:
        print(
            f"[WARN] 残基数与预期不同: expected={cfg['expected_residues']}, "
            f"actual={raw_residues}"
        )

    if errors:
        print("\n[FAIL] 对齐检查未通过，前若干问题如下：")
        for e in errors[:20]:
            print("  -", e)
        return False

    if csv_rows != raw_residues:
        print(f"[FAIL] CSV残基数 != 原始残基数: {csv_rows} != {raw_residues}")
        return False

    if len(csv_pids) != raw_proteins:
        print(f"[FAIL] CSV蛋白数 != 原始蛋白数: {len(csv_pids)} != {raw_proteins}")
        return False

    print("[PASS] protein_id / residue_index / amino_acid / label / soft_prob 全部检查通过")
    return True


def main():
    print("\nDG-PBs-KD teacher soft-label 严格对齐检查\n")

    results = []
    for cfg in DATASETS:
        try:
            results.append(check_one(cfg))
        except Exception as e:
            print(f"[FAIL] {cfg['name']} 检查过程中发生异常: {e}")
            results.append(False)

    print("\n" + "=" * 72)
    if all(results):
        print("[ALL PASS] 两个训练集均严格对齐，可以进入 DG-PBs-KD 数据加载修改。")
        sys.exit(0)
    else:
        print("[STOP] 至少一个训练集未通过检查，先不要开始KD训练。")
        sys.exit(1)


if __name__ == "__main__":
    main()
