# -*- coding: utf-8 -*-
"""
DG-PBs-KD data loader

与原 data_loader.py 保持同一套 ESM-2 + KNN 图构建思路，
但新增：
1) teacher_prob: 每个真实残基的教师连续软标签
2) is_real: 标记真实残基（当前原始蛋白全部为 True）
3) has_teacher: 当前蛋白是否具有教师软标签
4) 严格检查序列/标签/soft-label 对齐，禁止长度不一致时静默改成全0

训练集：
    soft_label_csv = teacher_soft_labels/DNA-573_Train_teacher_soft_labels.csv
或：
    soft_label_csv = teacher_soft_labels/DNA-646_Train_teacher_soft_labels.csv

独立测试集：
    soft_label_csv = None
测试时不需要教师。
"""

import csv
import glob
import os
import re
import traceback
from collections import defaultdict

import esm
import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data


class KDProteinDataset:
    def __init__(
        self,
        path,
        soft_label_csv=None,
        device="cpu",
        knn_k=7,
        max_records=None,
        verbose=True,
    ):
        self.path = path
        self.soft_label_csv = soft_label_csv
        self.device = device
        self.knn_k = knn_k
        self.max_records = max_records
        self.verbose = verbose

        self.teacher_map = None
        if soft_label_csv is not None:
            self.teacher_map = self._load_teacher_soft_labels(soft_label_csv)

        # 与原 DG-PBs 一致：ESM-2 t33 650M，1280维残基表示
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        self.proteins = self.load_all()

    @staticmethod
    def _normalize_pid(pid):
        pid = str(pid).strip()
        if pid.startswith(">"):
            pid = pid[1:]
        return pid

    def _load_teacher_soft_labels(self, csv_path):
        csv_path = os.path.abspath(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 teacher soft-label 文件: {csv_path}")

        required = {
            "protein_id",
            "residue_index",
            "amino_acid",
            "label",
            "teacher_soft_prob",
        }

        by_pid = defaultdict(dict)

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"teacher soft-label CSV 缺少必要列: {sorted(missing)}"
                )

            for line_no, row in enumerate(reader, start=2):
                pid = self._normalize_pid(row["protein_id"])
                idx = int(float(row["residue_index"]))
                aa = str(row["amino_acid"]).strip().upper()
                label = int(float(row["label"]))
                prob = float(row["teacher_soft_prob"])

                if idx < 1:
                    raise ValueError(f"CSV第{line_no}行 residue_index 非法: {idx}")
                if label not in (0, 1):
                    raise ValueError(f"CSV第{line_no}行 label 非0/1: {label}")
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(
                        f"CSV第{line_no}行 teacher_soft_prob 越界: {prob}"
                    )

                if idx in by_pid[pid]:
                    raise ValueError(
                        f"teacher soft-label 出现重复残基: {pid}, residue={idx}"
                    )

                by_pid[pid][idx] = (aa, label, prob)

        if self.verbose:
            n_res = sum(len(v) for v in by_pid.values())
            print(
                f"[KD] 已加载 teacher soft labels: "
                f"{len(by_pid)} proteins, {n_res} residues"
            )

        return dict(by_pid)

    @staticmethod
    def _is_binary_line(s):
        s = s.strip()
        return bool(s) and set(s) <= {"0", "1"}

    def _parse_record_lines(self, record_lines, file_name, record_index):
        """
        支持常规三行记录，也支持 DNA-573 的额外第4行标签。
        与原 DG-PBs 协议保持一致：使用记录中第一行0/1标签。
        """
        if len(record_lines) < 3:
            return None

        name_line = record_lines[0].strip()
        name_match = re.match(r">(\S+)", name_line)
        name = (
            name_match.group(1)
            if name_match
            else f"{os.path.basename(file_name)}_record{record_index}"
        )
        name = self._normalize_pid(name)

        block = [x.strip() for x in record_lines[1:] if x.strip()]

        sequence = None
        label_line = None

        for item in block:
            if sequence is None and not self._is_binary_line(item):
                sequence = item.upper()
            elif label_line is None and self._is_binary_line(item):
                label_line = item

        if sequence is None:
            raise ValueError(f"{name}: 未找到氨基酸序列")
        if label_line is None:
            raise ValueError(f"{name}: 未找到0/1标签行")

        labels = [int(c) for c in label_line]

        # 这里不再像原 data_loader.py 那样静默替换为全0
        if len(labels) != len(sequence):
            raise ValueError(
                f"{name}: sequence/label 长度不一致: "
                f"{len(sequence)} != {len(labels)}"
            )

        return name, sequence, labels

    def _get_teacher_tensor(self, name, sequence, labels):
        n = len(sequence)

        # 独立测试集不需要teacher
        if self.teacher_map is None:
            teacher_prob = torch.full((n,), float("nan"), dtype=torch.float32)
            return teacher_prob, False

        if name not in self.teacher_map:
            raise KeyError(
                f"{name}: 在 teacher soft-label CSV 中找不到该蛋白。"
            )

        rows = self.teacher_map[name]

        if len(rows) != n:
            raise ValueError(
                f"{name}: teacher残基数与sequence不一致: "
                f"{len(rows)} != {n}"
            )

        probs = []
        for idx, (aa_raw, y_raw) in enumerate(zip(sequence, labels), start=1):
            if idx not in rows:
                raise KeyError(f"{name}: teacher CSV 缺少 residue_index={idx}")

            aa_csv, y_csv, prob = rows[idx]

            if aa_csv != aa_raw:
                raise ValueError(
                    f"{name} residue {idx}: amino_acid不匹配 "
                    f"raw={aa_raw}, teacher={aa_csv}"
                )
            if y_csv != y_raw:
                raise ValueError(
                    f"{name} residue {idx}: label不匹配 "
                    f"raw={y_raw}, teacher={y_csv}"
                )

            probs.append(prob)

        return torch.tensor(probs, dtype=torch.float32), True

    def _create_knn_graph(self, features, k=7):
        try:
            n = len(features)

            if n <= 1:
                return torch.empty((2, 0), dtype=torch.long)

            # 小蛋白时避免 n_neighbors >= n_samples
            effective_k = min(k, n - 1)

            adj = kneighbors_graph(
                features,
                effective_k,
                mode="connectivity",
                include_self=False,
            )
            return torch.tensor(np.array(adj.nonzero()), dtype=torch.long)

        except Exception as e:
            raise RuntimeError(f"KNN graph 构建失败: {e}")

    def _build_graph(self, name, sequence, labels):
        teacher_prob, has_teacher = self._get_teacher_tensor(
            name, sequence, labels
        )

        batch = [(name, sequence)]
        _, _, batch_tokens = self.batch_converter(batch)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])

        token_representations = results["representations"][33]
        embeddings = token_representations[0, 1 : len(sequence) + 1, :]

        if embeddings.size(0) != len(sequence):
            raise ValueError(
                f"{name}: ESM残基数异常 "
                f"{embeddings.size(0)} != {len(sequence)}"
            )

        protein_context = embeddings.mean(dim=0)
        edge_index = self._create_knn_graph(
            embeddings.detach().cpu().numpy(),
            k=self.knn_k,
        )

        return Data(
            x=embeddings.detach().cpu(),
            edge_index=edge_index,
            y=torch.tensor(labels, dtype=torch.long),

            # ---- DG-PBs-KD 新增字段 ----
            teacher_prob=teacher_prob,
            is_real=torch.ones(len(sequence), dtype=torch.bool),
            has_teacher=torch.tensor([has_teacher], dtype=torch.bool),

            protein_context=protein_context.detach().cpu(),
            name=name,
            sequence=sequence,
        )

    def load_all(self):
        proteins = []

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据路径不存在: {self.path}")

        # 支持直接传单个txt，也支持目录
        if os.path.isfile(self.path):
            files = [self.path]
        else:
            files = sorted(glob.glob(os.path.join(self.path, "*.txt")))

        if not files:
            raise FileNotFoundError(f"没有找到txt数据文件: {self.path}")

        if self.verbose:
            print(f"[KD] Found {len(files)} txt file(s)")

        count = 0

        for file in files:
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().splitlines()

                starts = [
                    i for i, line in enumerate(content)
                    if line.strip().startswith(">")
                ]
                starts.append(len(content))

                for i in range(len(starts) - 1):
                    start = starts[i]
                    end = starts[i + 1]
                    record_lines = content[start:end]

                    parsed = self._parse_record_lines(
                        record_lines, file, i + 1
                    )
                    if parsed is None:
                        continue

                    name, sequence, labels = parsed
                    data = self._build_graph(name, sequence, labels)
                    proteins.append(data)
                    count += 1

                    if self.verbose:
                        print(
                            f"[KD] Loaded {name}: "
                            f"{len(sequence)} residues, "
                            f"{sum(labels)} positive, "
                            f"teacher={'yes' if bool(data.has_teacher.item()) else 'no'}"
                        )

                    if (
                        self.max_records is not None
                        and count >= self.max_records
                    ):
                        return proteins

            except Exception:
                print(
                    f"[KD] Error processing {file}:\n"
                    f"{traceback.format_exc()}"
                )
                raise

        return proteins

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        return self.proteins[idx]


# 兼容后续代码中习惯写 ProteinDataset(...)
ProteinDataset = KDProteinDataset
