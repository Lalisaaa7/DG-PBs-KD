# -*- coding: utf-8 -*-
"""
DG-PBs-KD augmentation

关键规则：
1) 只对训练折调用本函数；
2) 原始真实残基：is_real=True，保留 teacher_prob；
3) DDPM 合成正节点：is_real=False，y=1，teacher_prob=NaN；
4) 合成节点绝不参与 KD；
5) 不删除真实残基；若图已达到 max_nodes_per_graph，则不再添加合成节点。
"""

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data


def _quality_filter(generated_samples, real_samples, threshold=0.7):
    if len(generated_samples) == 0 or len(real_samples) == 0:
        return generated_samples, 0.0

    distances = pairwise_distances(
        generated_samples, real_samples, metric="euclidean"
    )
    min_distances = np.min(distances, axis=1)

    max_dist = np.max(min_distances) + 1e-8
    scores = 1.0 - (min_distances / max_dist)
    mask = scores >= threshold

    return generated_samples[mask], float(np.mean(scores))


def _diversity_filter(samples, threshold=0.3):
    if len(samples) <= 1:
        return samples, 1.0

    distances = pairwise_distances(samples, metric="euclidean")
    np.fill_diagonal(distances, np.inf)

    keep = []
    for i in range(len(samples)):
        if len(keep) == 0 or np.min(distances[i]) >= threshold:
            keep.append(i)

    return samples[keep], len(keep) / len(samples)


def _create_knn_edges(x, k=7):
    n = x.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k = min(int(k), n - 1)
    adj = kneighbors_graph(
        x.detach().cpu().numpy(),
        n_neighbors=k,
        mode="connectivity",
        include_self=False,
    )
    return torch.tensor(np.array(adj.nonzero()), dtype=torch.long)


def _ensure_kd_fields(data):
    """确保真实图具有 teacher_prob / is_real 字段。"""
    n = data.y.numel()

    if not hasattr(data, "is_real"):
        data.is_real = torch.ones(n, dtype=torch.bool)

    if not hasattr(data, "teacher_prob"):
        data.teacher_prob = torch.full(
            (n,), float("nan"), dtype=torch.float32
        )

    if data.is_real.numel() != n:
        raise ValueError(
            f"{data.name}: is_real长度错误 "
            f"{data.is_real.numel()} != {n}"
        )

    if data.teacher_prob.numel() != n:
        raise ValueError(
            f"{data.name}: teacher_prob长度错误 "
            f"{data.teacher_prob.numel()} != {n}"
        )

    return data


def kd_augment_dataset(dataset, diffusion_model, config):
    """
    对传入的“当前训练折”进行增强。

    注意：
    dataset 必须已经是 train_fold；
    不允许把完整训练集传进来后再做CV划分。
    """
    augmented = []
    quality_stats = []
    diversity_stats = []

    for data in dataset:
        data = _ensure_kd_fields(data)

        # 输入必须是纯真实图。避免重复增强。
        if not bool(data.is_real.all()):
            raise ValueError(
                f"{data.name}: 输入图中已经存在 synthetic node，"
                "请勿重复增强。"
            )

        pos_mask = data.y == 1
        n_pos = int(pos_mask.sum().item())
        n_neg = int((data.y == 0).sum().item())
        n_real = int(data.y.numel())

        if n_pos == 0:
            augmented.append(data)
            continue

        # 不删除真实残基；只根据剩余容量决定最多可加多少合成节点。
        max_nodes = int(getattr(config, "max_nodes_per_graph", 10**9))
        capacity = max(0, max_nodes - n_real)
        if capacity <= 0:
            augmented.append(data)
            continue

        target_ratio = float(getattr(config, "target_ratio", 0.9))
        min_samples = int(getattr(config, "min_samples_per_protein", 5))
        max_augment_ratio = float(getattr(config, "max_augment_ratio", 2.0))

        target_pos = int((n_pos + n_neg) * target_ratio)
        n_to_generate = max(min_samples, target_pos - n_pos)
        n_to_generate = min(
            n_to_generate,
            int(max(1, n_pos) * max_augment_ratio),
            capacity,
        )

        if n_to_generate <= 0:
            augmented.append(data)
            continue

        protein_context = data.protein_context.to(config.device)

        # 多生成候选样本用于质量筛选。
        candidates = diffusion_model.generate_positive_sample(
            protein_context,
            num_samples=max(n_to_generate * 3, n_to_generate),
            verbose=bool(getattr(config, "verbose_loading", False)),
        )

        if candidates is None or len(candidates) == 0:
            augmented.append(data)
            continue

        candidates = np.asarray(candidates, dtype=np.float32)
        real_pos = data.x[pos_mask].detach().cpu().numpy()

        quality_threshold = float(
            getattr(config, "quality_threshold", 0.7)
        )
        diversity_threshold = float(
            getattr(config, "diversity_threshold", 0.3)
        )

        quality_samples, q_score = _quality_filter(
            candidates,
            real_pos,
            threshold=quality_threshold,
        )

        if len(quality_samples) > 0:
            diverse_samples, d_score = _diversity_filter(
                quality_samples,
                threshold=diversity_threshold,
            )
        else:
            # 保留原工程的回退思路，但只取目标数量。
            diverse_samples = candidates[:n_to_generate]
            d_score = 0.5

        final_samples = diverse_samples[:n_to_generate]

        if len(final_samples) == 0:
            augmented.append(data)
            continue

        quality_stats.append(q_score)
        diversity_stats.append(d_score)

        new_x = torch.as_tensor(final_samples, dtype=torch.float32)
        n_syn = new_x.size(0)

        new_y = torch.ones(n_syn, dtype=torch.long)
        new_is_real = torch.zeros(n_syn, dtype=torch.bool)
        new_teacher = torch.full(
            (n_syn,), float("nan"), dtype=torch.float32
        )

        updated_x = torch.cat([data.x.cpu(), new_x], dim=0)
        updated_y = torch.cat([data.y.cpu(), new_y], dim=0)
        updated_is_real = torch.cat(
            [data.is_real.cpu(), new_is_real], dim=0
        )
        updated_teacher = torch.cat(
            [data.teacher_prob.cpu(), new_teacher], dim=0
        )

        edge_index = _create_knn_edges(
            updated_x,
            k=int(getattr(config, "knn_k", 7)),
        )

        kwargs = dict(
            x=updated_x,
            edge_index=edge_index,
            y=updated_y,
            teacher_prob=updated_teacher,
            is_real=updated_is_real,
            protein_context=data.protein_context.cpu(),
            name=str(data.name) + "_kd_aug",
            original_name=str(data.name),
            original_num_nodes=torch.tensor([n_real], dtype=torch.long),
            synthetic_num_nodes=torch.tensor([n_syn], dtype=torch.long),
        )

        if hasattr(data, "has_teacher"):
            kwargs["has_teacher"] = data.has_teacher.cpu()

        # sequence只对应真实残基，因此作为元数据保留，不用于增强节点索引。
        if hasattr(data, "sequence"):
            kwargs["sequence"] = data.sequence

        aug_graph = Data(**kwargs)

        # 最终强校验
        total = aug_graph.y.numel()
        assert aug_graph.x.size(0) == total
        assert aug_graph.is_real.numel() == total
        assert aug_graph.teacher_prob.numel() == total
        assert torch.all(aug_graph.y[~aug_graph.is_real] == 1)
        assert torch.isnan(
            aug_graph.teacher_prob[~aug_graph.is_real]
        ).all()

        augmented.append(aug_graph)

    if quality_stats:
        print(
            "[KD-AUG] 完成："
            f"平均质量={np.mean(quality_stats):.3f}, "
            f"平均多样性={np.mean(diversity_stats):.3f}"
        )

    return augmented
