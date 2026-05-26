# train_stgcn_biisc.py
"""
Обучение Simple ST-GCN на COCO-17 ключевых точках из .npz (BIISC + YOLOPose).

Запуск:
python train_stgcn_biisc.py --config config_stgcn.yaml
python train_stgcn_biisc.py --config config_stgcn.yaml --exp-name run_v2

TensorBoard:
tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

ACTION_TO_IDX: Dict[str, int] = {
    "CALL": 0, "COUG": 1, "DRIN": 2, "SCRA": 3,
    "SNEE": 4, "STRE": 5, "WAVE": 6, "WIPE": 7,
}
IDX_TO_ACTION: Dict[int, str] = {v: k for k, v in ACTION_TO_IDX.items()}
CLASS_NAMES: List[str] = [k for k, _ in sorted(ACTION_TO_IDX.items(), key=lambda kv: kv[1])]

# COCO-17 skeleton (для графа)
COCO_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # голова
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # руки
    (5, 11), (6, 12), (11, 12),                 # торс
    (11, 13), (13, 15), (12, 14), (14, 16),     # ноги
]

# COCO-17: пары суставов лево↔право (для горизонтального флипа)
LR_JOINT_PAIRS: List[Tuple[int, int]] = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]

# ─────────────────────────────────────────────────────────────────────────────
# Конфиг и устройство
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_device(gpu_id: int = 0) -> torch.device:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"{'='*60}")
        print(f"🚀 GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"   CUDA: {torch.version.cuda} | Torch: {torch.__version__}")
        print(f"{'='*60}\n")
    else:
        device = torch.device("cpu")
        print(f"{'='*60}")
        print("⚠️  GPU недоступен — используем CPU")
        print(f"{'='*60}\n")
    return device

# ─────────────────────────────────────────────────────────────────────────────
# Граф смежности COCO-17
# ─────────────────────────────────────────────────────────────────────────────

def get_coco_adjacency(num_joints: int = 17) -> torch.Tensor:
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in COCO_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(num_joints, dtype=np.float32)
    d = A.sum(axis=1)
    d_inv = np.where(d > 0, d ** -0.5, 0.0)
    D = np.diag(d_inv)
    A_norm = D @ A @ D
    return torch.from_numpy(A_norm)  # (17,17)

# ─────────────────────────────────────────────────────────────────────────────
# Аугментации скелета
# ─────────────────────────────────────────────────────────────────────────────

def _skeleton_center(kpts: np.ndarray) -> Tuple[float, float]:
    visible = kpts[:, :, 2] > 0.1
    if visible.sum() == 0:
        return 0.5, 0.5
    return float(kpts[:, :, 0][visible].mean()), float(kpts[:, :, 1][visible].mean())

def aug_gaussian_noise(kpts: np.ndarray, std: float) -> np.ndarray:
    out = kpts.copy()
    visible = (out[:, :, 2:3] > 0.1).astype(np.float32)
    noise = np.random.normal(0.0, std, out[:, :, :2].shape).astype(np.float32)
    out[:, :, :2] += noise * visible
    return out

def aug_joint_dropout(kpts: np.ndarray, prob: float) -> np.ndarray:
    out = kpts.copy()
    mask = np.random.rand(17) < prob
    out[:, mask, :] = 0.0
    return out

def aug_conf_noise(kpts: np.ndarray, std: float) -> np.ndarray:
    out = kpts.copy()
    noise = np.random.normal(0.0, std, out[:, :, 2].shape).astype(np.float32)
    out[:, :, 2] = np.clip(out[:, :, 2] + noise, 0.0, 1.0)
    return out

def aug_temporal_crop(kpts: np.ndarray, max_frames: int) -> np.ndarray:
    T = kpts.shape[0]
    if T <= max_frames:
        return kpts
    start = np.random.randint(0, T - max_frames + 1)
    return kpts[start: start + max_frames].copy()

def aug_temporal_speed(kpts: np.ndarray, speed_range: Tuple[float, float]) -> np.ndarray:
    T = kpts.shape[0]
    speed = np.random.uniform(*speed_range)
    new_T = max(2, int(T / speed))
    old_idx = np.linspace(0, T - 1, new_T)
    new_kpts = np.zeros((new_T, kpts.shape[1], kpts.shape[2]), dtype=np.float32)
    for i in range(new_T):
        lo = int(old_idx[i])
        hi = min(lo + 1, T - 1)
        frac = old_idx[i] - lo
        new_kpts[i] = kpts[lo] * (1.0 - frac) + kpts[hi] * frac
    return new_kpts

def aug_horizontal_flip(kpts: np.ndarray) -> np.ndarray:
    out = kpts.copy()
    out[:, :, 0] = 1.0 - out[:, :, 0]
    for left, right in LR_JOINT_PAIRS:
        out[:, [left, right], :] = out[:, [right, left], :]
    return out

def aug_zoom(kpts: np.ndarray, zoom_range: Tuple[float, float]) -> np.ndarray:
    out = kpts.copy()
    scale = np.random.uniform(*zoom_range)
    cx, cy = _skeleton_center(out)
    out[:, :, 0] = cx + (out[:, :, 0] - cx) * scale
    out[:, :, 1] = cy + (out[:, :, 1] - cy) * scale
    return out

def aug_translate(kpts: np.ndarray, std: float) -> np.ndarray:
    out = kpts.copy()
    out[:, :, 0] += np.random.normal(0.0, std)
    out[:, :, 1] += np.random.normal(0.0, std)
    return out

def aug_rotate_2d(kpts: np.ndarray, angle_range: Tuple[float, float]) -> np.ndarray:
    out = kpts.copy()
    rad = np.deg2rad(np.random.uniform(*angle_range))
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    cx, cy = _skeleton_center(out)
    x = out[:, :, 0] - cx
    y = out[:, :, 1] - cy
    out[:, :, 0] = cx + x * cos_a - y * sin_a
    out[:, :, 1] = cy + x * sin_a + y * cos_a
    return out

def aug_perspective_x(kpts: np.ndarray, strength_range: Tuple[float, float]) -> np.ndarray:
    out = kpts.copy()
    strength = np.random.uniform(*strength_range)
    cx, cy = _skeleton_center(out)
    out[:, :, 0] = cx + (out[:, :, 0] - cx) + strength * (out[:, :, 1] - cy)
    return out

def aug_perspective_y(kpts: np.ndarray, strength_range: Tuple[float, float]) -> np.ndarray:
    out = kpts.copy()
    strength = np.random.uniform(*strength_range)
    cx, cy = _skeleton_center(out)
    out[:, :, 1] = cy + (out[:, :, 1] - cy) + strength * (out[:, :, 0] - cx)
    return out

def aug_scale_bones(kpts: np.ndarray, scale_range: Tuple[float, float]) -> np.ndarray:
    out = kpts.copy()
    cx, cy = _skeleton_center(out)
    out[:, :, 0] = cx + (out[:, :, 0] - cx) * np.random.uniform(*scale_range)
    out[:, :, 1] = cy + (out[:, :, 1] - cy) * np.random.uniform(*scale_range)
    return out


class SkeletonAugmentor:
    """
    Оркестратор аугментаций. Читает секцию augmentation: из YAML конфига.
    Применяет цепочку: temporal → cctv-angle → spatial → noise.
    """
    def __init__(self, aug_cfg: dict):
        self.cfg = aug_cfg or {}

    def __call__(self, kpts: np.ndarray, max_frames: int) -> np.ndarray:
        c = self.cfg

        # ── Temporal ──────────────────────────────────────────────────
        if c.get("temporal_crop", False):
            kpts = aug_temporal_crop(kpts, max_frames)
        if c.get("temporal_speed", False):
            kpts = aug_temporal_speed(kpts, tuple(c.get("speed_range", [0.8, 1.2])))

        # ── CCTV / angle ──────────────────────────────────────────────
        if c.get("horizontal_flip", False) and np.random.rand() < c.get("flip_prob", 0.5):
            kpts = aug_horizontal_flip(kpts)
        if c.get("rotate_2d", False):
            kpts = aug_rotate_2d(kpts, tuple(c.get("rotate_range", [-10, 10])))
        if c.get("perspective_x", False):
            kpts = aug_perspective_x(kpts, tuple(c.get("perspective_x_range", [-0.10, 0.10])))
        if c.get("perspective_y", False):
            kpts = aug_perspective_y(kpts, tuple(c.get("perspective_y_range", [-0.08, 0.08])))

        # ── Spatial ───────────────────────────────────────────────────
        if c.get("zoom", False):
            kpts = aug_zoom(kpts, tuple(c.get("zoom_range", [0.88, 1.12])))
        if c.get("translate", False):
            kpts = aug_translate(kpts, c.get("translate_std", 0.04))
        if c.get("scale_bones", False):
            kpts = aug_scale_bones(kpts, tuple(c.get("scale_bones_range", [0.88, 1.12])))

        # ── Noise ─────────────────────────────────────────────────────
        if c.get("joint_dropout", False):
            kpts = aug_joint_dropout(kpts, c.get("joint_dropout_prob", 0.08))
        if c.get("noise_std", 0.0) > 0:
            kpts = aug_gaussian_noise(kpts, c["noise_std"])
        if c.get("conf_noise", False):
            kpts = aug_conf_noise(kpts, c.get("conf_noise_std", 0.08))

        return kpts


# ─────────────────────────────────────────────────────────────────────────────
# Загрузка .npz (BIISC)
# ─────────────────────────────────────────────────────────────────────────────

def _np_scalar_as_str(arr) -> str:
    if isinstance(arr, str):
        return arr
    if hasattr(arr, "item"):
        return str(arr.item())
    return str(arr)

def load_items(
    keypoints_dir: str,
    cfg: dict,
) -> Tuple[List[dict], List[dict], List[dict]]:
    kp_dir = Path(keypoints_dir)
    use_hf = cfg["data"].get("use_hf", True)
    test_subjects = set(cfg["data"]["test_subjects"])
    val_ratio = cfg["data"].get("val_split", 0.15)

    train_all: List[dict] = []
    test_items: List[dict] = []

    for npz_path in sorted(kp_dir.glob("*.npz")):
        if not use_hf and "_HF" in npz_path.stem:
            continue
        try:
            d = np.load(str(npz_path), allow_pickle=True)
        except Exception as exc:
            print(f"[WARN] {npz_path.name}: {exc}")
            continue

        action  = _np_scalar_as_str(d.get("action", ""))
        subject = _np_scalar_as_str(d.get("subject", ""))
        kpts    = d["keypoints"].astype(np.float32)  # (T,17,3)

        if action not in ACTION_TO_IDX:
            continue

        item = dict(kpts=kpts, label=ACTION_TO_IDX[action],
                    action=action, subject=subject, file=npz_path.name)
        if subject in test_subjects:
            test_items.append(item)
        else:
            train_all.append(item)

    if not train_all:
        raise RuntimeError(f"Нет train данных в {keypoints_dir}")

    y_train = [it["label"] for it in train_all]
    idx_train, idx_val = train_test_split(
        range(len(train_all)), test_size=val_ratio,
        stratify=y_train, random_state=42,
    )
    train_items = [train_all[i] for i in idx_train]
    val_items   = [train_all[i] for i in idx_val]

    print(f"Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")
    c = Counter(it["action"] for it in train_items)
    print("Train class counts:", {k: c.get(k, 0) for k in CLASS_NAMES})

    return train_items, val_items, test_items

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonDataset(Dataset):
    """
    Приводит (T,17,3) → (C,T,V,M) для ST-GCN.
    Аугментации управляются через SkeletonAugmentor (секция augmentation: в YAML).
    """

    def __init__(
        self,
        items: List[dict],
        max_frames: int = 50,
        in_channels: int = 3,
        augment: bool = False,
        aug_cfg: Optional[dict] = None,
    ):
        self.items       = items
        self.max_frames  = max_frames
        self.in_channels = in_channels
        self.augment     = augment
        self.augmentor   = SkeletonAugmentor(aug_cfg) if (augment and aug_cfg) else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it   = self.items[idx]
        kpts = it["kpts"].copy()  # (T, 17, 3)

        # ── Аугментация (до pad/trim — temporal_crop работает корректно) ──
        if self.augment and self.augmentor is not None:
            kpts = self.augmentor(kpts, self.max_frames)

        # ── Pad / trim до max_frames ──────────────────────────────────────
        T = kpts.shape[0]
        if T < self.max_frames:
            pad  = np.zeros((self.max_frames - T, 17, 3), dtype=np.float32)
            kpts = np.concatenate([kpts, pad], axis=0)
        else:
            kpts = kpts[: self.max_frames]

        # ── Выбор каналов ────────────────────────────────────────────────
        data = kpts[:, :, :2] if self.in_channels == 2 else kpts  # (T,17,C)

        # (T, V, C) → (C, T, V, 1)
        data = np.transpose(data, (2, 0, 1))[:, :, :, None]

        return torch.from_numpy(data).float(), torch.tensor(it["label"], dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# ST-GCN
# ─────────────────────────────────────────────────────────────────────────────

class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.shape
        x_flat = x.view(N, C * T, V)
        x_flat = torch.matmul(x_flat, A)
        x = x_flat.view(N, C, T, V)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialTemporalAttention(nn.Module):
    """
    Лёгкий spatial + temporal attention для тензора (N, C, T, V).

    Spatial : усредняем по T → (N,C,V) → MLP → (1,V) → sigmoid → веса суставов.
    Temporal: усредняем по V → (N,C,T) → MLP → (1,T) → sigmoid → веса кадров.

    Веса зависят от текущих признаков (input-dependent) — в отличие от
    статичного edge importance. Применяется после TCN внутри STGCNBlock.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(1, channels // reduction)

        # Spatial MLP: (1, C, V) → (1, 1, V)
        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(channels, r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(r, 1, kernel_size=1, bias=False),
        )

        # Temporal MLP: (1, C, T) → (1, 1, T)
        self.temporal_mlp = nn.Sequential(
            nn.Conv1d(channels, r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(r, 1, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        N, C, T, V = x.shape

        # Spatial: какие суставы важны
        xs = x.mean(dim=2)                      # (N, C, V)
        xs = xs.mean(dim=0, keepdim=True)        # (1, C, V)
        w_s = self.sigmoid(self.spatial_mlp(xs)) # (1, 1, V)
        w_s = w_s.view(1, 1, 1, V)              # broadcast по N,C,T

        # Temporal: какие кадры важны
        xt = x.mean(dim=3)                       # (N, C, T)
        xt = xt.mean(dim=0, keepdim=True)        # (1, C, T)
        w_t = self.sigmoid(self.temporal_mlp(xt))# (1, 1, T)
        w_t = w_t.view(1, 1, T, 1)              # broadcast по N,C,V

        return x * w_s * w_t


class STGCNBlock(nn.Module):
    """
    Один ST-GCN блок: GCN → TCN → ST-Attention (опц.) → Residual → ReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size_t: int = 9,
        stride_t: int = 1,
        dropout: float = 0.0,
        use_attention: bool = False,
        att_reduction: int = 16,
    ):
        super().__init__()

        self.gcn = GraphConvolution(in_channels, out_channels)

        padding_t = (kernel_size_t - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(kernel_size_t, 1),
                stride=(stride_t, 1),
                padding=(padding_t, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )

        # Spatial-Temporal Attention (опционально)
        self.attention = (
            SpatialTemporalAttention(out_channels, reduction=att_reduction)
            if use_attention else nn.Identity()
        )

        # Residual
        if in_channels != out_channels or stride_t != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride_t, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x   = self.gcn(x, A)
        x   = self.tcn(x)
        x   = self.attention(x)   # ST-Attention применяется после TCN
        x   = self.relu(x + res)
        return x


class STGCN_Simple(nn.Module):
    """
    ST-GCN с произвольным числом блоков + edge importance + ST-Attention.

    hidden_channels: список каналов, определяет число блоков.
    stride_at:       номера блоков (1-based) с temporal stride=2.
    use_edge_importance: обучаемая маска на матрицу смежности.
    use_st_attention:    spatial-temporal attention после TCN.
    att_from_block:      с какого блока (1-based) включать attention.
    att_reduction:       коэф. сжатия в attention MLP.
    """

    def __init__(
        self,
        num_class: int,
        num_joints: int = 17,
        in_channels: int = 3,
        hidden_channels: List[int] = None,
        dropout: float = 0.5,
        tcn_kernel_size: int = 9,
        stride_at: List[int] = None,
        use_edge_importance: bool = True,
        use_st_attention: bool = True,
        att_from_block: int = 1,
        att_reduction: int = 16,
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        if stride_at is None:
            stride_at = []

        A = get_coco_adjacency(num_joints)
        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        channels = [in_channels] + hidden_channels
        blocks = []
        for i, (c_in, c_out) in enumerate(zip(channels[:-1], channels[1:]), start=1):
            stride_t   = 2 if i in stride_at else 1
            blk_drop   = dropout if i == len(hidden_channels) else 0.0
            use_att    = use_st_attention and (i >= att_from_block)
            blocks.append(
                STGCNBlock(
                    c_in, c_out,
                    kernel_size_t=tcn_kernel_size,
                    stride_t=stride_t,
                    dropout=blk_drop,
                    use_attention=use_att,
                    att_reduction=att_reduction,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Edge importance: обучаемая маска (V,V) для каждого блока
        if use_edge_importance:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones_like(self.A)) for _ in self.blocks
            ])
        else:
            self.edge_importance = None

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_channels[-1], num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        x = x.squeeze(-1)  # (N, C, T, V)

        # BN на входе
        x_bn = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x_bn = self.data_bn(x_bn)
        x    = x_bn.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for i, block in enumerate(self.blocks):
            A_eff = self.A * self.edge_importance[i] if self.edge_importance else self.A
            x = block(x, A_eff)

        x = self.pool(x)       # (N, C_last, 1, 1)
        x = x.view(N, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_figure(cm: np.ndarray, labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig

def cfg_to_markdown(cfg: dict) -> str:
    lines = ["| Параметр | Значение |", "|---|---|"]
    def flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                flatten(v, prefix=f"{key}.")
            else:
                lines.append(f"| `{key}` | `{v}` |")
    flatten(cfg)
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer = None,
    mode: str = "train",
    grad_clip: float = 0.0,
):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_true, all_pred = [], []

    pbar = tqdm(loader, desc=f"{mode}", ncols=100)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        out  = model(x)
        loss = criterion(out, y)

        if is_train:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = out.argmax(dim=1)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{(preds==y).float().mean().item():.4f}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    return {
        "loss":         running_loss / len(loader.dataset),
        "acc":          float((all_true == all_pred).mean()),
        "f1_macro":     float(f1_score(all_true, all_pred, average="macro",    zero_division=0)),
        "f1_weighted":  float(f1_score(all_true, all_pred, average="weighted", zero_division=0)),
        "per_class_f1": f1_score(all_true, all_pred, average=None,
                                 labels=list(range(len(CLASS_NAMES))), zero_division=0),
        "cm":      confusion_matrix(all_true, all_pred, labels=list(range(len(CLASS_NAMES)))),
        "y_true":  all_true,
        "y_pred":  all_pred,
    }

#
#  importance
#

# ─────────────────────────────────────────────────────────────────────────────
# Визуализация важности суставов
# ─────────────────────────────────────────────────────────────────────────────

# COCO-17 имена суставов
JOINT_NAMES = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

# COCO-17: позиции суставов для рисования скелета (x, y) в единицах изображения
# Примерная схема "человечка" сверху вниз
JOINT_POS = np.array([
    [0.50, 0.05],  # 0  nose
    [0.44, 0.10],  # 1  left_eye
    [0.56, 0.10],  # 2  right_eye
    [0.38, 0.13],  # 3  left_ear
    [0.62, 0.13],  # 4  right_ear
    [0.35, 0.28],  # 5  left_shoulder
    [0.65, 0.28],  # 6  right_shoulder
    [0.22, 0.45],  # 7  left_elbow
    [0.78, 0.45],  # 8  right_elbow
    [0.13, 0.60],  # 9  left_wrist
    [0.87, 0.60],  # 10 right_wrist
    [0.38, 0.58],  # 11 left_hip
    [0.62, 0.58],  # 12 right_hip
    [0.35, 0.77],  # 13 left_knee
    [0.65, 0.77],  # 14 right_knee
    [0.33, 0.95],  # 15 left_ankle
    [0.67, 0.95],  # 16 right_ankle
], dtype=np.float32)


def extract_edge_importance_per_joint(model: STGCN_Simple) -> np.ndarray:
    """
    Извлекает важность каждого сустава из edge importance матриц.
    Для каждого блока суммируем строки (сколько импульса идёт из каждого сустава).
    Возвращает (num_blocks, V) — среднее по строкам для каждого блока.
    """
    if model.edge_importance is None:
        return None

    result = []
    for ei in model.edge_importance:
        w = ei.detach().cpu().numpy()          # (V, V)
        joint_w = w.sum(axis=1)                # (V,) — суммарный исходящий вес
        joint_w = (joint_w - joint_w.min()) / (joint_w.max() - joint_w.min() + 1e-8)
        result.append(joint_w)
    return np.stack(result, axis=0)            # (num_blocks, V)


def collect_spatial_attention_weights(
    model: STGCN_Simple,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
) -> Optional[np.ndarray]:
    """
    Прогоняет num_batches батчей через модель, собирает spatial attention веса
    через forward hook на каждом SpatialTemporalAttention модуле.
    Возвращает (num_blocks, V) — усреднённые веса суставов.
    """
    att_weights: List[List[np.ndarray]] = [[] for _ in model.blocks]
    hooks = []

    for block_idx, block in enumerate(model.blocks):
        if not isinstance(block.attention, SpatialTemporalAttention):
            continue

        def make_hook(bidx):
            def hook(module, inp, out):
                # Восстанавливаем spatial weight из forward:
                # module — это SpatialTemporalAttention
                # inp[0]: (N, C, T, V)
                x = inp[0]
                N, C, T, V = x.shape
                xs = x.mean(dim=2)
                xs = xs.mean(dim=0, keepdim=True)
                w_s = module.sigmoid(module.spatial_mlp(xs))  # (1, 1, V)
                att_weights[bidx].append(w_s.squeeze().detach().cpu().numpy())
            return hook

        h = block.attention.register_forward_hook(make_hook(block_idx))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            model(x.to(device))

    for h in hooks:
        h.remove()

    result = []
    for bidx in range(len(model.blocks)):
        if att_weights[bidx]:
            mean_w = np.stack(att_weights[bidx], axis=0).mean(axis=0)  # (V,)
            mean_w = (mean_w - mean_w.min()) / (mean_w.max() - mean_w.min() + 1e-8)
            result.append(mean_w)
        else:
            result.append(np.ones(17) * 0.5)

    return np.stack(result, axis=0)  # (num_blocks, V)


def plot_skeleton_heatmap(
    joint_weights: np.ndarray,         # (V,) — веса суставов [0,1]
    title: str = "Joint Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Рисует скелет COCO-17 с суставами, закрашенными по важности.
    Цвет: синий (мало) → красный (много).
    Размер узла пропорционален важности.
    """
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)

    cmap = plt.cm.RdYlBu_r

    # Рёбра скелета
    for i, j in COCO_EDGES:
        xi, yi = JOINT_POS[i]
        xj, yj = JOINT_POS[j]
        # Цвет ребра = среднее двух суставов
        edge_w = (joint_weights[i] + joint_weights[j]) / 2.0
        ax.plot([xi, xj], [yi, yj],
                color=cmap(edge_w), linewidth=2.5, alpha=0.7, zorder=1)

    # Узлы суставов
    scatter = ax.scatter(
        JOINT_POS[:, 0], JOINT_POS[:, 1],
        c=joint_weights,
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        s=150 + joint_weights * 600,   # размер пропорционален важности
        zorder=2,
        edgecolors="white",
        linewidths=1.5,
    )

    # Подписи суставов
    short_names = [
        "nose", "L.eye", "R.eye", "L.ear", "R.ear",
        "L.sho", "R.sho", "L.elb", "R.elb", "L.wri", "R.wri",
        "L.hip", "R.hip", "L.kne", "R.kne", "L.ank", "R.ank",
    ]
    offsets = [
        (-0.06,  0.00), (-0.07, -0.01), ( 0.06, -0.01),
        (-0.08,  0.00), ( 0.07,  0.00),
        (-0.09,  0.00), ( 0.08,  0.00),
        (-0.09,  0.00), ( 0.08,  0.00),
        (-0.09,  0.00), ( 0.08,  0.00),
        (-0.09,  0.00), ( 0.08,  0.00),
        (-0.09,  0.00), ( 0.08,  0.00),
        (-0.09,  0.00), ( 0.08,  0.00),
    ]
    for vi in range(17):
        ox, oy = offsets[vi]
        ax.text(
            JOINT_POS[vi, 0] + ox,
            JOINT_POS[vi, 1] + oy,
            short_names[vi],
            fontsize=7,
            ha="center", va="center",
            color="black",
            alpha=0.85,
        )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Importance", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_importance_per_block(
    weights_per_block: np.ndarray,    # (num_blocks, V)
    source_name: str = "Edge Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Рисует решётку skeleton heatmap — по одному скелету на каждый блок.
    """
    num_blocks = weights_per_block.shape[0]
    cols = min(num_blocks, 3)
    rows = (num_blocks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                              figsize=(4.5 * cols, 7.5 * rows))
    axes = np.array(axes).reshape(-1)

    cmap = plt.cm.RdYlBu_r

    for bidx in range(num_blocks):
        ax = axes[bidx]
        w  = weights_per_block[bidx]

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(1.05, -0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Block {bidx + 1}", fontsize=11)

        for i, j in COCO_EDGES:
            xi, yi = JOINT_POS[i]
            xj, yj = JOINT_POS[j]
            edge_w = (w[i] + w[j]) / 2.0
            ax.plot([xi, xj], [yi, yj],
                    color=cmap(edge_w), linewidth=2.5, alpha=0.7)

        sc = ax.scatter(
            JOINT_POS[:, 0], JOINT_POS[:, 1],
            c=w, cmap=cmap, vmin=0.0, vmax=1.0,
            s=100 + w * 500,
            edgecolors="white", linewidths=1.2, zorder=2,
        )
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)

    # скрываем лишние axes
    for bidx in range(num_blocks, len(axes)):
        axes[bidx].axis("off")

    fig.suptitle(f"{source_name} — по блокам", fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_joint_importance(
    model: STGCN_Simple,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    writer: SummaryWriter,
):
    """
    Главная функция — вызывается в конце обучения.
    Сохраняет PNG и логирует в TensorBoard.
    """
    vis_dir = run_dir / "joint_importance"
    os.makedirs(vis_dir, exist_ok=True)

    # ── 1. Edge Importance ────────────────────────────────────────────────
    ei_weights = extract_edge_importance_per_joint(model)
    if ei_weights is not None:
        # Среднее по всем блокам
        ei_mean = ei_weights.mean(axis=0)  # (V,)
        fig_ei_mean = plot_skeleton_heatmap(
            ei_mean,
            title="Edge Importance (среднее по блокам)",
            save_path=str(vis_dir / "edge_importance_mean.png"),
        )
        writer.add_figure("JointImportance/EdgeImportance_mean", fig_ei_mean)
        plt.close(fig_ei_mean)

        # По каждому блоку
        fig_ei_blocks = plot_importance_per_block(
            ei_weights,
            source_name="Edge Importance",
            save_path=str(vis_dir / "edge_importance_blocks.png"),
        )
        writer.add_figure("JointImportance/EdgeImportance_blocks", fig_ei_blocks)
        plt.close(fig_ei_blocks)

        # Текстовый вывод топ-5 суставов
        print("\n=== Edge Importance — топ суставов (среднее) ===")
        top_idx = np.argsort(ei_mean)[::-1]
        for rank, vi in enumerate(top_idx, 1):
            print(f"  {rank:2d}. {JOINT_NAMES[vi]:<16} {ei_mean[vi]:.4f}")

    # ── 2. Spatial Attention ──────────────────────────────────────────────
    has_att = any(isinstance(b.attention, SpatialTemporalAttention)
                  for b in model.blocks)
    if has_att:
        print("\n=== Сбор Spatial Attention весов (val set) ===")
        att_weights = collect_spatial_attention_weights(
            model, val_loader, device, num_batches=20
        )

        att_mean = att_weights.mean(axis=0)  # (V,)
        fig_att_mean = plot_skeleton_heatmap(
            att_mean,
            title="Spatial Attention (среднее по блокам)",
            save_path=str(vis_dir / "spatial_attention_mean.png"),
        )
        writer.add_figure("JointImportance/SpatialAttention_mean", fig_att_mean)
        plt.close(fig_att_mean)

        fig_att_blocks = plot_importance_per_block(
            att_weights,
            source_name="Spatial Attention",
            save_path=str(vis_dir / "spatial_attention_blocks.png"),
        )
        writer.add_figure("JointImportance/SpatialAttention_blocks", fig_att_blocks)
        plt.close(fig_att_blocks)

        print("\n=== Spatial Attention — топ суставов (среднее) ===")
        top_idx = np.argsort(att_mean)[::-1]
        for rank, vi in enumerate(top_idx, 1):
            print(f"  {rank:2d}. {JOINT_NAMES[vi]:<16} {att_mean[vi]:.4f}")

    print(f"\nКарты важности сохранены: {vis_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config_stgcn.yaml")
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()

    cfg      = load_config(args.config)
    exp_name = args.exp_name or cfg.get("experiment_name", "stgcn_biisc")

    ts       = time.strftime("%Y%m%d_%H%M%S")
    run_dir  = Path(cfg["paths"].get("results_dir", "runs")) / f"{exp_name}_{ts}"
    ckpt_dir = run_dir / "checkpoints"
    tb_dir   = run_dir / "tensorboard"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)
    shutil.copy2(args.config, run_dir / "config_used.yaml")

    device = setup_device(cfg["gpu"].get("gpu_id", 0))

    # ── Данные ────────────────────────────────────────────────────────────
    print("=== Загрузка .npz ===")
    train_items, val_items, test_items = load_items(cfg["paths"]["keypoints_dir"], cfg)

    max_frames  = cfg["data"]["max_frames"]
    in_channels = cfg["data"]["in_channels"]
    aug_cfg     = cfg.get("augmentation", {})

    # ConcatDataset для физического увеличения датасета (опционально)
    copies  = cfg["data"].get("augment_copies", 0)
    base_ds = SkeletonDataset(train_items, max_frames=max_frames,
                               in_channels=in_channels, augment=False)
    aug_copies = [
        SkeletonDataset(train_items, max_frames=max_frames,
                        in_channels=in_channels, augment=True, aug_cfg=aug_cfg)
        for _ in range(copies)
    ]
    train_ds = ConcatDataset([base_ds] + aug_copies) if copies > 0 else \
               SkeletonDataset(train_items, max_frames=max_frames,
                               in_channels=in_channels, augment=True, aug_cfg=aug_cfg)

    val_ds = SkeletonDataset(val_items, max_frames=max_frames,
                              in_channels=in_channels, augment=False)
    test_ds = SkeletonDataset(test_items, max_frames=max_frames,
                               in_channels=in_channels, augment=False)

    num_workers = cfg["gpu"].get("num_workers", 0)
    pin_memory  = cfg["gpu"].get("pin_memory", True)
    batch_size  = cfg["training"]["batch_size"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    print(f"Train dataset size: {len(train_ds)}"
          + (f" ({1+copies}× копий)" if copies > 0 else ""))

    # ── Модель ────────────────────────────────────────────────────────────
    print("=== Инициализация модели ===")
    m_cfg   = cfg["model"]
    hidden  = m_cfg.get("hidden_channels", [64, 128, 256])
    dropout = m_cfg.get("dropout", 0.5)
    tcn_k   = m_cfg.get("tcn_kernel_size", 9)
    stride_at = m_cfg.get("stride_at", [])
    use_edge  = m_cfg.get("use_edge_importance", True)
    use_att   = m_cfg.get("use_st_attention", True)
    att_from  = m_cfg.get("att_from_block", 1)
    att_red   = m_cfg.get("att_reduction", 16)

    model = STGCN_Simple(
        num_class=len(CLASS_NAMES),
        num_joints=m_cfg.get("num_joints", 17),
        in_channels=in_channels,
        hidden_channels=hidden,
        dropout=dropout,
        tcn_kernel_size=tcn_k,
        stride_at=stride_at,
        use_edge_importance=use_edge,
        use_st_attention=use_att,
        att_from_block=att_from,
        att_reduction=att_red,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Блоков ST-GCN     : {len(hidden)}")
    print(f"Каналы            : {in_channels} → {' → '.join(map(str, hidden))}")
    print(f"Edge importance   : {'✓' if use_edge else '✗'}")
    print(f"ST-Attention      : {'✓ (с блока ' + str(att_from) + ')' if use_att else '✗'}")
    print(f"Total params      : {total_p:,} | Trainable: {train_p:,}")

    # ── Оптимизатор / scheduler ───────────────────────────────────────────
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(),
                            lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"].get("weight_decay", 0.0))

    sched_type = cfg["training"].get("scheduler", "step")
    if sched_type == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["training"].get("scheduler_step", 20),
            gamma=cfg["training"].get("scheduler_gamma", 0.5),
        )
    elif sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"].get("num_epochs", 60)
        )
    else:
        scheduler = None

    grad_clip      = cfg["training"].get("grad_clip", 0.0)
    num_epochs     = cfg["training"]["num_epochs"]
    early_patience = cfg["training"].get("early_stopping_patience", -1)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("config", cfg_to_markdown(cfg), global_step=0)
    try:
        x0, _ = next(iter(train_loader))
        writer.add_graph(model, x0.to(device))
    except Exception as e:
        print(f"[WARN] Не удалось записать graph: {e}")

    # ── Обучение ──────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_epoch  = -1
    no_improve  = 0
    log_hist    = cfg["tensorboard"].get("log_histograms", False)
    cm_every    = cfg["tensorboard"].get("cm_every_n", 10)

    print("=== Обучение ===")
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_m = run_one_epoch(model, train_loader, device, criterion,
                                optimizer, mode="train", grad_clip=grad_clip)
        val_m   = run_one_epoch(model, val_loader,   device, criterion,
                                mode="val")

        if scheduler is not None:
            scheduler.step()

        # Scalars
        writer.add_scalar("Loss/train",     train_m["loss"],       epoch)
        writer.add_scalar("Loss/val",       val_m["loss"],         epoch)
        writer.add_scalar("Acc/train",      train_m["acc"],        epoch)
        writer.add_scalar("Acc/val",        val_m["acc"],          epoch)
        writer.add_scalar("F1_macro/train", train_m["f1_macro"],   epoch)
        writer.add_scalar("F1_macro/val",   val_m["f1_macro"],     epoch)
        if scheduler:
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        for ci, f1c in enumerate(val_m["per_class_f1"]):
            writer.add_scalar(f"F1_val/{CLASS_NAMES[ci]}", float(f1c), epoch)

        if cm_every > 0 and epoch % cm_every == 0:
            fig = plot_confusion_matrix_figure(val_m["cm"], CLASS_NAMES)
            writer.add_figure("ConfusionMatrix/val", fig, global_step=epoch)
            plt.close(fig)

        if log_hist:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

        # Checkpoint
        cur_f1 = val_m["f1_macro"]
        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f" ✓ Новый лучший F1_macro={best_val_f1:.4f} (эпоха {epoch})")
        else:
            no_improve += 1

        if early_patience > 0 and no_improve >= early_patience:
            print(f"\n[EARLY STOP] нет улучшения {early_patience} эпох подряд.")
            break

        writer.flush()

    # ── Тест ──────────────────────────────────────────────────────────────
    print("\n=== Тестирование best_model ===")
    if (ckpt_dir / "best_model.pt").exists():
        model.load_state_dict(
            torch.load(ckpt_dir / "best_model.pt", map_location=device)
        )
    model.to(device)

    test_m = run_one_epoch(model, test_loader, device, criterion, mode="test")

    print(f"\nTest Accuracy : {test_m['acc']:.4f}")
    print(f"Test F1-macro : {test_m['f1_macro']:.4f}")
    print("Per-class F1  :")
    for ci, f1c in enumerate(test_m["per_class_f1"]):
        print(f"  {CLASS_NAMES[ci]}: {float(f1c):.4f}")

    report = classification_report(
        test_m["y_true"], test_m["y_pred"],
        target_names=CLASS_NAMES, zero_division=0,
    )
    print("\nClassification report:\n", report)
    (run_dir / "test_report.txt").write_text(report, encoding="utf-8")

    fig_test = plot_confusion_matrix_figure(test_m["cm"], CLASS_NAMES)
    writer.add_figure("ConfusionMatrix/test", fig_test, global_step=0)
    plt.close(fig_test)

        # ── Визуализация важности суставов ───────────────────────────────────
    print("\n=== Визуализация важности суставов ===")
    visualize_joint_importance(model, val_loader, device, run_dir, writer)

    writer.close()


    writer.close()
    print(f"\nГотово. Логи и чекпоинты: {run_dir}")


if __name__ == "__main__":
    main()
