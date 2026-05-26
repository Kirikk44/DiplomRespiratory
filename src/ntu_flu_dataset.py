"""
ntu_flu_dataset.py

Pipeline в __getitem__:
  .skeleton (T,25,3)
      ↓  [3D aug: CCTV-углы, вращение Y, flip]       ← управляется конфигом
      ↓  NTU-25 → COCO-17 + project_2d(view)
      ↓  нормализация (hip-center + shoulder-scale)
      ↓  resize → num_frames
      ↓  [2D aug: mirror, rotate2d, temporal-crop, noise]
      ↓  (2, T, 17) tensor
"""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pytorch_lightning as pl

from augmentation import (
    VirtualCamera,
    apply_transform_to_skeleton,
    PRESET_CAMERAS,
    read_skeleton_file,
)

# ── NTU-25 → COCO-17 таблица ─────────────────────────────────────────────
# Для каждого COCO-индекса указываем NTU-индекс
NTU_TO_COCO_IDX = [
    3,   # 0  nose          ← NTU 3  head
    3,   # 1  left_eye      ← NTU 3  head (approx)
    3,   # 2  right_eye     ← NTU 3  head (approx)
    3,   # 3  left_ear      ← NTU 3  head (approx)
    3,   # 4  right_ear     ← NTU 3  head (approx)
    4,   # 5  left_shoulder ← NTU 4
    8,   # 6  right_shoulder← NTU 8
    5,   # 7  left_elbow    ← NTU 5
    9,   # 8  right_elbow   ← NTU 9
    6,   # 9  left_wrist    ← NTU 6
    10,  # 10 right_wrist   ← NTU 10
    12,  # 11 left_hip      ← NTU 12
    16,  # 12 right_hip     ← NTU 16
    13,  # 13 left_knee     ← NTU 13
    17,  # 14 right_knee    ← NTU 17
    14,  # 15 left_ankle    ← NTU 14
    18,  # 16 right_ankle   ← NTU 18
]

NTU_FLU_CLASSES = [
    "blow_nose", "brush_teeth", "clapping",  "cough",
    "drink",     "hand_wave",   "nausea",    "nod_head",
    "phone_call","rub_hands",   "sniff",     "wipe_face", "yawn",
]


# ─────────────────────────────────────────────────────────────────────────
# Augmentor
# ─────────────────────────────────────────────────────────────────────────

class SkeletonAugmentor:
    """
    Управляется через dict cfg.augmentation.
    Поддерживает 3D и 2D аугментации независимо.
    """

    def __init__(self, aug: dict):
        self.enabled = aug.get("enabled", True)

        # 3D ─────────────────────────────────────────
        self.aug3d_enabled   = aug.get("aug3d_enabled", True)
        # список ключей из PRESET_CAMERAS для случайного выбора
        self.cctv_presets_3d = [
            PRESET_CAMERAS[k]
            for k in aug.get("cctv_presets_3d", [])
            if k in PRESET_CAMERAS
        ]
        self.cctv_prob       = aug.get("cctv_prob", 0.5)

        # случайный поворот вокруг Y (горизонт.) прямо в 3D
        self.rot3d_y_prob    = aug.get("rot3d_y_prob",  0.5)
        self.rot3d_y_range   = aug.get("rot3d_y_range", 30.0)   # градусы

        # 2D ─────────────────────────────────────────
        self.aug2d_enabled   = aug.get("aug2d_enabled", True)
        self.mirror_prob     = aug.get("mirror_prob",   0.5)
        self.rotate2d_prob   = aug.get("rotate2d_prob", 0.5)
        self.rotate2d_range  = aug.get("rotate2d_range",15.0)
        self.noise_std       = aug.get("noise_std",     0.02)
        self.time_crop_prob  = aug.get("time_crop_prob",0.4)
        self.time_crop_frac  = aug.get("time_crop_frac",0.15)

    # ── 3D ───────────────────────────────────────────────────────────────
    def apply_3d(self, seq3d: np.ndarray) -> np.ndarray:
        """seq3d: (T, 25, 3)"""
        if not (self.enabled and self.aug3d_enabled):
            return seq3d

        # случайный CCTV-угол
        if self.cctv_presets_3d and np.random.rand() < self.cctv_prob:
            cam = self.cctv_presets_3d[np.random.randint(len(self.cctv_presets_3d))]
            R   = cam.get_transform_matrix()
            seq3d = apply_transform_to_skeleton(seq3d, R, center_joint=0)

        # случайный поворот вокруг Y
        if np.random.rand() < self.rot3d_y_prob:
            angle = np.random.uniform(-self.rot3d_y_range, self.rot3d_y_range)
            cam   = VirtualCamera("rand_y", rotation_y=angle)
            R     = cam.get_transform_matrix()
            seq3d = apply_transform_to_skeleton(seq3d, R, center_joint=0)

        return seq3d

    # ── 2D ───────────────────────────────────────────────────────────────
    def apply_2d(self, seq: np.ndarray) -> np.ndarray:
        """seq: (T, 17, 2)"""
        if not (self.enabled and self.aug2d_enabled):
            return seq

        T = seq.shape[0]

        # temporal crop
        if np.random.rand() < self.time_crop_prob:
            crop  = max(1, int(T * self.time_crop_frac))
            start = np.random.randint(0, crop)
            end   = T - np.random.randint(0, crop)
            end   = max(end, start + 2)
            idx   = np.linspace(start, end - 1, T)
            idx   = np.clip(idx, 0, T - 1).astype(int)
            seq   = seq[idx]

        # зеркало
        if np.random.rand() < self.mirror_prob:
            seq = seq.copy()
            seq[:, :, 0] = -seq[:, :, 0]

        # поворот в плоскости XY
        if np.random.rand() < self.rotate2d_prob:
            a   = np.random.uniform(-self.rotate2d_range, self.rotate2d_range)
            a  *= np.pi / 180
            c, s = np.cos(a), np.sin(a)
            R    = np.array([[c, -s], [s, c]], dtype=np.float32)
            seq  = seq @ R.T

        # гауссовый шум
        seq = seq + np.random.normal(0, self.noise_std, seq.shape).astype(np.float32)

        return seq.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────

class NTUFluDataset(Dataset):
    """
    Читает .skeleton файлы, применяет 3D-аугментации → конвертирует
    в COCO-17 2D → нормализация → 2D-аугментации.

    projection_view: 'front' | 'side' | 'top'
      front : (x, y)   — стандартный фронтальный вид
      side  : (z, y)   — боковой вид (полезно для бенчмарка)
      top   : (x, z)   — вид сверху (CCTV overhead)
    """

    def __init__(
        self,
        skeleton_root: str,
        classes: list,
        num_frames:       int   = 64,
        conf_thresh:      float = 0.15,
        augmentor:        Optional[SkeletonAugmentor] = None,
        projection_view:  str   = "front",
        # фиксированный CCTV-пресет для бенчмарка (один пресет на весь датасет)
        benchmark_camera: Optional[VirtualCamera] = None,
    ):
        self.num_frames       = num_frames
        self.conf_thresh      = conf_thresh
        self.augmentor        = augmentor
        self.projection_view  = projection_view
        self.benchmark_camera = benchmark_camera
        self.cls2idx          = {c: i for i, c in enumerate(classes)}

        root = Path(skeleton_root)
        self.samples = []
        for cls in classes:
            for p in sorted((root / cls).glob("*.skeleton")):
                self.samples.append((p, self.cls2idx[cls]))

        mode = []
        if augmentor and augmentor.enabled:
            if augmentor.aug3d_enabled: mode.append("3D-aug")
            if augmentor.aug2d_enabled: mode.append("2D-aug")
        else:
            mode.append("no-aug")
        if benchmark_camera:
            mode.append(f"bench:{benchmark_camera.name}")

        print(f"NTUFluDataset [{'+'.join(mode)}] "
              f"view={projection_view}: "
              f"{len(self.samples)} сэмплов | {len(classes)} классов")

    def __len__(self): return len(self.samples)

    # ── NTU 25pt → 3D COCO 17pt ──────────────────────────────────────────
    @staticmethod
    def ntu25_to_coco17_3d(seq3d: np.ndarray) -> np.ndarray:
        """(T,25,3) → (T,17,3)"""
        T = seq3d.shape[0]
        out = np.zeros((T, 17, 3), dtype=np.float32)
        for coco_i, ntu_i in enumerate(NTU_TO_COCO_IDX):
            out[:, coco_i, :] = seq3d[:, ntu_i, :]
        return out

    # ── проекция 3D → 2D ──────────────────────────────────────────────────
    @staticmethod
    def project_2d(coco3d: np.ndarray, view: str) -> np.ndarray:
        """
        (T,17,3) → (T,17,2)
        front : x,y  |  side : z,y  |  top : x,z
        """
        if view == "front":
            return coco3d[:, :, [0, 1]]
        elif view == "side":
            return coco3d[:, :, [2, 1]]
        elif view == "top":
            return coco3d[:, :, [0, 2]]
        else:
            raise ValueError(f"Неизвестный projection_view: {view}")

    # ── нормализация ──────────────────────────────────────────────────────
    @staticmethod
    def normalize(xy: np.ndarray) -> np.ndarray:
        """hip-center + shoulder-scale"""
        hip   = (xy[:, 11:12] + xy[:, 12:13]) / 2.0
        xy    = xy - hip
        scale = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=-1).mean() + 1e-6
        return (xy / scale).astype(np.float32)

    # ── resize → num_frames ───────────────────────────────────────────────
    def resize(self, seq: np.ndarray) -> np.ndarray:
        T   = seq.shape[0]
        idx = np.linspace(0, T - 1, self.num_frames)
        return seq[np.clip(idx, 0, T - 1).astype(int)]

    # ── __getitem__ ───────────────────────────────────────────────────────
    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 1. Читаем .skeleton → (T, 25, 3)
        seq3d = read_skeleton_file(str(path)).astype(np.float32)

        # 2. Бенчмарк: фиксированный CCTV-угол (применяется ДО аугментаций)
        if self.benchmark_camera is not None:
            R     = self.benchmark_camera.get_transform_matrix()
            seq3d = apply_transform_to_skeleton(seq3d, R, center_joint=0)

        # 3. 3D-аугментации (случайные CCTV-углы, поворот вокруг Y)
        if self.augmentor:
            seq3d = self.augmentor.apply_3d(seq3d)

        # 4. NTU-25 → COCO-17 (3D)
        coco3d = self.ntu25_to_coco17_3d(seq3d)   # (T,17,3)

        # 5. Проекция в нужную плоскость → 2D
        xy = self.project_2d(coco3d, self.projection_view)   # (T,17,2)

        # 6. Нормализация
        xy = self.normalize(xy)

        # 7. Resize → num_frames
        xy = self.resize(xy)

        # 8. 2D-аугментации
        if self.augmentor:
            xy = self.augmentor.apply_2d(xy)

        # 9. (T,17,2) → (2, T, 17)
        x = torch.from_numpy(xy.transpose(2, 0, 1).copy())
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────

class NTUFluDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        cfg     = self.cfg
        classes = list(cfg.data.classes)
        aug_cfg = dict(cfg.augmentation)

        aug = SkeletonAugmentor(aug_cfg)
        view = cfg.data.get("projection_view", "front")

        # ── Полный датасет без аугментации (для разбивки) ─────────────────
        full_ds = NTUFluDataset(
            cfg.data.skeleton_root, classes,
            cfg.data.num_frames, cfg.data.conf_thresh,
            augmentor=None, projection_view=view,
        )
        n_val   = int(len(full_ds) * cfg.train.val_split)
        n_train = len(full_ds) - n_val
        gen     = torch.Generator().manual_seed(cfg.train.seed)
        train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=gen)

        # ── Train: с аугментацией, те же индексы ───────────────────────────
        aug_ds = NTUFluDataset(
            cfg.data.skeleton_root, classes,
            cfg.data.num_frames, cfg.data.conf_thresh,
            augmentor=aug, projection_view=view,
        )
        self.train_ds = Subset(aug_ds, train_sub.indices)
        self.val_ds   = val_sub

        # ── Benchmark датасеты (один на каждый CCTV-пресет) ───────────────
        self.benchmark_datasets = {}
        if cfg.get("cctv_benchmark", {}).get("enabled", False):
            for key in cfg.cctv_benchmark.presets:
                if key not in PRESET_CAMERAS:
                    print(f"⚠ Пресет {key!r} не найден в PRESET_CAMERAS, пропущен")
                    continue
                bview = cfg.cctv_benchmark.get("projection_view", view)
                self.benchmark_datasets[key] = NTUFluDataset(
                    cfg.data.skeleton_root, classes,
                    cfg.data.num_frames, cfg.data.conf_thresh,
                    augmentor=None,
                    projection_view=bview,
                    benchmark_camera=PRESET_CAMERAS[key],
                )

        print(f"\nTrain: {len(self.train_ds)} | Val: {len(self.val_ds)}")
        if self.benchmark_datasets:
            print(f"Benchmark датасеты: {list(self.benchmark_datasets.keys())}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )

    def benchmark_dataloader(self, preset_key: str):
        """Отдельный DataLoader для одного CCTV-пресета."""
        ds = self.benchmark_datasets[preset_key]
        return DataLoader(
            ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
        )