# biisk_dataset.py
"""
DataModule для датасета BIISK.
Формат: .npz с ключами:
  skeleton : (T, 17, 3)  — x, y, confidence (COCO-17)
  class_name: str
  video     : str

Pipeline в __getitem__:
  (T, 17, 3)
    ↓ берём только (x, y) → (T, 17, 2)
    ↓ normalize (hip-center + shoulder-scale)
    ↓ resize → num_frames
    ↓ [2D aug: mirror, rotate, temporal-crop, noise]
    ↓ transpose → (2, T, 17) tensor
"""

from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pytorch_lightning as pl
from omegaconf import DictConfig


# ─────────────────────────────────────────────────────────────────────────
# 2D Augmentor (не зависит от NTU, только 2D операции)
# ─────────────────────────────────────────────────────────────────────────
class BiiskAugmentor:
    def __init__(self, aug: dict):
        self.enabled        = aug.get("enabled",         True)
        self.mirror_prob    = aug.get("mirror_prob",     0.5)
        self.rotate2d_prob  = aug.get("rotate2d_prob",   0.4)
        self.rotate2d_range = aug.get("rotate2d_range",  15.0)
        self.noise_std      = aug.get("noise_std",       0.02)
        self.time_crop_prob = aug.get("time_crop_prob",  0.4)
        self.time_crop_frac = aug.get("time_crop_frac",  0.15)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        """seq: (T, 17, 3) -> (T, 17, 3), каналы [x, y, conf]"""
        if not self.enabled:
            return seq.astype(np.float32)

        T = seq.shape[0]

        xy   = seq[:, :, :2].copy()   # (T, 17, 2)
        conf = seq[:, :, 2:3].copy()  # (T, 17, 1)

        # temporal crop
        if np.random.rand() < self.time_crop_prob:
            crop  = max(1, int(T * self.time_crop_frac))
            start = np.random.randint(0, crop)
            end   = max(T - np.random.randint(0, crop), start + 2)
            idx   = np.clip(np.linspace(start, end - 1, T), 0, T - 1).astype(int)
            xy   = xy[idx]
            conf = conf[idx]

        # mirror по X
        if np.random.rand() < self.mirror_prob:
            xy[:, :, 0] *= -1

        # rotate только xy
        if np.random.rand() < self.rotate2d_prob:
            a = np.random.uniform(-self.rotate2d_range, self.rotate2d_range) * np.pi / 180.0
            c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            xy = xy @ R.T

        # noise только в xy
        xy = xy + np.random.normal(0, self.noise_std, xy.shape).astype(np.float32)

        return np.concatenate([xy, conf], axis=2).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────
class BiiskDataset(Dataset):
    """
    Читает .npz файлы из папок по классам:
      skeleton_root/
        cough/*.npz
        sneeze/*.npz
        ...

    Не зависит от NTU, не делает NTU→COCO конвертацию.
    """

    def __init__(
        self,
        skeleton_root: str,
        classes:       List[str],
        num_frames:    int                     = 64,
        augmentor:     Optional[BiiskAugmentor] = None,
        conf_thresh:   float                   = 0.0,
        cache:         bool                    = False,   # загрузить всё в RAM (1920 файлов ≈ 200MB)
    ):
        self.num_frames  = num_frames
        self.augmentor   = augmentor
        self.conf_thresh = conf_thresh
        self.cls2idx     = {c: i for i, c in enumerate(classes)}

        root = Path(skeleton_root)
        self.samples: List[tuple] = []
        for cls in classes:
            folder = root / cls
            if not folder.exists():
                print(f"⚠ Папка не найдена: {folder}")
                continue
            for p in sorted(folder.glob("*.npz")):
                self.samples.append((p, self.cls2idx[cls]))

        if not self.samples:
            raise FileNotFoundError(
                f"Не найдено .npz файлов в {root} для классов {classes}\n"
                f"Абсолютный путь: {root.resolve()}"
            )

        # опциональный RAM-кэш
        self._cache: dict = {}
        self._do_cache = cache
        if cache:
            print(f"Предзагрузка {len(self.samples)} файлов в RAM...")
            for i, (p, _) in enumerate(self.samples):
                self._cache[i] = self._load_npz(p)
            print("✅ Кэш заполнен")

        print(
            f"BiiskDataset: {len(self.samples)} сэмплов | "
            f"{len(classes)} классов | cache={cache}"
        )

    # ── Загрузка одного .npz ──────────────────────────────────────────────
    @staticmethod
    def _load_npz(path: Path) -> np.ndarray:
        data = np.load(path)
        return data["skeleton"].astype(np.float32)    # (T, 17, 3) — x, y, conf

    # ── Нормализация ──────────────────────────────────────────────────────
    @staticmethod
    def _normalize(seq: np.ndarray) -> np.ndarray:
        """seq: (T, 17, 3) → (T, 17, 3). Нормализуем только x,y; conf не трогаем."""
        xy   = seq[:, :, :2]                              # (T, 17, 2)
        conf = seq[:, :, 2:3]                             # (T, 17, 1)
        hip  = (xy[:, 11:12, :] + xy[:, 12:13, :]) / 2.0
        xy   = xy - hip
        scale = np.linalg.norm(xy[:, 5, :] - xy[:, 6, :], axis=-1).mean() + 1e-6
        xy   = (xy / scale).astype(np.float32)
        return np.concatenate([xy, conf], axis=2)          # (T, 17, 3)

    # ── Resize → num_frames ───────────────────────────────────────────────
    def _resize(self, seq: np.ndarray) -> np.ndarray:
        """(T, 17, 2) → (num_frames, 17, 2)"""
        T   = seq.shape[0]
        idx = np.linspace(0, T - 1, self.num_frames)
        return seq[np.clip(idx, 0, T - 1).astype(int)]

    # ── __len__ / __getitem__ ─────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 1. Загрузка
        if self._do_cache and idx in self._cache:
            seq = self._cache[idx].copy()
        else:
            seq = self._load_npz(path)

        # 2. Нормализация
        seq = self._normalize(seq)

        # 3. Resize → num_frames
        seq = self._resize(seq)

        # 4. Аугментация (BiiskAugmentor должен уметь работать с (T,17,3))
        if self.augmentor is not None:
            seq = self.augmentor(seq)

        # 5. Маскировка плохих суставов
        if self.conf_thresh > 0.0:
            mask = seq[:, :, 2] >= self.conf_thresh   # (T, 17)
            seq[:, :, :2] *= mask[:, :, None]

        # 6. (T, 17, 3) → (3, T, 17)
        x = torch.from_numpy(seq.transpose(2, 0, 1).copy())   # (3, T, 17)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────
class BiiskDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        cfg     = self.cfg
        classes = list(cfg.data.classes)
        aug_cfg = dict(cfg.augmentation)

        aug = BiiskAugmentor(aug_cfg)

        # Полный датасет без аугментации для разбивки train/val
        full_ds = BiiskDataset(
            skeleton_root = cfg.data.skeleton_root,
            classes       = classes,
            num_frames    = cfg.data.num_frames,
            augmentor     = None,
            conf_thresh   = cfg.data.get("conf_thresh", 0.0),
            cache         = cfg.data.get("cache", False),
        )

        n_val   = int(len(full_ds) * cfg.train.val_split)
        n_train = len(full_ds) - n_val
        gen     = torch.Generator().manual_seed(cfg.train.seed)
        train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=gen)

        # Train: те же индексы, но с аугментацией
        aug_ds = BiiskDataset(
            skeleton_root = cfg.data.skeleton_root,
            classes       = classes,
            num_frames    = cfg.data.num_frames,
            augmentor     = aug,
            conf_thresh   = cfg.data.get("conf_thresh", 0.0),
            cache         = cfg.data.get("cache", False),
        )
        self.train_ds = Subset(aug_ds, train_sub.indices)
        self.val_ds   = val_sub
        self.classes  = classes

        print(f"Train: {len(self.train_ds)} | Val: {len(self.val_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.cfg.train.batch_size,
            shuffle     = True,
            num_workers = self.cfg.train.num_workers,
            pin_memory  = True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = self.cfg.train.batch_size,
            shuffle     = False,
            num_workers = self.cfg.train.num_workers,
            pin_memory  = True,
        )