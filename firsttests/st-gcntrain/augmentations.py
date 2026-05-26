# augmentations.py
"""
Аугментации скелетных кейпоинтов (T, 17, 3) для датасета BIISC/CCTV.

Все функции:
  - Принимают kpts: np.ndarray (T, 17, 3) — (x, y, conf), координаты [0,1]
  - Возвращают новый массив той же формы (оригинал не изменяется)

Группы аугментаций:
  [CCTV]     Имитация другого угла/позиции камеры
  [TEMPORAL] Изменение временной оси
  [SPATIAL]  Пространственные преобразования скелета
  [NOISE]    Шум детекции (имитация ошибок YOLO)
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

# ─── COCO-17: пары суставов лево↔право ───────────────────────────────────────
# Используются при горизонтальном флипе
LR_JOINT_PAIRS: List[Tuple[int, int]] = [
    (1,  2),   # left_eye      ↔ right_eye
    (3,  4),   # left_ear      ↔ right_ear
    (5,  6),   # left_shoulder ↔ right_shoulder
    (7,  8),   # left_elbow    ↔ right_elbow
    (9,  10),  # left_wrist    ↔ right_wrist
    (11, 12),  # left_hip      ↔ right_hip
    (13, 14),  # left_knee     ↔ right_knee
    (15, 16),  # left_ankle    ↔ right_ankle
]


# ─── Утилиты ──────────────────────────────────────────────────────────────────

def _skeleton_center(kpts: np.ndarray) -> Tuple[float, float]:
    """Центр масс видимых суставов (среднее по всем кадрам и суставам)."""
    visible = kpts[:, :, 2] > 0.1          # (T, V) bool
    if visible.sum() == 0:
        return 0.5, 0.5
    cx = float(kpts[:, :, 0][visible].mean())
    cy = float(kpts[:, :, 1][visible].mean())
    return cx, cy


# ═════════════════════════════════════════════════════════════════════════════
# [NOISE] Шум детекции
# ═════════════════════════════════════════════════════════════════════════════

def aug_gaussian_noise(kpts: np.ndarray, std: float = 0.02) -> np.ndarray:
    """
    [NOISE] Гауссов шум на (x, y) видимых суставов.
    Имитирует погрешность детекции YOLO.
    """
    out = kpts.copy()
    visible = (out[:, :, 2:3] > 0.1).astype(np.float32)
    noise = np.random.normal(0.0, std, out[:, :, :2].shape).astype(np.float32)
    out[:, :, :2] += noise * visible
    return out


def aug_joint_dropout(kpts: np.ndarray, prob: float = 0.10) -> np.ndarray:
    """
    [NOISE] Случайное обнуление суставов — одна маска на весь клип.
    Имитирует окклюзию или потерю детекции YOLO на протяжении всего действия.
    Prob = вероятность потери каждого из 17 суставов.
    """
    out = kpts.copy()
    mask = np.random.rand(17) < prob        # (V,) — одна маска на весь клип
    out[:, mask, :] = 0.0
    return out


def aug_conf_noise(kpts: np.ndarray, std: float = 0.08) -> np.ndarray:
    """
    [NOISE] Шум в канале confidence.
    Модель не должна слепо доверять уверенности YOLO.
    """
    out = kpts.copy()
    noise = np.random.normal(0.0, std, out[:, :, 2].shape).astype(np.float32)
    out[:, :, 2] = np.clip(out[:, :, 2] + noise, 0.0, 1.0)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# [TEMPORAL] Временная ось
# ═════════════════════════════════════════════════════════════════════════════

def aug_temporal_crop(kpts: np.ndarray, max_frames: int) -> np.ndarray:
    """
    [TEMPORAL] Случайный временной кроп вместо всегда с кадра 0.
    Модель учится распознавать действие с любой фазы.
    """
    T = kpts.shape[0]
    if T <= max_frames:
        return kpts
    start = np.random.randint(0, T - max_frames + 1)
    return kpts[start: start + max_frames].copy()


def aug_temporal_speed(
    kpts: np.ndarray,
    speed_range: Tuple[float, float] = (0.7, 1.3),
) -> np.ndarray:
    """
    [TEMPORAL] Случайное изменение скорости выполнения действия.
    speed > 1 → ускорение (меньше кадров).
    speed < 1 → замедление (больше кадров).
    Линейная интерполяция между кадрами.
    Имитирует людей с разной скоростью кашля/чихания.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# [SPATIAL] Пространственные преобразования
# ═════════════════════════════════════════════════════════════════════════════

def aug_horizontal_flip(kpts: np.ndarray) -> np.ndarray:
    """
    [SPATIAL] Горизонтальный флип: x → 1-x + swap левых и правых суставов.
    В датасете уже есть _HF версии, но эта аугментация применяется случайно
    в каждую эпоху, что увеличивает разнообразие.
    """
    out = kpts.copy()
    out[:, :, 0] = 1.0 - out[:, :, 0]
    for left, right in LR_JOINT_PAIRS:
        out[:, [left, right], :] = out[:, [right, left], :]
    return out


def aug_zoom(
    kpts: np.ndarray,
    zoom_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    [SPATIAL] Масштабирование скелета вокруг его центра.
    Имитирует разное расстояние субъекта от CCTV камеры.
    scale > 1 → человек ближе (скелет крупнее).
    scale < 1 → человек дальше (скелет мельче).
    """
    out = kpts.copy()
    scale = np.random.uniform(*zoom_range)
    cx, cy = _skeleton_center(out)
    out[:, :, 0] = cx + (out[:, :, 0] - cx) * scale
    out[:, :, 1] = cy + (out[:, :, 1] - cy) * scale
    return out


def aug_translate(kpts: np.ndarray, std: float = 0.05) -> np.ndarray:
    """
    [SPATIAL] Случайный сдвиг скелета по x и y.
    Имитирует разное положение субъекта в кадре CCTV
    (не всегда строго по центру).
    """
    out = kpts.copy()
    dx = np.random.normal(0.0, std)
    dy = np.random.normal(0.0, std)
    out[:, :, 0] += dx
    out[:, :, 1] += dy
    return out


def aug_rotate_2d(
    kpts: np.ndarray,
    angle_range: Tuple[float, float] = (-15.0, 15.0),
) -> np.ndarray:
    """
    [CCTV] Поворот скелета в плоскости изображения.
    Имитирует наклон/перекос CCTV камеры (камера установлена не строго горизонтально).
    """
    out = kpts.copy()
    angle = np.random.uniform(*angle_range)
    rad   = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    cx, cy = _skeleton_center(out)

    x = out[:, :, 0] - cx
    y = out[:, :, 1] - cy
    out[:, :, 0] = cx + x * cos_a - y * sin_a
    out[:, :, 1] = cy + x * sin_a + y * cos_a
    return out


def aug_perspective_x(
    kpts: np.ndarray,
    strength_range: Tuple[float, float] = (-0.15, 0.15),
) -> np.ndarray:
    """
    [CCTV] Имитация бокового угла камеры (поворот субъекта в 3D по оси Y).
    При повороте тело выглядит более «плоским» по горизонтали:
        x_new = cx + (x - cx) + strength * (y - cy)
    Позволяет генерировать промежуточные углы между FCE, LFT, RGT.
    Наиболее важная CCTV-аугментация для данного датасета.
    """
    out = kpts.copy()
    strength = np.random.uniform(*strength_range)
    cx, cy   = _skeleton_center(out)

    x = out[:, :, 0] - cx
    y = out[:, :, 1] - cy
    out[:, :, 0] = cx + x + strength * y
    return out


def aug_perspective_y(
    kpts: np.ndarray,
    strength_range: Tuple[float, float] = (-0.10, 0.10),
) -> np.ndarray:
    """
    [CCTV] Имитация высоты установки CCTV камеры.
    Камера высоко → верхние точки скелета сдвигаются к центру,
    нижние «растягиваются»:
        y_new = cy + (y - cy) + strength * (x - cx)
    Один угол CCTV снимает почти горизонтально, другой — сильно сверху вниз.
    """
    out = kpts.copy()
    strength = np.random.uniform(*strength_range)
    cx, cy   = _skeleton_center(out)

    x = out[:, :, 0] - cx
    y = out[:, :, 1] - cy
    out[:, :, 1] = cy + y + strength * x
    return out


def aug_scale_bones(
    kpts: np.ndarray,
    scale_range: Tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    """
    [SPATIAL] Независимое масштабирование по x и y.
    Имитирует людей разного телосложения (широкие плечи, высокий рост).
    В отличие от zoom, оси масштабируются независимо.
    """
    out = kpts.copy()
    sx  = np.random.uniform(*scale_range)
    sy  = np.random.uniform(*scale_range)
    cx, cy = _skeleton_center(out)

    out[:, :, 0] = cx + (out[:, :, 0] - cx) * sx
    out[:, :, 1] = cy + (out[:, :, 1] - cy) * sy
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Оркестратор — читает конфиг и применяет цепочку аугментаций
# ═════════════════════════════════════════════════════════════════════════════

class SkeletonAugmentor:
    """
    Применяет включённые аугментации к (T, 17, 3) кейпоинтам.
    Управляется секцией augmentation: в YAML конфиге.

    Порядок применения фиксирован и логически обоснован:
      temporal → cctv-angle → spatial → noise
    """

    def __init__(self, aug_cfg: dict):
        self.cfg = aug_cfg or {}

    def __call__(self, kpts: np.ndarray, max_frames: int) -> np.ndarray:

        # ── Temporal ──────────────────────────────────────────────────
        if self.cfg.get("temporal_crop", False):
            kpts = aug_temporal_crop(kpts, max_frames)

        if self.cfg.get("temporal_speed", False):
            speed_range = tuple(self.cfg.get("speed_range", [0.7, 1.3]))
            kpts = aug_temporal_speed(kpts, speed_range)

        # ── CCTV angle ────────────────────────────────────────────────
        if self.cfg.get("horizontal_flip", False):
            if np.random.rand() < self.cfg.get("flip_prob", 0.5):
                kpts = aug_horizontal_flip(kpts)

        if self.cfg.get("rotate_2d", False):
            angle_range = tuple(self.cfg.get("rotate_range", [-15, 15]))
            kpts = aug_rotate_2d(kpts, angle_range)

        if self.cfg.get("perspective_x", False):
            r = tuple(self.cfg.get("perspective_x_range", [-0.15, 0.15]))
            kpts = aug_perspective_x(kpts, r)

        if self.cfg.get("perspective_y", False):
            r = tuple(self.cfg.get("perspective_y_range", [-0.10, 0.10]))
            kpts = aug_perspective_y(kpts, r)

        # ── Spatial ───────────────────────────────────────────────────
        if self.cfg.get("zoom", False):
            zoom_range = tuple(self.cfg.get("zoom_range", [0.85, 1.15]))
            kpts = aug_zoom(kpts, zoom_range)

        if self.cfg.get("translate", False):
            std = self.cfg.get("translate_std", 0.05)
            kpts = aug_translate(kpts, std)

        if self.cfg.get("scale_bones", False):
            r = tuple(self.cfg.get("scale_bones_range", [0.85, 1.15]))
            kpts = aug_scale_bones(kpts, r)

        # ── Noise ─────────────────────────────────────────────────────
        if self.cfg.get("joint_dropout", False):
            prob = self.cfg.get("joint_dropout_prob", 0.10)
            kpts = aug_joint_dropout(kpts, prob)

        noise_std = self.cfg.get("noise_std", 0.0)
        if noise_std > 0:
            kpts = aug_gaussian_noise(kpts, noise_std)

        if self.cfg.get("conf_noise", False):
            std = self.cfg.get("conf_noise_std", 0.08)
            kpts = aug_conf_noise(kpts, std)

        return kpts
