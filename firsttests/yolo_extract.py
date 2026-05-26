# yolo_extract.py
"""
Извлекает COCO-17 скелетные ключевые точки из видео с помощью YOLO Pose.

Для каждого видео сохраняется .npz файл:
    keypoints  : np.ndarray (T, 17, 3)  — [кадр, точка, (x_norm, y_norm, conf)]
    Нулевой вектор записывается для кадров без обнаруженного человека.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Поддерживаемые модели (имя → файл весов)
# ------------------------------------------------------------------
SUPPORTED_MODELS: Dict[str, str] = {
    "yolov8n-pose": "yolov8n-pose.pt",
    "yolov8s-pose": "yolov8s-pose.pt",
    "yolov8m-pose": "yolov8m-pose.pt",
    "yolov8l-pose": "yolov8l-pose.pt",
    "yolov8x-pose": "yolov8x-pose.pt",
    "yolo11n-pose": "yolo11n-pose.pt",
    "yolo11s-pose": "yolo11s-pose.pt",
    "yolo11m-pose": "yolo11m-pose.pt",
    "yolo11l-pose": "yolo11l-pose.pt",
    "yolo11x-pose": "yolo11x-pose.pt",
}

COCO_KEYPOINTS: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17


# ------------------------------------------------------------------
class YOLOPoseExtractor:
    """
    Обёртка над YOLO Pose для покадрового извлечения скелетов из видео.

    Параметры
    ---------
    model_name      : ключ из SUPPORTED_MODELS
    conf_threshold  : минимальная уверенность детекции человека
    device          : 'cpu', 'cuda', 'mps' и т.д.
    normalize       : если True — сохраняет координаты, нормализованные к [0, 1]
                      (рекомендуется: инвариантно к разрешению)
    """

    def __init__(
        self,
        model_name: str = "yolov8n-pose",
        conf_threshold: float = 0.3,
        device: str = "cuda",
        normalize: bool = True,
    ):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Неизвестная модель '{model_name}'. "
                f"Доступные: {list(SUPPORTED_MODELS)}"
            )
        self.model_name      = model_name
        self.conf_threshold  = conf_threshold
        self.device          = device
        self.normalize       = normalize

        logger.info(f"Загрузка модели: {SUPPORTED_MODELS[model_name]}  device={device}")
        self.model = YOLO(SUPPORTED_MODELS[model_name])
        logger.info("Модель загружена.")

    # ------------------------------------------------------------------
    def extract_video(self, video_path: str) -> np.ndarray:
        """
        Извлекает ключевые точки из каждого кадра видео.

        Возвращает
        ----------
        np.ndarray  shape (T, 17, 3)
            Оси: [frame_index, keypoint_index, (x, y, confidence)]
            Координаты нормализованы (0..1) при normalize=True.
            Нулевой вектор — кадр без обнаруженного человека.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {video_path}")

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1

        frames_kpts: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(
                frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )

            kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

            for result in results:
                if result.keypoints is None or result.keypoints.data.shape[0] == 0:
                    continue

                # result.keypoints.data : Tensor (N, 17, 3) — (x_px, y_px, conf)
                data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)

                # Выбираем человека с наибольшей уверенностью bounding box
                if result.boxes is not None and len(result.boxes) > 0:
                    best = int(np.argmax(result.boxes.conf.cpu().numpy()))
                else:
                    best = 0

                kpts = data[best].copy()  # (17, 3)

                if self.normalize:
                    kpts[:, 0] /= w   # x → [0, 1]
                    kpts[:, 1] /= h   # y → [0, 1]
                break

            frames_kpts.append(kpts)

        cap.release()

        if not frames_kpts:
            logger.warning(f"Нет кадров: {video_path}")
            return np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)

        return np.stack(frames_kpts, axis=0)  # (T, 17, 3)

    # ------------------------------------------------------------------
    def extract_dataset(
        self,
        df: pd.DataFrame,
        output_dir: str,
        skip_existing: bool = True,
    ) -> Dict[str, str]:
        """
        Прогоняет экстракцию по всем видео в DataFrame.

        Сохраняет:
            {output_dir}/{video_stem}.npz  →  ключ 'keypoints': (T, 17, 3)
            {output_dir}/index.json        →  маппинг filename → npz_path

        Возвращает
        ----------
        dict  filename → путь к .npz
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        index_map: Dict[str, str] = {}
        failed: List[str] = []

        for _, row in tqdm(df.iterrows(), total=len(df),
                           desc=f"Extraction [{self.model_name}]"):
            stem     = Path(row["filepath"]).stem
            npz_path = out / f"{stem}.npz"

            if skip_existing and npz_path.exists():
                index_map[row["filename"]] = str(npz_path)
                continue

            try:
                kpts = self.extract_video(row["filepath"])
                np.savez_compressed(
                    str(npz_path),
                    keypoints  = kpts,
                    filename   = np.array(row["filename"]),
                    action     = np.array(row["action"]),
                    subject    = np.array(row["subject"]),
                    split      = np.array(row["split"]),
                )
                index_map[row["filename"]] = str(npz_path)
            except Exception as exc:
                logger.error(f"Ошибка при обработке {row['filepath']}: {exc}")
                failed.append(row["filepath"])

        if failed:
            logger.warning(f"{len(failed)} видео не обработано.")

        index_path = out / "index.json"
        index_path.write_text(json.dumps(index_map, indent=2, ensure_ascii=False))
        logger.info(f"Экстракция завершена. Индекс: {index_path}")

        return index_map
