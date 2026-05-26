# evaluate.py
"""
Загружает извлечённые .npz кейпоинты, обучает лёгкий классификатор
(Random Forest), оценивает качество на тест-выборке и сохраняет артефакты.

Схема признаков (per video):
    temporal stats (mean, std, min, max) по (x, y) → 4 × 17 × 2 = 136
    mean confidence per keypoint               →           17
    ──────────────────────────────────────────────────────────
    итого                                      →          153 признака
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

NUM_KPT = 17
TARGET_CLASSES = ("COUG", "SNEE")   # приоритетные классы для отчётности


# ------------------------------------------------------------------
def compute_video_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Агрегирует (T, 17, 3) → вектор признаков (153,).

    Предполагается, что координаты нормализованы (0..1).
    """
    T = keypoints.shape[0]
    if T == 0:
        return np.zeros(153, dtype=np.float32)

    xy   = keypoints[:, :, :2]   # (T, 17, 2)
    conf = keypoints[:, :,  2]   # (T, 17)

    # Временны́е статистики по (x, y)
    mean_xy = xy.mean(axis=0).ravel()   # 34
    std_xy  = xy.std(axis=0).ravel()    # 34
    min_xy  = xy.min(axis=0).ravel()    # 34
    max_xy  = xy.max(axis=0).ravel()    # 34

    # Средняя уверенность по каждой точке
    mean_conf = conf.mean(axis=0)       # 17

    return np.concatenate([mean_xy, std_xy, min_xy, max_xy, mean_conf]).astype(np.float32)


# ------------------------------------------------------------------
def load_features_from_dir(
    npz_dir: str,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Загружает .npz файлы и строит матрицу признаков.

    Возвращает
    ----------
    X          : (N, 153)  — матрица признаков
    y          : (N,)      — коды действий ('COUG', 'SNEE', …)
    filenames  : (N,)      — имена видеофайлов
    """
    npz_dir = Path(npz_dir)
    X, y, filenames = [], [], []
    missing = 0

    for _, row in df.iterrows():
        stem     = Path(row["filepath"]).stem
        npz_path = npz_dir / f"{stem}.npz"

        if not npz_path.exists():
            missing += 1
            continue

        data  = np.load(str(npz_path), allow_pickle=True)
        kpts  = data["keypoints"]                 # (T, 17, 3)
        feat  = compute_video_features(kpts)      # (153,)

        X.append(feat)
        y.append(row["action"])
        filenames.append(row["filename"])

    if missing:
        print(f"[load_features] Пропущено {missing} файлов (не найдены .npz).")

    return np.array(X, dtype=np.float32), np.array(y), filenames


# ------------------------------------------------------------------
class ActionEvaluator:
    """
    Обёртка над sklearn-классификатором для оценки распознавания действий.

    Метод fit()     — обучает классификатор на train-признаках.
    Метод evaluate()— возвращает все метрики, включая COUG и SNEE.
    Метод save_results() — сохраняет metrics.csv/.txt, confusion_matrix.png, config.json.
    """

    def __init__(self, model_name: str, config: Optional[Dict] = None):
        self.model_name  = model_name
        self.config      = config or {}
        self.clf         = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
        self.le      = LabelEncoder()
        self.classes_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        y_enc        = self.le.fit_transform(y_train)
        self.classes_ = list(self.le.classes_)
        self.clf.fit(X_train, y_enc)
        print(f"[Evaluator] Обучено на {len(X_train)} образцах, "
              f"{len(self.classes_)} классов: {self.classes_}")

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.le.inverse_transform(self.clf.predict(X))

    # ------------------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = self.predict(X_test)

        metrics: Dict = {
            "accuracy":         float(accuracy_score(y_test, y_pred)),
            "f1_macro":         float(f1_score(y_test, y_pred, average="macro",    zero_division=0)),
            "f1_weighted":      float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "precision_macro":  float(precision_score(y_test, y_pred, average="macro",    zero_division=0)),
            "recall_macro":     float(recall_score(y_test,  y_pred, average="macro",    zero_division=0)),
        }

        # Метрики для целевых классов (COUG, SNEE)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for cls in TARGET_CLASSES:
            if cls in report_dict:
                for metric in ("precision", "recall", "f1-score"):
                    key = f"{cls}_{metric.replace('-score', '')}"
                    metrics[key] = float(report_dict[cls][metric])

        metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)
        metrics["y_test"] = list(y_test)
        metrics["y_pred"] = list(y_pred)

        return metrics

    # ------------------------------------------------------------------
    def save_results(
        self,
        metrics: Dict,
        results_dir: str,
        extra_config: Optional[Dict] = None,
    ):
        """
        Сохраняет все артефакты эксперимента:
            metrics.csv          — численные метрики
            metrics.txt          — читаемый отчёт + classification_report
            confusion_matrix.png — матрица ошибок (seaborn heatmap)
            config.json          — конфигурация запуска
        """
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)

        scalar = {
            k: v for k, v in metrics.items()
            if k not in ("classification_report", "y_test", "y_pred")
        }

        # --- metrics.csv ---
        pd.DataFrame([scalar]).to_csv(out / "metrics.csv", index=False)

        # --- metrics.txt ---
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(out / "metrics.txt", "w", encoding="utf-8") as f:
            f.write(f"Модель      : {self.model_name}\n")
            f.write(f"Дата/время  : {ts}\n\n")
            f.write("── Агрегированные метрики ──\n")
            for k, v in scalar.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                f.write(f"  {k:<35}: {val_str}\n")
            f.write("\n── Classification Report ──\n")
            f.write(metrics.get("classification_report", ""))

        # --- confusion_matrix.png ---
        cm = confusion_matrix(
            metrics["y_test"],
            metrics["y_pred"],
            labels=self.classes_,
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.classes_,
            yticklabels=self.classes_,
            ax=ax,
        )
        ax.set_xlabel("Предсказано",  fontsize=12)
        ax.set_ylabel("Фактически",   fontsize=12)
        ax.set_title(f"Confusion Matrix — {self.model_name}\n{ts}", fontsize=13)
        plt.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        plt.close(fig)

        # --- config.json ---
        cfg = {
            "model_name":       self.model_name,
            "timestamp":        ts,
            "classifier":       repr(self.clf),
            "n_features":       153,
            "target_classes":   list(TARGET_CLASSES),
            "all_classes":      self.classes_,
        }
        cfg.update(self.config)
        if extra_config:
            cfg.update(extra_config)

        (out / "config.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print(f"[Evaluator] Результаты сохранены → {out}")
