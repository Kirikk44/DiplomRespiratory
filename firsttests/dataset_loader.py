# dataset_loader.py
import re
from pathlib import Path
from typing import Optional
import pandas as pd


class DatasetLoader:
    """
    Сканирует директорию датасета, парсит имена файлов и строит DataFrame.
    Формат имени: {SID}_{G}_{ACTION}_{LOCO}_{POSE}[_HF].avi
    """

    # Субъекты для тестовой выборки согласно описанию датасета
    TEST_SUBJECTS = {"S002", "S003", "S004", "S005", "S006"}

    ACTION_MAP = {
        "CALL": "answer_phone_call",
        "COUG": "cough",
        "DRIN": "drink_water",
        "SCRA": "scratch_head",
        "SNEE": "sneeze",
        "STRE": "stretch_arms",
        "WAVE": "wave_hand",
        "WIPE": "wipe_glasses",
    }
    GENDER_MAP     = {"M": "male",     "F": "female"}
    LOCOMOTION_MAP = {"STD": "standing", "WLK": "walking"}
    POSE_MAP       = {"FCE": "face_camera", "LFT": "face_left", "RGT": "face_right"}

    _PATTERN = re.compile(
        r"^(S\d{3})_([MF])_(CALL|COUG|DRIN|SCRA|SNEE|STRE|WAVE|WIPE)"
        r"_(STD|WLK)_(FCE|LFT|RGT)(_HF)?\.avi$"
    )

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Рекурсивно сканирует dataset_dir и парсит все подходящие .avi файлы."""
        records = []
        for filepath in sorted(self.dataset_dir.rglob("*.avi")):
            m = self._PATTERN.match(filepath.name)
            if not m:
                continue
            subject, gender, action, loco, pose, hf = m.groups()
            records.append({
                "filepath":     str(filepath),
                "filename":     filepath.name,
                "subject":      subject,
                "gender":       gender,
                "action":       action,           # код (COUG, SNEE, …)
                "action_label": self.ACTION_MAP[action],
                "locomotion":   loco,
                "pose":         pose,
                "is_flipped":   hf is not None,
                "split":        "test" if subject in self.TEST_SUBJECTS else "train",
            })

        self.df = pd.DataFrame(records)
        print(f"[DatasetLoader] Загружено {len(self.df)} видео из {self.dataset_dir}")
        return self.df

    # ------------------------------------------------------------------
    def get_split(self, split: str) -> pd.DataFrame:
        """Возвращает train или test подмножество."""
        self._check_loaded()
        return self.df[self.df["split"] == split].reset_index(drop=True)

    # ------------------------------------------------------------------
    def print_stats(self, save_path: Optional[str] = None) -> str:
        """Выводит сводную статистику в консоль и (опционально) сохраняет в файл."""
        self._check_loaded()
        lines: list[str] = []

        def section(title: str, series: pd.Series):
            lines.append(f"\n{'─' * 45}")
            lines.append(f"  {title}")
            lines.append("─" * 45)
            for k, v in series.items():
                lines.append(f"  {str(k):<35} {v:>6}")

        total = len(self.df)
        lines.append("=" * 45)
        lines.append(f"  СТАТИСТИКА ДАТАСЕТА  (всего: {total})")
        lines.append(f"  Директория: {self.dataset_dir}")
        lines.append("=" * 45)

        section("Train / Test split",      self.df["split"].value_counts())
        section("По коду действия",        self.df["action"].value_counts().sort_index())
        section("По полу",                 self.df["gender"].value_counts())
        section("По типу передвижения",    self.df["locomotion"].value_counts())
        section("По ракурсу",              self.df["pose"].value_counts())
        section("Оригинал / HF-версии",
                self.df["is_flipped"]
                    .map({True: "flipped", False: "original"})
                    .value_counts())

        lines.append(f"\n{'─' * 45}")
        lines.append("  Распределение действий по split")
        lines.append("─" * 45)
        cross = self.df.groupby(["split", "action"]).size().unstack(fill_value=0)
        lines.append(cross.to_string())

        # Целевые классы
        for code in ("COUG", "SNEE"):
            sub = self.df[self.df["action"] == code]
            lines.append(f"\n  [{code}] train={len(sub[sub.split=='train'])}  "
                         f"test={len(sub[sub.split=='test'])}")

        output = "\n".join(lines)
        print(output)

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(output, encoding="utf-8")
            print(f"[DatasetLoader] Статистика сохранена → {save_path}")

        return output

    def _check_loaded(self):
        if self.df is None:
            raise RuntimeError("Сначала вызовите load().")
    