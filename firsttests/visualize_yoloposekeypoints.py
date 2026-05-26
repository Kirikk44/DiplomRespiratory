# skeleton_visualizer.py
"""
Визуализатор COCO-17 скелетных ключевых точек из .npz файлов.

Режимы (subcommands):
  play   — интерактивный просмотр одного .npz (OpenCV)
  gif    — сохранить анимацию как GIF или MP4
  batch  — последовательно просмотреть N файлов из директории
  sheet  — PNG-коллаж из средних кадров всех видео

Управление в режиме play:
  Пробел       — пауза / воспроизведение
  ← A  /  → D  — кадр назад / вперёд
  R            — перемотка в начало
  S            — сохранить текущий кадр как PNG
  G            — сохранить всю анимацию как GIF
  +  /  -      — увеличить / уменьшить скорость
  Q / Esc      — выход
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.animation as mpl_anim
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# COCO-17 метаданные
# ─────────────────────────────────────────────────────────────────────────────
COCO_KPT_NAMES: List[str] = [
    "nose",         # 0
    "left_eye",     # 1
    "right_eye",    # 2
    "left_ear",     # 3
    "right_ear",    # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# (kpt_a, kpt_b, color_BGR)  — цвет по группам тела
_Y  = (0, 220, 255)   # голова   — жёлтый
_G  = (60, 220, 60)   # торс     — зелёный
_BL = (255, 140, 30)  # лево     — синий
_RD = (30, 80, 255)   # право    — красный
_CL = (200, 220, 0)   # лев.нога — голубой
_CR = (0, 200, 220)   # пр.нога  — циан

SKELETON_EDGES: List[Tuple[int, int, Tuple[int, int, int]]] = [
    # голова
    (0,  1,  _Y), (0,  2,  _Y),
    (1,  3,  _Y), (2,  4,  _Y),
    # торс
    (5,  6,  _G), (5, 11,  _G), (6, 12,  _G), (11, 12, _G),
    # левая рука
    (5,  7, _BL), (7,  9, _BL),
    # правая рука
    (6,  8, _RD), (8, 10, _RD),
    # левая нога
    (11, 13, _CL), (13, 15, _CL),
    # правая нога
    (12, 14, _CR), (14, 16, _CR),
]

# Цвет точки по индексу (BGR)
KPT_COLOR: List[Tuple[int, int, int]] = [
    _Y, _Y, _Y, _Y, _Y,          # голова
    _BL, _RD, _BL, _RD, _BL, _RD,  # плечи, локти, запястья
    _G,  _G,                      # бёдра
    _CL, _CR, _CL, _CR,           # колени, лодыжки
]

CONF_THRESHOLD = 0.20   # ниже — точка не отображается

# waitKeyEx коды стрелок (Windows)
_KEY_LEFT  = 2424832
_KEY_RIGHT = 2555904


# ─────────────────────────────────────────────────────────────────────────────
# Низкоуровневые функции рисования
# ─────────────────────────────────────────────────────────────────────────────

def _scale_kpts(kpts: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Приводит координаты к пиксельным.
    Если max(x, y) ≤ 1.5 — считаем нормализованными [0, 1].
    """
    out = kpts.astype(np.float32).copy()
    if out[:, :2].max() <= 1.5:
        out[:, 0] *= w
        out[:, 1] *= h
    return out


def draw_skeleton(
    canvas: np.ndarray,
    kpts: np.ndarray,          # (17, 3) — (x, y, conf)
    joint_radius: int = 5,
    bone_thickness: int = 2,
) -> np.ndarray:
    """Рисует скелет на canvas. Возвращает новый кадр (оригинал не изменяется)."""
    h, w = canvas.shape[:2]
    frame = canvas.copy()
    kp = _scale_kpts(kpts, w, h)

    # Кости
    for (a, b, color) in SKELETON_EDGES:
        xa, ya, ca = kp[a]
        xb, yb, cb = kp[b]
        if ca < CONF_THRESHOLD or cb < CONF_THRESHOLD:
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                 color, bone_thickness, cv2.LINE_AA)

    # Суставы
    for i, (x, y, c) in enumerate(kp):
        if c < CONF_THRESHOLD:
            continue
        color = KPT_COLOR[i]
        r = max(2, int(joint_radius * min(float(c), 1.0)))
        cv2.circle(frame, (int(x), int(y)), r,     color,       -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), r + 1, (0, 0, 0),    1, cv2.LINE_AA)

    return frame


def _overlay_text(
    frame: np.ndarray,
    lines: List[str],
    pos: Tuple[int, int] = (8, 22),
    scale: float = 0.52,
    color: Tuple = (255, 255, 255),
) -> np.ndarray:
    """Добавляет многострочную надпись с тёмной обводкой."""
    frame = frame.copy()
    x, y0 = pos
    step = int(scale * 50)
    for i, line in enumerate(lines):
        y = y0 + i * step
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color,     1, cv2.LINE_AA)
    return frame


def _progress_bar(frame: np.ndarray, idx: int, total: int) -> np.ndarray:
    h, w = frame.shape[:2]
    frame = frame.copy()
    cv2.rectangle(frame, (0, h - 4), (w, h), (50, 50, 50), -1)
    bar_w = int(w * idx / max(total - 1, 1))
    cv2.rectangle(frame, (0, h - 4), (bar_w, h), (0, 180, 255), -1)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты загрузки
# ─────────────────────────────────────────────────────────────────────────────

def load_npz(npz_path: str) -> Tuple[np.ndarray, str, str, str]:
    """
    Загружает .npz файл.
    Возвращает: (keypoints (T,17,3), action_code, subject_id, filename)
    """
    data     = np.load(str(npz_path), allow_pickle=True)
    kpts     = data["keypoints"]
    action   = str(data["action"])   if "action"   in data else "?"
    subject  = str(data["subject"])  if "subject"  in data else "?"
    filename = str(data["filename"]) if "filename" in data else Path(npz_path).stem
    return kpts, action, subject, filename


def _load_video_frames(video_path: str, max_frames: int) -> List[Optional[np.ndarray]]:
    frames: List[Optional[np.ndarray]] = [None] * max_frames
    cap = cv2.VideoCapture(str(video_path))
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames[i] = frame
    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonVisualizer:
    """
    Визуализатор скелетных ключевых точек из .npz файлов.

    Параметры
    ---------
    canvas_w, canvas_h : размер окна / холста (пиксели)
    fps                : скорость воспроизведения
    bg_color           : BGR цвет фона (когда нет оригинального видео)
    """

    def __init__(
        self,
        canvas_w: int = 640,
        canvas_h: int = 480,
        fps: float = 10.0,
        bg_color: Tuple[int, int, int] = (28, 28, 28),
    ):
        self.canvas_w  = canvas_w
        self.canvas_h  = canvas_h
        self.fps       = fps
        self.bg_color  = bg_color

    # ── Построение одного кадра ──────────────────────────────────────────
    def _build_frame(
        self,
        kpts_frame: np.ndarray,
        frame_idx: int,
        total: int,
        label: str,
        video_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if video_frame is not None:
            canvas = cv2.resize(video_frame, (self.canvas_w, self.canvas_h))
        else:
            canvas = np.full((self.canvas_h, self.canvas_w, 3),
                             self.bg_color, dtype=np.uint8)

        canvas = draw_skeleton(canvas, kpts_frame)

        det   = int((kpts_frame[:, 2] > CONF_THRESHOLD).sum())
        lines = [
            label,
            f"Frame {frame_idx + 1}/{total}",
            f"Detected: {det}/17 kpts",
        ]
        canvas = _overlay_text(canvas, lines)
        canvas = _progress_bar(canvas, frame_idx, total)
        return canvas

    # ── Режим 1: интерактивный просмотр ─────────────────────────────────
    def play(
        self,
        npz_path: str,
        video_path: Optional[str] = None,
    ):
        """
        Интерактивное воспроизведение с управлением клавиатурой.

        Клавиши:
          Пробел  — пауза/воспроизведение
          ← / A   — предыдущий кадр
          → / D   — следующий кадр
          R       — в начало
          +       — быстрее (fps × 1.5)
          -       — медленнее (fps / 1.5)
          S       — сохранить кадр PNG в visualizations/
          G       — сохранить GIF в visualizations/
          Q / Esc — выход
        """
        kpts, action, subject, filename = load_npz(npz_path)
        T     = kpts.shape[0]
        label = f"{subject} | {action}"

        vframes: List[Optional[np.ndarray]] = [None] * T
        if video_path and Path(video_path).exists():
            vframes = _load_video_frames(video_path, T)

        wname   = f"Skeleton Viewer  [{action}]  {subject}"
        cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(wname, self.canvas_w, self.canvas_h)

        idx      = 0
        playing  = True
        fps      = self.fps
        delay_ms = max(1, int(1000 / fps))

        print(f"\n[Viewer] {filename}  |  {T} кадров  |  {action}")
        print("  Пробел=пауза  ←/A-→/D=кадр  R=начало  ±=скорость  S=PNG  G=GIF  Q=выход\n")

        while True:
            canvas = self._build_frame(kpts[idx], idx, T, label, vframes[idx])
            cv2.imshow(wname, canvas)

            wait = delay_ms if playing else 40
            # waitKeyEx возвращает полный код клавиши (стрелки работают на Windows)
            raw  = cv2.waitKeyEx(wait)
            key  = raw & 0xFF

            if key in (ord("q"), 27):                          # Q / Esc → выход
                break
            elif key == ord(" "):                              # Пробел → пауза
                playing = not playing
                print(f"  {'▶' if playing else '⏸'} {'play' if playing else 'pause'}")
            elif key in (ord("a"),) or raw == _KEY_LEFT:       # ← → предыдущий
                idx     = max(0, idx - 1)
                playing = False
            elif key in (ord("d"),) or raw == _KEY_RIGHT:      # → → следующий
                idx     = min(T - 1, idx + 1)
                playing = False
            elif key == ord("r"):                              # R → начало
                idx = 0
            elif key == ord("+") or key == ord("="):           # + → быстрее
                fps      = min(fps * 1.5, 60.0)
                delay_ms = max(1, int(1000 / fps))
                print(f"  Speed: {fps:.1f} fps")
            elif key == ord("-"):                              # - → медленнее
                fps      = max(fps / 1.5, 1.0)
                delay_ms = max(1, int(1000 / fps))
                print(f"  Speed: {fps:.1f} fps")
            elif key == ord("s"):                              # S → PNG
                Path("visualizations").mkdir(exist_ok=True)
                out = f"visualizations/{Path(npz_path).stem}_f{idx:03d}.png"
                cv2.imwrite(out, canvas)
                print(f"  [S] Saved → {out}")
            elif key == ord("g"):                              # G → GIF
                cv2.destroyWindow(wname)
                self.save_gif(npz_path, video_path=video_path)
                cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(wname, self.canvas_w, self.canvas_h)

            if playing:
                idx = (idx + 1) % T

        cv2.destroyWindow(wname)

    # ── Режим 2: сохранение GIF / MP4 ───────────────────────────────────
    def save_gif(
        self,
        npz_path: str,
        output_path: Optional[str] = None,
        video_path: Optional[str] = None,
        fmt: str = "gif",
    ):
        """
        Сохраняет скелетную анимацию как GIF (PillowWriter) или MP4 (ffmpeg).
        """
        kpts, action, subject, filename = load_npz(npz_path)
        T     = kpts.shape[0]
        label = f"{subject} | {action}"

        vframes: List[Optional[np.ndarray]] = [None] * T
        if video_path and Path(video_path).exists():
            vframes = _load_video_frames(video_path, T)

        # Строим все кадры заранее
        print(f"[GIF] Рендеринг {T} кадров…")
        rgb_frames = []
        for i in range(T):
            bgr   = self._build_frame(kpts[i], i, T, label, vframes[i])
            rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # Matplotlib FuncAnimation + PillowWriter
        fig, ax = plt.subplots(
            figsize=(self.canvas_w / 100, self.canvas_h / 100), dpi=100
        )
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        im  = ax.imshow(rgb_frames[0])

        def _update(i: int):
            im.set_data(rgb_frames[i])
            return (im,)

        ani = mpl_anim.FuncAnimation(
            fig, _update, frames=T,
            interval=int(1000 / self.fps),
            blit=True,
        )

        Path("visualizations").mkdir(exist_ok=True)
        stem = output_path or f"visualizations/{Path(npz_path).stem}.{fmt}"

        if fmt == "gif":
            writer = mpl_anim.PillowWriter(fps=self.fps)
            ani.save(stem, writer=writer)
        elif fmt in ("mp4", "avi"):
            ani.save(stem, writer="ffmpeg", fps=self.fps)

        plt.close(fig)
        print(f"[GIF] Сохранено → {stem}")

    # ── Режим 3: пакетный просмотр ──────────────────────────────────────
    def batch(
        self,
        npz_dir: str,
        action_filter: Optional[str] = None,
        n: int = 10,
        video_dir: Optional[str] = None,
    ):
        """Последовательно показывает N видео из директории."""
        npz_dir = Path(npz_dir)
        files   = sorted(npz_dir.glob("*.npz"))

        if action_filter:
            files = [f for f in files if action_filter.upper() in f.name]

        files = files[:n]
        print(f"\n[Batch] {len(files)} файлов  |  фильтр: {action_filter or 'все'}")

        for i, npz_path in enumerate(files):
            print(f"\n  [{i + 1}/{len(files)}]  {npz_path.name}")
            video_path = None
            if video_dir:
                for ext in (".avi", ".mp4", ".mov"):
                    vp = Path(video_dir) / (npz_path.stem + ext)
                    if vp.exists():
                        video_path = str(vp)
                        break
            self.play(str(npz_path), video_path=video_path)

    # ── Режим 4: контактный лист (коллаж) ───────────────────────────────
    def contact_sheet(
        self,
        npz_dir: str,
        action_filter: Optional[str] = None,
        cols: int = 6,
        output_path: Optional[str] = None,
    ):
        """
        Создаёт PNG-коллаж: по одному среднему кадру от каждого .npz.
        Удобно для быстрой проверки качества извлечения на всём датасете.
        """
        npz_dir = Path(npz_dir)
        files   = sorted(npz_dir.glob("*.npz"))
        if action_filter:
            files = [f for f in files if action_filter.upper() in f.name]

        if not files:
            print("[Sheet] Файлы не найдены.")
            return

        TW, TH = 192, 144   # размер одной миниатюры
        thumbs  = []

        for npz_path in files:
            kpts, action, subject, _ = load_npz(str(npz_path))
            mid    = kpts.shape[0] // 2
            canvas = np.full((TH, TW, 3), self.bg_color, dtype=np.uint8)
            canvas = draw_skeleton(canvas, kpts[mid],
                                   joint_radius=3, bone_thickness=1)
            canvas = _overlay_text(
                canvas,
                [f"{subject}", action],
                scale=0.38,
                color=(200, 200, 200),
            )
            thumbs.append(canvas)

        # Дополняем до кратного cols
        while len(thumbs) % cols:
            thumbs.append(np.full((TH, TW, 3), self.bg_color, dtype=np.uint8))

        rows  = len(thumbs) // cols
        grid  = np.vstack([
            np.hstack(thumbs[r * cols:(r + 1) * cols])
            for r in range(rows)
        ])

        Path("visualizations").mkdir(exist_ok=True)
        out = output_path or f"visualizations/sheet_{action_filter or 'all'}.png"
        cv2.imwrite(out, grid)
        print(f"[Sheet] Коллаж сохранён → {out}  ({len(files)} видео, {cols} кол.)")

        cv2.namedWindow("Contact Sheet", cv2.WINDOW_NORMAL)
        cv2.imshow("Contact Sheet", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Skeleton Visualizer — просмотр COCO-17 .npz ключевых точек",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--width",  type=int,   default=640,  help="Ширина окна")
    parser.add_argument("--height", type=int,   default=480,  help="Высота окна")
    parser.add_argument("--fps",    type=float, default=10.0, help="FPS воспроизведения")

    sub = parser.add_subparsers(dest="mode", required=True)

    # play
    p = sub.add_parser("play", help="Интерактивный просмотр одного .npz")
    p.add_argument("npz",       help="Путь к .npz файлу")
    p.add_argument("--video",   help="Путь к оригинальному .avi (опционально)")

    # gif
    g = sub.add_parser("gif", help="Сохранить анимацию как GIF или MP4")
    g.add_argument("npz",       help="Путь к .npz файлу")
    g.add_argument("--output",  help="Выходной файл (по умолчанию visualizations/*.gif)")
    g.add_argument("--video",   help="Путь к оригинальному .avi")
    g.add_argument("--fmt",     default="gif", choices=["gif", "mp4"],
                   help="Формат вывода")

    # batch
    b = sub.add_parser("batch", help="Последовательный просмотр из директории")
    b.add_argument("npz_dir",       help="Папка с .npz файлами")
    b.add_argument("--action",      help="Фильтр по коду действия (COUG, SNEE, …)")
    b.add_argument("--n",   type=int, default=10, help="Максимум видео")
    b.add_argument("--video-dir",   help="Папка с оригинальными .avi")

    # sheet
    s = sub.add_parser("sheet", help="PNG-коллаж средних кадров")
    s.add_argument("npz_dir",       help="Папка с .npz файлами")
    s.add_argument("--action",      help="Фильтр по коду действия")
    s.add_argument("--cols", type=int, default=6, help="Столбцов в коллаже")
    s.add_argument("--output",      help="Путь к PNG файлу")

    args = parser.parse_args()
    vis  = SkeletonVisualizer(
        canvas_w=args.width,
        canvas_h=args.height,
        fps=args.fps,
    )

    if args.mode == "play":
        vis.play(args.npz, video_path=args.video)

    elif args.mode == "gif":
        vis.save_gif(args.npz, output_path=args.output,
                     video_path=args.video, fmt=args.fmt)

    elif args.mode == "batch":
        vis.batch(args.npz_dir, action_filter=args.action,
                  n=args.n, video_dir=args.video_dir)

    elif args.mode == "sheet":
        vis.contact_sheet(args.npz_dir, action_filter=args.action,
                          cols=args.cols, output_path=args.output)


if __name__ == "__main__":
    main()
