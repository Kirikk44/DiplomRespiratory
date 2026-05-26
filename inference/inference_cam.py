"""
inference_webcam.py
Real-time cough/action recognition from webcam using STGCNStage2 + MediaPipe Pose.

Requirements:
    pip install mediapipe opencv-python torch numpy scipy omegaconf

Usage:
    python inference_webcam.py --checkpoint path/to/checkpoint.ckpt --camera 0
"""

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COCO17_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

BIISK_CLASSES = ["cough", "call", "drink", "scratch", "sneeze", "stretch", "wave", "wipe"]

# MediaPipe Pose (33 landmarks) → COCO-17 mapping
MP_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _build_adjacency(num_joints: int = 17) -> torch.Tensor:
    A = torch.zeros(num_joints, num_joints)
    for i, j in COCO17_EDGES:
        A[i, j] = A[j, i] = 1.0
    A += torch.eye(num_joints)
    D = A.sum(dim=1).pow(-0.5)
    D[D == float("inf")] = 0
    return D.unsqueeze(1) * A * D.unsqueeze(0)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, temporal_kernel=9, stride=1, dropout=0.0):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(temporal_kernel, 1),
                      padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_ch), nn.Dropout(dropout),
        )
        self.skip = (nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=(stride, 1)),
                                   nn.BatchNorm2d(out_ch))
                     if in_ch != out_ch or stride != 1 else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        xsp = torch.einsum("nctv,vw->nctw", x, self.A)
        xsp = self.gcn(xsp)
        return self.relu(self.tcn(xsp) + self.skip(x))


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_joints):
        super().__init__()
        self.gap     = nn.AdaptiveAvgPool2d((1, num_joints))
        self.conv    = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        B, C, T, V = x.shape
        gap = self.gap(x).squeeze(2)
        w   = self.softmax(self.conv(gap).squeeze(1))
        return x * w.view(B, 1, 1, V)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        B, C, T, V = x.shape
        gap = x.mean(dim=-1)
        w   = torch.sigmoid(self.conv(gap))
        return x * w.unsqueeze(-1)


class STGCNHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.3, num_joints=17,
                 use_spatial_attn=False, use_temporal_attn=False):
        super().__init__()
        self.spatial_attn  = SpatialAttention(in_dim, num_joints) if use_spatial_attn  else nn.Identity()
        self.temporal_attn = TemporalAttention(in_dim)            if use_temporal_attn else nn.Identity()
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten    = nn.Flatten()
        self.proj       = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x    = self.spatial_attn(x)
        x    = self.temporal_attn(x)
        feat = self.proj(self.flatten(self.pool(x)))
        return self.classifier(feat), feat


class STGCN(nn.Module):
    """Inference-only wrapper — no Lightning, no loss."""
    def __init__(self, num_classes=8, hidden_dim=256, dropout=0.3, num_joints=17,
                 use_spatial_attn=True, use_temporal_attn=True, in_channels=2):
        super().__init__()
        A = _build_adjacency(num_joints)
        self.data_bn  = nn.BatchNorm1d(in_channels * num_joints)   # matches checkpoint key
        self.backbone = nn.ModuleList([
            STGCNBlock(in_channels, 64,  A, dropout=dropout),
            STGCNBlock(64,  64,  A, dropout=dropout),
            STGCNBlock(64,  64,  A, dropout=dropout),
            STGCNBlock(64,  128, A, dropout=dropout, stride=2),
            STGCNBlock(128, 128, A, dropout=dropout),
            STGCNBlock(128, hidden_dim, A, dropout=dropout, stride=2),
            STGCNBlock(hidden_dim, hidden_dim, A, dropout=dropout),
        ])
        self.head = STGCNHead(hidden_dim, num_classes, dropout=dropout,
                              num_joints=num_joints,
                              use_spatial_attn=use_spatial_attn,
                              use_temporal_attn=use_temporal_attn)

    def forward(self, x):           # x: (B, 2, T, 17)
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2)
        for layer in self.backbone:
            x = layer(x)
        logits, _ = self.head(x)
        return logits



def load_model(ckpt_path: str, device: torch.device,
               num_classes: int = 8,
               use_spatial_attn: bool = True,
               use_temporal_attn: bool = True) -> STGCN:

    model = STGCN(num_classes=num_classes,
                  use_spatial_attn=use_spatial_attn,
                  use_temporal_attn=use_temporal_attn).to(device)

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    sd_raw = ckpt.get("state_dict", ckpt)

    # Drop keys that don't belong to model weights (e.g. class_weights buffer)
    IGNORE = {"class_weights"}
    sd = {k: v for k, v in sd_raw.items() if k not in IGNORE}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing  ({len(missing)}): {missing[:6]}")
    leftover = [k for k in unexpected if k not in IGNORE]
    if leftover:
        print(f"[WARN] unexpected ({len(leftover)}): {leftover[:6]}")

    model.eval()
    return model


def normalize_skeleton(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 17, 2)  pixel or [0-1] coords
    → centered on hip midpoint, scaled by torso length
    """
    hip  = (seq[:, 11, :] + seq[:, 12, :]) / 2.0
    neck = (seq[:,  5, :] + seq[:,  6, :]) / 2.0
    hip_c  = np.median(hip,  axis=0)
    neck_c = np.median(neck, axis=0)
    torso  = np.linalg.norm(neck_c - hip_c) + 1e-6
    return ((seq - hip_c[None, None, :]) / torso).astype(np.float32)


def make_pose_detector():
    """
    Tries new API (mediapipe >= 0.10) first, falls back to legacy solutions API.
    Returns (detector_obj, api_version: str)
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # Download model if missing
        model_path = Path("pose_landmarker_lite.task")
        if not model_path.exists():
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/latest/"
                   "pose_landmarker_lite.task")
            print(f"[INFO] downloading MediaPipe model from {url} ...")
            urllib.request.urlretrieve(url, model_path)
            print("[INFO] download complete")

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = mp_vision.PoseLandmarker.create_from_options(options)
        print("[INFO] using MediaPipe Task API (>= 0.10)")
        return detector, "task"

    except Exception as e:
        print(f"[INFO] Task API unavailable ({e}), trying legacy solutions API …")

    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[INFO] using MediaPipe legacy solutions API (< 0.10)")
        return pose, "legacy"
    except Exception as e2:
        raise RuntimeError(f"Cannot initialise MediaPipe Pose: {e2}") from e2


def get_landmarks_task(detector, rgb_frame: np.ndarray, timestamp_ms: int):
    """New Task API inference → list[landmark] or None."""
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result   = detector.detect_for_video(mp_image, timestamp_ms)
    if result.pose_landmarks:
        return result.pose_landmarks[0]   # list of NormalizedLandmark
    return None


def get_landmarks_legacy(pose, rgb_frame: np.ndarray):
    """Legacy solutions API inference → list[landmark] or None."""
    result = pose.process(rgb_frame)
    if result.pose_landmarks:
        return result.pose_landmarks.landmark
    return None


def landmarks_to_coco17(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """(33,) landmarks → (17, 2) pixel coords."""
    coords = np.zeros((17, 2), dtype=np.float32)
    for coco_idx, mp_idx in enumerate(MP_TO_COCO17):
        lm = landmarks[mp_idx]
        coords[coco_idx, 0] = lm.x * frame_w
        coords[coco_idx, 1] = lm.y * frame_h
    return coords


class CoughDetector:
    def __init__(self, model: STGCN, device: torch.device,
                 window: int = 64, stride: int = 8, smooth_alpha: float = 0.6):
        self.model   = model
        self.device  = device
        self.window  = window
        self.stride  = stride
        self.alpha   = smooth_alpha
        self.buffer  = collections.deque(maxlen=window)
        self.frame_count = 0
        self.probs   = np.ones(len(BIISK_CLASSES)) / len(BIISK_CLASSES)
        self.label   = "collecting…"
        self.confidence = 0.0

    def push(self, keypoints_xy: np.ndarray):
        self.buffer.append(keypoints_xy.copy())
        self.frame_count += 1
        if len(self.buffer) == self.window and self.frame_count % self.stride == 0:
            self._infer()

    def _infer(self):
        seq = np.stack(self.buffer, axis=0)    # (T, 17, 2)
        seq = normalize_skeleton(seq)

        # (B=1, C=2, T, V=17)
        x = torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)

        logits = out[0] if isinstance(out, (tuple, list)) else out
        new_p  = F.softmax(logits[0], dim=0).cpu().numpy()
        self.probs = self.alpha * new_p + (1 - self.alpha) * self.probs

        idx = int(np.argmax(self.probs))
        self.label      = BIISK_CLASSES[idx]
        self.confidence = float(self.probs[idx])


COLORS = {"cough": (0,0,255), "sneeze": (0,165,255), "default": (0,200,0)}

def draw_skeleton(frame, kps, color=(0,255,0)):
    for i, j in COCO17_EDGES:
        xi, yi = int(kps[i,0]), int(kps[i,1])
        xj, yj = int(kps[j,0]), int(kps[j,1])
        if xi==0 and yi==0: continue
        if xj==0 and yj==0: continue
        cv2.line(frame, (xi,yi), (xj,yj), color, 2)
    for k in range(17):
        x, y = int(kps[k,0]), int(kps[k,1])
        if x==0 and y==0: continue
        cv2.circle(frame, (x,y), 4, color, -1)

def draw_hud(frame, label, confidence, probs, buf_fill, window):
    h, w   = frame.shape[:2]
    color  = COLORS.get(label, COLORS["default"])
    cv2.rectangle(frame, (0,0), (w,50), (0,0,0), -1)
    cv2.putText(frame, f"{label.upper()}  {confidence*100:.1f}%",
                (10,36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    bar_w = int(w * buf_fill / window)
    cv2.rectangle(frame, (0,h-8), (bar_w,h), (100,200,100), -1)
    cv2.putText(frame, f"buf {buf_fill}/{window}", (w-130,h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    bar_h = 14
    for i, (cls, p) in enumerate(zip(BIISK_CLASSES, probs)):
        y0 = 60 + i*(bar_h+3)
        bw = int(180*p)
        c  = COLORS.get(cls, (120,200,120))
        cv2.rectangle(frame, (10,y0), (10+bw,y0+bar_h), c, -1)
        cv2.putText(frame, f"{cls[:6]:6s} {p*100:4.1f}%",
                    (10+bw+4, y0+11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)

def run(ckpt_path: str, camera_id: int = 0,
        window: int = 64, stride: int = 8,
        use_spatial_attn: bool = True,
        use_temporal_attn: bool = True,
        device_str: str = "auto"):

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))
    print(f"[INFO] device: {device}")

    model    = load_model(ckpt_path, device, len(BIISK_CLASSES),
                          use_spatial_attn, use_temporal_attn)
    print(f"[INFO] model loaded from {ckpt_path}")

    detector = CoughDetector(model, device, window=window, stride=stride)
    pose_det, api = make_pose_detector()

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] press  Q  to quit")
    fps_cnt, t0 = 0, time.time()
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] cannot read camera frame")
            break

        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection (API-agnostic)
        if api == "task":
            landmarks = get_landmarks_task(pose_det, rgb, int(time.time() * 1000))
        else:
            landmarks = get_landmarks_legacy(pose_det, rgb)

        kps = np.zeros((17,2), dtype=np.float32)
        if landmarks is not None:
            kps = landmarks_to_coco17(landmarks, w, h)
            draw_skeleton(frame, kps, color=COLORS.get(detector.label, COLORS["default"]))

        detector.push(kps)
        draw_hud(frame, detector.label, detector.confidence,
                 detector.probs, len(detector.buffer), window)

        fps_cnt += 1
        frame_idx += 1
        if fps_cnt % 30 == 0:
            fps = 30 / (time.time() - t0)
            t0  = time.time()
            cv2.putText(frame, f"{fps:.1f} fps",
                        (w-90, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        cv2.imshow("Cough detector — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if api == "task":
        pose_det.close()
    else:
        pose_det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Real-time cough recognition from webcam")
    p.add_argument("--checkpoint", "-c", required=True)
    p.add_argument("--camera",    "-v", type=int, default=0)
    p.add_argument("--window",    "-w", type=int, default=64)
    p.add_argument("--stride",    "-s", type=int, default=8)
    p.add_argument("--no-spatial-attn",  action="store_true")
    p.add_argument("--no-temporal-attn", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    run(
        ckpt_path=args.checkpoint,
        camera_id=args.camera,
        window=args.window,
        stride=args.stride,
        use_spatial_attn=not args.no_spatial_attn,
        use_temporal_attn=not args.no_temporal_attn,
        device_str=args.device,
    )
