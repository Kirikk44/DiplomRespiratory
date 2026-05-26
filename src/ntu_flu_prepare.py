"""
prepare_ntu_flu.py
Раскидывает flu-relevant .skeleton файлы NTU по папкам классов.
Никакой конвертации — только копирование.

Запуск:
    python prepare_ntu_flu.py
    python prepare_ntu_flu.py --ignored NTU_RGBD120_samples_with_missing_skeletons.txt
"""

from pathlib import Path
from shutil import copyfile
from collections import Counter
import argparse

# action_id (1-indexed) → имя папки
FLU_ACTION_MAP = {
    1:   "drink",
    3:   "brush_teeth",
    10:  "clapping",
    23:  "hand_wave",
    28:  "phone_call",
    34:  "rub_hands",
    35:  "nod_head",
    37:  "wipe_face",
    41:  "cough",       # A041 = sneeze/cough
    48:  "nausea",
    79:  "sniff",
    103: "yawn",
    105: "blow_nose",
}


def parse_action_id(stem: str) -> int:
    stem = stem.upper()
    idx = stem.rfind("A")
    if idx == -1:
        return -1
    try:
        return int(stem[idx + 1: idx + 4])
    except ValueError:
        return -1


def prepare(
    roots: list,
    out_root: str = "ntu_flu_skeletons",
    ignored_txt: str | None = None,
    skip_existing: bool = True,
):
    out_root = Path(out_root)
    roots = [Path(r) for r in roots]

    ignored = set()
    if ignored_txt and Path(ignored_txt).exists():
        for line in Path(ignored_txt).read_text(encoding="utf-8").splitlines():
            ignored.add(line.strip().replace(".skeleton", ""))
        print(f"Игнорируем {len(ignored)} файлов из missing-list")

    all_files = []
    for root in roots:
        found = list(root.rglob("*.skeleton"))
        print(f"{root}: найдено {len(found)} .skeleton")
        all_files.extend(found)

    counts  = Counter()
    skipped = 0

    for src in all_files:
        stem = src.stem
        if stem in ignored:
            skipped += 1
            continue

        action_id = parse_action_id(stem)
        if action_id not in FLU_ACTION_MAP:
            continue

        cls = FLU_ACTION_MAP[action_id]
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / src.name

        if skip_existing and dst.exists():
            counts[cls] += 1
            continue

        copyfile(src, dst)
        counts[cls] += 1

    print("\nРазложено по классам:")
    total = 0
    for cls in sorted(counts):
        print(f"  {cls:<15}: {counts[cls]:>6}")
        total += counts[cls]
    print(f"  {'ИТОГО':<15}: {total:>6}")
    print(f"  Пропущено/ignored: {skipped}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=[
        r"..\nturgbd_skeletons_s001_to_s017\nturgb+d_skeletons",
        r"..\nturgbd_skeletons_s018_to_s032",
    ])
    p.add_argument("--out",     default="ntu_flu_skeletons")
    p.add_argument("--ignored", default=None)
    p.add_argument("--no-skip", action="store_true")
    args = p.parse_args()
    prepare(args.roots, args.out, args.ignored, not args.no_skip)