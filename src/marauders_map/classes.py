"""COCO class names used by the default YOLO model, with a name→id lookup.

Anything in here can be passed to `marauders --classes ...` or set per-camera in
`house.yaml`. Common animals are highlighted in `COMMON` for easy discovery.
"""
from __future__ import annotations

# Standard COCO-80 class list (Ultralytics default ordering).
COCO_NAMES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(COCO_NAMES)}

# The subset most useful for household tracking (people + pets + visitors).
COMMON: tuple[str, ...] = ("person", "cat", "dog", "bird")


def resolve_classes(spec: str) -> set[int]:
    """Parse a comma-separated class spec into COCO ids.

    Accepts names ("person,cat,dog"), integers ("0,15,16"), or a mix.
    Unknown names raise ValueError with a hint.
    """
    out: set[int] = set()
    for token in spec.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t.isdigit():
            out.add(int(t))
            continue
        if t not in NAME_TO_ID:
            hint = ", ".join(COMMON)
            raise ValueError(f"unknown class {t!r}; try one of: {hint} (or a numeric id)")
        out.add(NAME_TO_ID[t])
    return out


def class_name(cls: int) -> str:
    return COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else f"cls{cls}"
