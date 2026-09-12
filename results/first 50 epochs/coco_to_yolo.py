"""
Convert COCO-format annotations to YOLO .txt label files.

Usage:
    python coco_to_yolo.py \
        --coco-json  <path/to/annotations.json> \
        --images-dir <path/to/images> \
        --labels-dir <path/to/output/labels>

Each image gets one .txt file with one line per bounding box:
    <class_id> <cx> <cy> <w> <h>   (all values normalised 0-1)
Images that have no annotations get an empty .txt file.
"""

import argparse
import json
from pathlib import Path


def convert(coco_json: Path, images_dir: Path, labels_dir: Path) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json) as f:
        data = json.load(f)

    # Map image_id -> (file_name, width, height)
    images = {
        img["id"]: (img["file_name"], img["width"], img["height"])
        for img in data["images"]
    }

    # Map category_id -> zero-based class index (sorted by id for stability)
    cat_ids = sorted(cat["id"] for cat in data["categories"])
    cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {img_id: [] for img_id in images}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    written = skipped = 0
    for img_id, (file_name, img_w, img_h) in images.items():
        stem = Path(file_name).stem
        label_path = labels_dir / f"{stem}.txt"

        lines = []
        for ann in anns_by_image[img_id]:
            x, y, w, h = ann["bbox"]          # COCO: top-left x,y + width,height
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Clamp to [0, 1] to handle occasional out-of-bounds annotations
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            cls = cat_to_idx[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines))
        written += 1
        if not lines:
            skipped += 1

    print(f"Done. {written} label files written ({skipped} empty) -> {labels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="COCO JSON -> YOLO .txt labels")
    parser.add_argument("--coco-json",  required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--labels-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.coco_json.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.coco_json}")
    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    convert(args.coco_json, args.images_dir, args.labels_dir)


if __name__ == "__main__":
    main()
