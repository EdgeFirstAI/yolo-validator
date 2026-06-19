# benchmarks/coco_dataset.py
"""COCO shadow dataset setup and subset creation.

Builds an idempotent shadow COCO dataset: symlinks val2017 images and the GT
annotation JSON, converts GT to YOLO-format labels via ultralytics, and writes
a coco-val.yaml dataset config for ultralytics val runs.

Subset creation picks the first N images (sorted by filename for
reproducibility), creates symlinks to the shadow dataset's images and labels,
and writes a filtered GT JSON + a subset yaml.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# ---- COCO 80 class names (0-indexed, matching ultralytics ordering) ----
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _names_yaml_block() -> str:
    """Build the 'names:' block for ultralytics dataset YAML."""
    lines = ["names:"]
    for i, name in enumerate(COCO80_NAMES):
        lines.append(f"  {i}: {name}")
    return "\n".join(lines)


def _write_dataset_yaml(yaml_path: Path, path_root: Path, train_rel: str, val_rel: str) -> None:
    content = (
        f"path: {path_root}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        f"nc: 80\n"
        f"{_names_yaml_block()}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def _write_val_txt(txt_path: Path, images_dir: Path) -> None:
    """Write a val2017.txt image list in Ultralytics format.

    Ultralytics detects is_coco=True when the val path ends with 'val2017.txt',
    which enables the coco80→coco91 category_id mapping for JSON predictions.
    """
    jpg_paths = sorted(images_dir.glob("*.jpg"))
    txt_path.write_text("\n".join(str(p) for p in jpg_paths) + "\n", encoding="utf-8")


def setup_shadow(
    shadow_root: str | Path,
    images_dir: str | Path,
    gt_json: str | Path,
    force: bool = False,
) -> Path:
    """Build a shadow COCO dataset (idempotent).

    Creates:
      shadow_root/images/val2017/          — symlinks to images_dir/*.jpg
      shadow_root/annotations/instances_val2017.json  — symlink to gt_json
      shadow_root/labels/val2017/*.txt     — YOLO-format labels via convert_coco
      shadow_root/coco-val.yaml            — ultralytics dataset config

    Args:
        shadow_root: directory to build the shadow dataset in.
        images_dir: path to COCO val2017 images (e.g. ~/Datasets/COCO/val2017).
        gt_json: path to instances_val2017.json.
        force: if True, rebuild even if already done.

    Returns:
        Path to shadow_root.
    """
    shadow_root = Path(shadow_root).expanduser().resolve()
    images_dir = Path(images_dir).expanduser().resolve()
    gt_json = Path(gt_json).expanduser().resolve()

    yaml_path = shadow_root / "coco-val.yaml"
    labels_dir = shadow_root / "labels" / "val2017"

    if yaml_path.exists() and labels_dir.exists() and not force:
        # Check that labels actually have files (not just an empty dir)
        if any(labels_dir.iterdir()):
            print(f"[shadow] already built at {shadow_root} (use force=True to rebuild)")
            return shadow_root

    # 1. Create image symlinks
    img_shadow_dir = shadow_root / "images" / "val2017"
    img_shadow_dir.mkdir(parents=True, exist_ok=True)
    print(f"[shadow] symlinking images from {images_dir} ...")
    for jpg in sorted(images_dir.glob("*.jpg")):
        link = img_shadow_dir / jpg.name
        if not link.exists():
            link.symlink_to(jpg)

    # 2. Symlink annotations
    ann_dir = shadow_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_link = ann_dir / "instances_val2017.json"
    if not ann_link.exists():
        ann_link.symlink_to(gt_json)

    # 3. Convert COCO annotations to YOLO format.
    #    convert_coco uses increment_path() internally, so if shadow_root already
    #    exists it will write to shadow_root-2. Work around this by converting to
    #    a fresh temporary directory, then moving the labels into place.
    already_converted = labels_dir.exists() and any(labels_dir.glob("*.txt"))
    if not already_converted or force:
        import shutil
        import tempfile

        print("[shadow] converting COCO annotations to YOLO format ...")
        from ultralytics.data.converter import convert_coco

        # convert_coco uses increment_path() internally: it appends -2, -3, etc.
        # if the save_dir already exists. Work around by passing a non-existent
        # subdirectory inside the temp dir so increment_path finds nothing to bump.
        with tempfile.TemporaryDirectory(
            dir=shadow_root.parent, prefix="_coco_convert_outer_"
        ) as outer_tmp:
            convert_save_dir = Path(outer_tmp) / "convert_output"
            # convert_save_dir does NOT exist yet → increment_path will use it as-is
            convert_coco(
                labels_dir=str(ann_dir),
                save_dir=str(convert_save_dir),
                use_segments=True,
                use_keypoints=False,
                cls91to80=True,
            )
            # convert_coco writes to convert_save_dir/labels/val2017/
            tmp_labels = convert_save_dir / "labels" / "val2017"
            if not tmp_labels.exists():
                # Fallback: scan for any labels/val2017 directory produced
                candidates = list(Path(outer_tmp).rglob("labels/val2017"))
                if candidates:
                    tmp_labels = candidates[0]
                else:
                    raise RuntimeError(
                        f"convert_coco did not produce labels under {outer_tmp}"
                    )
            labels_dir.mkdir(parents=True, exist_ok=True)
            # Move individual txt files (don't overwrite the dir structure)
            for txt in tmp_labels.glob("*.txt"):
                dest = labels_dir / txt.name
                if not dest.exists() or force:
                    shutil.move(str(txt), str(dest))
        print(f"[shadow] YOLO labels written to {labels_dir}")
    else:
        print(f"[shadow] YOLO labels already exist at {labels_dir}")

    # 4. Write val2017.txt image list and dataset YAML.
    #    Ultralytics detects is_coco=True only when the val path ends with 'val2017.txt',
    #    enabling the coco80→coco91 category_id mapping needed for correct JSON predictions.
    val_txt = shadow_root / "images" / "val2017.txt"
    _write_val_txt(val_txt, img_shadow_dir)
    _write_dataset_yaml(yaml_path, shadow_root, "images/train2017", "images/val2017.txt")
    print(f"[shadow] wrote {yaml_path}")

    return shadow_root


def make_subset(
    shadow_root: str | Path,
    images_dir: str | Path,
    gt_json: str | Path,
    n: int,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Create an N-image subset of the shadow COCO dataset.

    Picks the first N images sorted by filename. Creates:
      shadow_root/subsets/subset{n}/images/val2017/  — symlinks to shadow images
      shadow_root/subsets/subset{n}/labels/val2017/  — symlinks to shadow labels
      shadow_root/subsets/subset{n}/annotations/instances_val2017_subset{n}.json
      shadow_root/subsets/subset{n}/subset_{n}.yaml

    Args:
        shadow_root: the shadow dataset root (produced by setup_shadow).
        images_dir: original COCO val2017 images dir (used to resolve image_ids from GT JSON).
        gt_json: path to instances_val2017.json.
        n: number of images to include.
        force: if True, rebuild even if already done.

    Returns:
        (subset_images_dir, subset_gt_json, subset_yaml) as Paths.
    """
    shadow_root = Path(shadow_root).expanduser().resolve()
    images_dir = Path(images_dir).expanduser().resolve()
    gt_json = Path(gt_json).expanduser().resolve()

    subset_dir = shadow_root / "subsets" / f"subset{n}"
    subset_yaml = subset_dir / f"subset_{n}.yaml"
    subset_images_dir = subset_dir / "images" / "val2017"
    subset_labels_dir = subset_dir / "labels" / "val2017"
    subset_gt_json = subset_dir / "annotations" / f"instances_val2017_subset{n}.json"

    if subset_yaml.exists() and not force:
        print(f"[subset{n}] already built at {subset_dir} (use force=True to rebuild)")
        return subset_images_dir, subset_gt_json, subset_yaml

    # Load GT JSON to get image metadata and build filename → image_id map
    with open(gt_json, encoding="utf-8") as f:
        full_gt = json.load(f)

    # All images sorted by filename for reproducibility
    all_images = sorted(full_gt["images"], key=lambda x: x["file_name"])
    selected = all_images[:n]
    selected_ids = {img["id"] for img in selected}

    # Filter annotations
    selected_annotations = [
        ann for ann in full_gt["annotations"]
        if ann["image_id"] in selected_ids
    ]

    # Write filtered GT JSON
    (subset_dir / "annotations").mkdir(parents=True, exist_ok=True)
    filtered_gt = {
        "images": selected,
        "annotations": selected_annotations,
        "categories": full_gt["categories"],
    }
    with open(subset_gt_json, "w", encoding="utf-8") as f:
        json.dump(filtered_gt, f)
    print(f"[subset{n}] wrote {subset_gt_json} ({len(selected)} images, {len(selected_annotations)} annotations)")

    # Symlink images from shadow dataset
    shadow_images_dir = shadow_root / "images" / "val2017"
    subset_images_dir.mkdir(parents=True, exist_ok=True)
    for img_meta in selected:
        fname = Path(img_meta["file_name"]).name
        link = subset_images_dir / fname
        target = shadow_images_dir / fname
        if not link.exists():
            link.symlink_to(target)

    # Symlink labels from shadow dataset
    shadow_labels_dir = shadow_root / "labels" / "val2017"
    subset_labels_dir.mkdir(parents=True, exist_ok=True)
    for img_meta in selected:
        stem = Path(img_meta["file_name"]).stem
        txt_name = stem + ".txt"
        link = subset_labels_dir / txt_name
        target = shadow_labels_dir / txt_name
        if not link.exists() and target.exists():
            link.symlink_to(target)

    # Write val2017.txt image list and subset YAML.
    #    Ultralytics detects is_coco=True only when val path ends with 'val2017.txt',
    #    enabling the coco80→coco91 category_id mapping for correct JSON predictions.
    subset_images_txt = subset_dir / "images" / "val2017.txt"
    _write_val_txt(subset_images_txt, subset_images_dir)
    _write_dataset_yaml(subset_yaml, subset_dir, "images/val2017.txt", "images/val2017.txt")
    print(f"[subset{n}] wrote {subset_yaml}")

    return subset_images_dir, subset_gt_json, subset_yaml
