"""Score an edgefirst-profiler predictions.parquet against COCO GT (offline).

The profiler emits predictions.parquet; mAP is normally scored by a separate
Studio validator service (network). This bridges that offline: it converts the
parquet to COCO-format detections and scores them through the harness's
``canonical_eval`` (pycocotools, restricted to the predicted image_ids).

Parquet schema (per detection row): name (COCO image_id), label_index (0-79),
box2d ([cx,cy,w,h] normalized), box2d_score, size ([W,H]), mask (PNG full-image
binary). Verified conventions: image_id=int(name); W,H=size[0],size[1];
bbox=[(cx-w/2)W,(cy-h/2)H,wW,hH]; category_id=coco80_to_coco91()[label_index];
mask decodes to (H,W).

Usage::

    python -m benchmarks.score_profiler_parquet \
        --parquet ~/Library/Caches/edgefirst-profiler/validation/v-209e/predictions.parquet \
        --gt ~/Dataset/COCO/annotations/instances_val2017.json \
        --iou bbox,segm
"""
from __future__ import annotations

import argparse
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils

from benchmarks.canonical_eval import canonical_eval
from yolo_validator.coco_output import coco80_to_coco91


def build_predictions(parquet_path, iou_types, conf=0.0):
    cmap = coco80_to_coco91()
    need_mask = "segm" in iou_types
    cols = ["name", "label_index", "box2d", "box2d_score", "size"]
    if need_mask:
        cols.append("mask")
    f = pq.ParquetFile(parquet_path)
    preds: list[dict] = []
    imgs: set[int] = set()
    shape_checked = False
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=cols)
        names = t.column("name").to_pylist()
        lidx = t.column("label_index").to_pylist()
        boxes = t.column("box2d").to_pylist()
        scores = t.column("box2d_score").to_pylist()
        sizes = t.column("size").to_pylist()
        masks = t.column("mask").to_pylist() if need_mask else None
        for i in range(len(names)):
            s = float(scores[i])
            if s < conf:
                continue
            iid = int(names[i])
            imgs.add(iid)
            w_img, h_img = int(sizes[i][0]), int(sizes[i][1])
            cx, cy, bw, bh = (float(v) for v in boxes[i])
            rec = {
                "image_id": iid,
                "category_id": int(cmap[int(lidx[i])]),
                "bbox": [(cx - bw / 2) * w_img, (cy - bh / 2) * h_img,
                         bw * w_img, bh * h_img],
                "score": s,
            }
            if need_mask:
                m = np.array(Image.open(io.BytesIO(masks[i])))
                if m.ndim == 3:
                    m = m[..., 0]
                if not shape_checked:
                    assert m.shape == (h_img, w_img), \
                        f"mask {m.shape} != (H,W)=({h_img},{w_img}) — size order wrong"
                    shape_checked = True
                rle = mask_utils.encode(np.asfortranarray((m > 0).astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                rec["segmentation"] = rle
            preds.append(rec)
    return preds, imgs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True)
    p.add_argument("--gt", default=os.path.expanduser(
        "~/Dataset/COCO/annotations/instances_val2017.json"))
    p.add_argument("--iou", default="bbox", help="comma list: bbox,segm")
    p.add_argument("--conf", type=float, default=0.0)
    a = p.parse_args()
    iou_types = tuple(x.strip() for x in a.iou.split(",") if x.strip())
    parquet = os.path.expanduser(a.parquet)
    preds, imgs = build_predictions(parquet, iou_types, a.conf)
    print(f"built {len(preds)} detections over {len(imgs)} images "
          f"(conf>={a.conf}) from {os.path.basename(os.path.dirname(parquet))}")
    res = canonical_eval(os.path.expanduser(a.gt), preds, iou_types)
    for t in iou_types:
        r = res[t]
        print(f"[{t}] AP={r['AP']:.4f}  AP50={r['AP50']:.4f}  AP75={r['AP75']:.4f}  "
              f"APs={r['APs']:.4f} APm={r['APm']:.4f} APl={r['APl']:.4f}")


if __name__ == "__main__":
    main()
