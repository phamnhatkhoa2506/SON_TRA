"""
Dump các crop zone (như VLM nhìn thấy) ra thư mục để inspect chất lượng input.

Mỗi crop = bbox zone + padding + viền đỏ ôm polygon zone target.
Tên file: zone_<zid>_area<area>.png  (sort dễ theo area)

Thay đổi DUMP_LIMIT / DUMP_FILTER để chọn subset cần xem.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ─── Config (đồng bộ với vision_local_label.py) ───
IMG_PATH    = './map_300dpi.png'

LABELS_PATH_PREFERRED = './results/v1_dpi_300/v1_labels_split.npy'
LABELS_PATH_FALLBACK  = './results/v1_dpi_300/v1_labels.npy'
ZONES_JSON_PREFERRED  = './results/v1_dpi_300/to_dan_pho_v1_labeled.json'
ZONES_JSON_FALLBACK   = './results/v1_dpi_300/to_dan_pho_v1.json'

OUT_DIR     = './zone_crops'

PADDING_PX  = 10
OUTLINE_BGR = (0, 0, 255)
OUTLINE_THK = 3
MIN_CROP    = 128
MIN_AREA    = 1500

# Bộ lọc: 'all' / 'unlabeled' / 'labeled' / 'duplicates'
DUMP_FILTER = 'all'

# Giới hạn số crop dump ra (None = không giới hạn). Áp dụng SAU khi sort theo area giảm dần.
DUMP_LIMIT  = 50


def make_zone_crop(img, labels, zone_id, bbox, pad, min_size):
    """Crop bbox zone + padding + outline đỏ. (giống vision_local_label.py)"""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bbox
    cw = x1 - x0
    ch = y1 - y0
    extra_w = max(0, (min_size - cw) // 2)
    extra_h = max(0, (min_size - ch) // 2)
    cx0 = max(0, x0 - pad - extra_w)
    cy0 = max(0, y0 - pad - extra_h)
    cx1 = min(W, x1 + pad + extra_w)
    cy1 = min(H, y1 + pad + extra_h)
    crop = img[cy0:cy1, cx0:cx1].copy()
    mask = (labels[cy0:cy1, cx0:cx1] == zone_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, contours, -1, OUTLINE_BGR, OUTLINE_THK)
    return crop


def main():
    # ── Load image ──
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"ERROR: không đọc được {IMG_PATH}")
        sys.exit(1)
    H, W = img.shape[:2]
    print(f"Image: {W}x{H}")

    # ── Auto-pick labels/zones ──
    if Path(LABELS_PATH_PREFERRED).exists() and Path(ZONES_JSON_PREFERRED).exists():
        labels_path = LABELS_PATH_PREFERRED
        zones_path  = ZONES_JSON_PREFERRED
    elif Path(LABELS_PATH_FALLBACK).exists() and Path(ZONES_JSON_FALLBACK).exists():
        labels_path = LABELS_PATH_FALLBACK
        zones_path  = ZONES_JSON_FALLBACK
    else:
        print("ERROR: không tìm thấy labels.npy / zones.json")
        sys.exit(1)
    print(f"Labels: {labels_path}")
    print(f"Zones:  {zones_path}")

    labels = np.load(labels_path)
    lh, lw = labels.shape
    print(f"Labels shape: {labels.shape}")

    # DPI sync: upscale labels + zone metadata nếu khác size ảnh
    if (lw, lh) != (W, H):
        sx = W / lw
        sy = H / lh
        INV_SCALE = (sx + sy) / 2
        print(f"   Upscale labels {lw}x{lh} → {W}x{H} (×{INV_SCALE:.2f})")
        labels = cv2.resize(labels.astype(np.int32), (W, H),
                            interpolation=cv2.INTER_NEAREST)
    else:
        sx = sy = INV_SCALE = 1.0

    with open(zones_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    zones = data['zones']
    print(f"Total zones: {len(zones)}")

    # Scale bbox + centroids nếu labels/zones ở DPI thấp hơn ảnh
    if INV_SCALE != 1.0:
        for z in zones:
            cx, cy = z['centroid_px']
            z['centroid_px'] = [int(round(cx * sx)), int(round(cy * sy))]
            if 'bbox_px' in z:
                bx0, by0, bx1, by1 = z['bbox_px']
                z['bbox_px'] = [int(round(bx0 * sx)), int(round(by0 * sy)),
                                int(round(bx1 * sx)), int(round(by1 * sy))]

    # ── Filter ──
    to_to_zones = defaultdict(list)
    for z in zones:
        if z.get('to_so') is not None:
            to_to_zones[z['to_so']].append(z['zone_id'])
    dup_zids = {zid for zids in to_to_zones.values() if len(zids) > 1
                for zid in zids}

    selected = []
    for z in zones:
        if z.get('area_px', 0) < MIN_AREA:
            continue
        if 'bbox_px' not in z:
            continue
        if DUMP_FILTER == 'unlabeled' and z.get('to_so') is not None:
            continue
        if DUMP_FILTER == 'labeled' and z.get('to_so') is None:
            continue
        if DUMP_FILTER == 'duplicates' and z['zone_id'] not in dup_zids:
            continue
        selected.append(z)

    selected.sort(key=lambda z: -z.get('area_px', 0))
    if DUMP_LIMIT is not None:
        selected = selected[:DUMP_LIMIT]
    print(f"Filter='{DUMP_FILTER}', dumping {len(selected)} crops")

    # ── Dump ──
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    for z in selected:
        zid = z['zone_id']
        area = z.get('area_px', 0)
        crop = make_zone_crop(img, labels, zid, z['bbox_px'],
                              PADDING_PX, MIN_CROP)
        if crop.size == 0:
            continue
        # Tên file gồm area để sort dễ + to_so nếu có
        to_so = z.get('to_so')
        suffix = f"_to{to_so}" if to_so is not None else ""
        fname = f"zone_{zid:04d}_area{area:06d}{suffix}.png"
        cv2.imwrite(str(out_dir / fname), crop)

    print(f"Done. {len(selected)} files in {out_dir}")
    # In sample paths
    files = sorted(out_dir.iterdir())[:5]
    for f in files:
        print(f"  {f}")


if __name__ == '__main__':
    main()
