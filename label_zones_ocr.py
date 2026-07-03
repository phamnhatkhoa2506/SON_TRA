"""
OCR-based labeling for segmented zones.
─ Đọc nhãn "TỔ XX" từ ảnh bản đồ
─ Match mỗi detection với zone tương ứng
─ Sinh ra JSON cuối cùng có trường `to_so`

Pipeline:
1. Load v5_labels.npy + to_dan_pho_v5.json + map_150dpi.png
2. Chạy EasyOCR trên toàn ảnh (vi+en)
3. Filter regex T[ỔỎO0]\\s*\\d+ → list of (to_number, x, y)
4. Với mỗi detection:
   a) Lookup labels[y, x] → zone_id (nếu rơi đúng vào zone)
   b) Nếu zone_id = 0 (rơi ngoài hoặc trên ranh giới) → tìm zone gần nhất (KDTree)
5. Resolve conflict (1 zone có >1 nhãn) → chọn nhãn gần centroid nhất
6. Save to_dan_pho_v5_labeled.json
"""

import cv2
import numpy as np
import json
import re
from collections import defaultdict


def _paddle_dets(res):
    """Parse one PaddleOCR v3 predict() result → [(bbox_4pts, text, conf)]."""
    # PaddleOCR v3 result object: dict-like or attribute-based access
    try:
        polys  = res['dt_polys']
        texts  = res['rec_texts']
        scores = res['rec_scores']
    except (TypeError, KeyError):
        polys  = getattr(res, 'dt_polys',   [])
        texts  = getattr(res, 'rec_texts',  [])
        scores = getattr(res, 'rec_scores', [])
    dets = []
    for i in range(len(texts)):
        if i >= len(polys):
            break
        box = polys[i]
        if hasattr(box, 'tolist'):
            box = box.tolist()
        conf = float(scores[i]) if i < len(scores) else 0.0
        dets.append((box, texts[i], conf))
    return dets
from scipy.spatial import cKDTree

# ─── Config ───
IMG_PATH = r'/content/map_150dpi.png'
LABELS_PATH = r'/content/v5_labels.npy'
ZONES_JSON = r'/content/to_dan_pho_v5.json'
OUTPUT_JSON = r'/content/to_dan_pho_v5_labeled.json'
OUTPUT_VIS = r'/content/v5_labeled_overlay.png'

# Pattern bắt "TỔ XX" — dấu Ổ có thể bị OCR đọc thành Ô, O, 0, Ó
TO_REGEX = re.compile(r'^\s*T[\u00d3\u00d4\u00d6\u00f3\u1ed0\u1ed2\u1ed4\u1ed6\u1ed8O0]+\s*[:.]?\s*(\d{1,3})\s*$', re.IGNORECASE)
# Loose: T + (optional o/O or non-word diacritic chars) + 2-3 digit number
# Handles: "To 53" (o=ổ), "T 65" (space), "TỔ65" via \W match
# EasyOCR: Ổ→Ô/O/0   PaddleOCR: Ổ→Ó (U+00D3) hoặc chữ số 6
# "TÓ 15"→15  "T6 35"→35  "T627"→27  "To 53"→53
TO_REGEX_LOOSE = re.compile(
    r'(?<![A-Za-z])T[ÓóoO6\W]{0,4}(\d{2,3})(?!\d)',
    re.IGNORECASE
)

# Search radius (px) khi detection rơi ngoài zone — tìm zone gần nhất
NEAREST_RADIUS_PX = 80


def main():
    print("1. Loading data")
    img = cv2.imread(IMG_PATH)
    H, W = img.shape[:2]
    print(f"   Image: {W}x{H}")

    labels = np.load(LABELS_PATH)
    print(f"   Labels array: {labels.shape}, n_zones={labels.max()}")

    with open(ZONES_JSON, 'r', encoding='utf-8') as f:
        zones_data = json.load(f)
    zones = zones_data['zones']
    print(f"   Zones loaded: {len(zones)}")

    # KDTree centroids → tìm zone gần nhất nhanh
    zone_centroids = np.array([z['centroid_px'] for z in zones])
    zone_ids = np.array([z['zone_id'] for z in zones])
    zone_id_to_idx = {z['zone_id']: i for i, z in enumerate(zones)}
    tree = cKDTree(zone_centroids)

    # ─── Run OCR with TILING ───
    # Cache reader in sys.modules để tránh PaddleX "already initialized" error
    # khi chạy lại cell trong Jupyter mà không restart kernel.
    print("\n2. Running OCR (tile-based)")
    import sys
    _READER_CACHE_KEY = '__label_zones_ocr_reader__'
    if _READER_CACHE_KEY in sys.modules:
        reader   = sys.modules[_READER_CACHE_KEY]['reader']
        OCR_ENGINE = sys.modules[_READER_CACHE_KEY]['engine']
        print(f"   OCR reader reused from cache ({OCR_ENGINE}). Tiling image...")
    else:
        try:
            from paddleocr import PaddleOCR
            reader = PaddleOCR(lang='vi', use_gpu=True, show_log=False)
            OCR_ENGINE = 'paddle'
            print("   PaddleOCR (vi) loaded. Tiling image...")
        except Exception as e:
            import easyocr
            reader = easyocr.Reader(['vi', 'en'], gpu=False, verbose=False)
            OCR_ENGINE = 'easyocr'
            print(f"   EasyOCR (vi+en) loaded (PaddleOCR failed: {e}). Tiling image...")
        sys.modules[_READER_CACHE_KEY] = {'reader': reader, 'engine': OCR_ENGINE}

    # Bảng màu fill của từng zone (BGR) — dùng để xóa nền trước OCR
    zone_fill_bgr = {z['zone_id']: np.array(z['color_rgb'][::-1], dtype=np.float32)
                     for z in zones}

    def clean_tile(tile_bgr, labels_tile, fill_thresh=80, bg=230):
        """Xóa màu fill của từng zone trong tile → chỉ còn text và ranh giới.
        fill_thresh: ngưỡng L1-distance để coi pixel là 'cùng màu fill'.
        """
        result = tile_bgr.copy().astype(np.float32)
        for zid in np.unique(labels_tile):
            if zid == 0:
                result[labels_tile == 0] = bg   # ngoài ward → trắng
                continue
            if zid not in zone_fill_bgr:
                continue
            fill = zone_fill_bgr[zid]
            zone_px = (labels_tile == zid)
            diff = np.abs(result - fill).sum(axis=2)
            result[(diff < fill_thresh) & zone_px] = bg  # fill → trắng
        return result.astype(np.uint8)

    # Tile config: 700×600 với overlap 120px
    # Text trên bản đồ 150 DPI ≈ 15-20px → cần upscale cao để CRAFT/DBNet detect được
    # UPSCALE=4: text ~15px → 60px — well above DBNet minimum (~20px)
    TILE_W, TILE_H = 700, 600
    OVERLAP = 120
    UPSCALE = 4

    # Sinh tile coords (x0, y0, x1, y1) trên ảnh gốc
    tiles = []
    for y0 in range(0, H, TILE_H - OVERLAP):
        for x0 in range(0, W, TILE_W - OVERLAP):
            x1 = min(x0 + TILE_W, W)
            y1 = min(y0 + TILE_H, H)
            tiles.append((x0, y0, x1, y1))
    print(f"   Tiles: {len(tiles)} (size {TILE_W}×{TILE_H}, overlap {OVERLAP}, upscale {UPSCALE}×)")

    detections = []  # global-coord (bbox, text, conf)
    for i, (x0, y0, x1, y1) in enumerate(tiles):
        crop = img[y0:y1, x0:x1]
        labels_tile = labels[y0:y1, x0:x1]
        # Xóa màu fill → chỉ còn text và ranh giới
        crop_clean = clean_tile(crop, labels_tile)
        if UPSCALE != 1:
            crop_up = cv2.resize(crop_clean, None, fx=UPSCALE, fy=UPSCALE,
                                 interpolation=cv2.INTER_CUBIC)
        else:
            crop_up = crop_clean
        try:
            if OCR_ENGINE == 'paddle':
                tile_dets = []
                for res in reader.predict(crop_up):
                    tile_dets.extend(_paddle_dets(res))
            else:
                tile_dets = reader.readtext(crop_up, detail=1, paragraph=False,
                                            text_threshold=0.3, low_text=0.2)
        except Exception as e:
            print(f"   Tile {i} error: {e}")
            continue

        # Convert local → global
        for bbox, text, conf in tile_dets:
            global_bbox = [
                [bb[0] / UPSCALE + x0, bb[1] / UPSCALE + y0]
                for bb in bbox
            ]
            detections.append((global_bbox, text, conf))

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(tiles)} tiles, total detections: {len(detections)}")

    print(f"   Total raw detections: {len(detections)}")

    # ─── 2b. Tesseract full-image pass ───
    # Thuật toán connected-component + LSTM của Tesseract khác hoàn toàn CRAFT/DBNet.
    # PSM 11 (sparse text) tìm text rải rác trên ảnh — lý tưởng cho bản đồ.
    print("\n2b. Tesseract supplementary pass on full image")
    try:
        import pytesseract
        from collections import defaultdict as _ddict

        # Xóa toàn bộ màu fill zone trên ảnh đầy đủ (1 lần, không tiling)
        img_tc = img.copy().astype(np.float32)
        for zid, fill in zone_fill_bgr.items():
            zone_px = (labels == zid)
            diff = np.abs(img_tc - fill).sum(axis=2)
            img_tc[(diff < 100) & zone_px] = 230
        img_tc[labels == 0] = 230
        gray_tc = cv2.cvtColor(img_tc.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        del img_tc  # giải phóng RAM

        TESS_UP = 2
        gray_up_tc = cv2.resize(gray_tc, None, fx=TESS_UP, fy=TESS_UP,
                                interpolation=cv2.INTER_CUBIC)
        print(f"   Tesseract input: {gray_up_tc.shape[1]}x{gray_up_tc.shape[0]}")

        tess_data = pytesseract.image_to_data(
            gray_up_tc,
            config='--psm 11 -l vie --oem 1',
            output_type=pytesseract.Output.DICT
        )
        del gray_up_tc

        # Nhóm từ theo (block, par, line) để ghép "TỔ" + "65" thành "TỔ 65"
        _line_words = _ddict(list)
        for i in range(len(tess_data['text'])):
            txt = (tess_data['text'][i] or '').strip()
            if not txt:
                continue
            key = (tess_data['block_num'][i], tess_data['par_num'][i],
                   tess_data['line_num'][i])
            _line_words[key].append(i)

        tess_added = 0
        for key, idxs in _line_words.items():
            combined = ' '.join((tess_data['text'][i] or '').strip() for i in idxs)
            m = TO_REGEX.match(combined) or TO_REGEX_LOOSE.search(combined)
            if not m or 'T' not in combined.upper():
                continue
            try:
                _to_num = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if not (1 <= _to_num <= 200):
                continue

            # Bbox union → tọa độ trên ảnh gốc (chia TESS_UP)
            _x0 = min(tess_data['left'][i] for i in idxs)
            _y0 = min(tess_data['top'][i] for i in idxs)
            _x1 = max(tess_data['left'][i] + tess_data['width'][i] for i in idxs)
            _y1 = max(tess_data['top'][i] + tess_data['height'][i] for i in idxs)
            x0o, y0o = _x0 / TESS_UP, _y0 / TESS_UP
            x1o, y1o = _x1 / TESS_UP, _y1 / TESS_UP
            fake_bbox = [[x0o, y0o], [x1o, y0o], [x1o, y1o], [x0o, y1o]]

            avg_conf = float(np.mean([
                max(0.0, float(tess_data['conf'][i])) for i in idxs
            ])) / 100.0

            detections.append((fake_bbox, combined, avg_conf))
            tess_added += 1

        print(f"   Tesseract: {tess_added} candidate lines added to detections pool")

    except ImportError:
        print("   pytesseract not installed. Run:")
        print("     !apt-get install -y tesseract-ocr tesseract-ocr-vie")
        print("     !pip install pytesseract")
    except Exception as e:
        print(f"   Tesseract error: {e}")

    # ─── Dedupe overlap ───
    # Khi 2 detection cách nhau <15px và có cùng text → giữ confidence cao hơn
    deduped = []
    for det in sorted(detections, key=lambda d: -d[2]):  # sort by conf desc
        bbox, text, conf = det
        cx = sum(p[0] for p in bbox) / len(bbox)
        cy = sum(p[1] for p in bbox) / len(bbox)
        # Check trùng với detection đã giữ
        is_dup = False
        for kept in deduped:
            kbb, ktext, _ = kept
            kcx = sum(p[0] for p in kbb) / 4
            kcy = sum(p[1] for p in kbb) / 4
            if abs(cx - kcx) < 20 and abs(cy - kcy) < 20 and \
               text.strip().lower() == ktext.strip().lower():
                is_dup = True
                break
        if not is_dup:
            deduped.append(det)
    print(f"   After dedupe: {len(deduped)}")
    detections = deduped

    # ─── DEBUG: dump detections likely related to TỔ (T + digits) ───
    print("\nDEBUG raw detections with 'T' + digits (conf>=0.2):")
    t_dets = [(bbox, txt, cf) for bbox, txt, cf in detections
              if cf >= 0.2 and 'T' in txt.upper()
              and any(c.isdigit() for c in txt)]
    for bbox, txt, cf in sorted(t_dets, key=lambda d: -d[2])[:60]:
        cx = sum(p[0] for p in bbox) / len(bbox)
        cy = sum(p[1] for p in bbox) / len(bbox)
        print(f"  {cf:.2f}  '{txt}'  @ ({cx:.0f},{cy:.0f})")
    print(f"  (total T+digit detections: {len(t_dets)})")

    # ─── Filter "TỔ XX" patterns ───
    print("\n3. Filter TỔ XX patterns")
    to_detections = []  # (to_num, cx, cy, raw_text, confidence)
    for det in detections:
        bbox, text, conf = det
        # Strip leading/trailing OCR noise (~, ', ", -, .)
        text_clean = re.sub(r"^[\s~'\".\-]+|[\s~'\".\-/:]+$", '', text.strip())
        # Thử cả 2 regex
        m = TO_REGEX.match(text_clean) or TO_REGEX_LOOSE.search(text_clean)
        if not m:
            continue
        # Kiểm tra confidence + có "T" trong text
        if conf < 0.2:
            continue
        if 'T' not in text_clean.upper():
            continue
        try:
            to_num = int(m.group(1))
        except (ValueError, IndexError):
            continue
        if to_num < 1 or to_num > 200:
            continue
        # Centroid của bbox
        bbox_arr = np.array(bbox)
        cx = float(bbox_arr[:, 0].mean())
        cy = float(bbox_arr[:, 1].mean())
        to_detections.append({
            'to_num': to_num,
            'cx': cx, 'cy': cy,
            'text': text_clean,
            'conf': float(conf),
        })

    print(f"   Matched 'TỔ XX' detections: {len(to_detections)}")
    for d in sorted(to_detections, key=lambda x: x['to_num'])[:10]:
        print(f"     TỔ {d['to_num']:>3} @ ({d['cx']:.0f}, {d['cy']:.0f}) conf={d['conf']:.2f} text='{d['text']}'")
    if len(to_detections) > 10:
        print(f"     ... and {len(to_detections) - 10} more")

    # ─── Match detection → zone ───
    print("\n4. Match detections to zones")
    zone_to_dets = defaultdict(list)  # zone_id → list of detections
    unmatched = []
    for det in to_detections:
        x, y = int(det['cx']), int(det['cy'])
        if 0 <= x < W and 0 <= y < H:
            zid = int(labels[y, x])
        else:
            zid = 0

        # Nếu rơi ngoài zone (background hoặc ranh giới) → tìm label phổ biến nhất
        # trong vùng lân cận thực tế (chính xác hơn KDTree centroid)
        if zid == 0:
            r = NEAREST_RADIUS_PX
            x_lo, x_hi = max(0, x - r), min(W, x + r + 1)
            y_lo, y_hi = max(0, y - r), min(H, y + r + 1)
            patch = labels[y_lo:y_hi, x_lo:x_hi]
            nz = patch[patch > 0]
            if len(nz) > 0:
                uniq, cnts = np.unique(nz, return_counts=True)
                zid = int(uniq[np.argmax(cnts)])
            else:
                unmatched.append(det)
                continue

        if zid not in zone_id_to_idx:
            unmatched.append(det)
            continue

        zone_to_dets[zid].append(det)

    print(f"   Detections matched to zone: {sum(len(v) for v in zone_to_dets.values())}")
    print(f"   Unmatched (no nearby zone): {len(unmatched)}")

    # ─── Pass 2: Targeted OCR cho các zone chưa có detection ───
    print("\n4b. Second pass: targeted OCR for unlabeled zones")
    MIN_AREA_PASS2 = 1500   # bỏ qua zone quá nhỏ (không có text)
    MIN_DIM_PASS2  = 300    # min dimension sau upscale để OCR tốt

    unlabeled_p2 = [z for z in zones
                    if z['zone_id'] not in zone_to_dets
                    and z['area_px'] >= MIN_AREA_PASS2]
    print(f"   Zones to retry: {len(unlabeled_p2)}")

    # Lưu 5 zone crop đầu tiên để debug (xem preprocessing trông như thế nào)
    DEBUG_DIR = r'/content/debug_pass2'
    import os; os.makedirs(DEBUG_DIR, exist_ok=True)

    pass2_count = 0
    debug_saved = 0
    for z in unlabeled_p2:
        zid = z['zone_id']
        mask_z = (labels == zid)
        ys_z, xs_z = np.where(mask_z)
        if len(ys_z) == 0:
            continue

        pad = 40
        bx1 = max(0, int(xs_z.min()) - pad)
        by1 = max(0, int(ys_z.min()) - pad)
        bx2 = min(W, int(xs_z.max()) + pad)
        by2 = min(H, int(ys_z.max()) + pad)

        crop = img[by1:by2, bx1:bx2].copy()
        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            continue

        crop_mask = mask_z[by1:by2, bx1:bx2]

        # 1. Pixel ngoài zone → trắng
        crop[~crop_mask.astype(bool)] = 230

        # 2. Xóa màu fill của zone → chỉ còn text (darker pixels)
        fill_bgr = np.array(z['color_rgb'][::-1], dtype=np.float32)
        diff_fill = np.abs(crop.astype(np.float32) - fill_bgr).sum(axis=2)
        crop[(diff_fill < 80) & crop_mask.astype(bool)] = 230

        # Upscale để text (~15% của min_dim) đạt ít nhất ~60px — OCR cần ít nhất 20-30px
        # Target min_dim ≥ 400: zone 150px → scale 2.7×, zone 400px → scale 1.0 (no-op)
        _TARGET_DIM = 400
        scale = max(1.0, _TARGET_DIM / min(ch, cw))
        scale = min(scale, 6.0)  # cap để tránh OOM trên zone nhỏ bất thường
        if scale > 1.01:
            crop_proc = cv2.resize(crop, (int(cw * scale), int(ch * scale)),
                                   interpolation=cv2.INTER_CUBIC)
        else:
            crop_proc = crop

        # Debug: lưu ảnh cho 5 zone đầu tiên được xử lý
        if debug_saved < 5:
            cv2.imwrite(f'{DEBUG_DIR}/zone_{zid}_masked.png', crop)
            cv2.imwrite(f'{DEBUG_DIR}/zone_{zid}_proc.png', crop_proc)
            debug_saved += 1

        try:
            if OCR_ENGINE == 'paddle':
                p2_dets = []
                for res in reader.predict(crop_proc):
                    p2_dets.extend(_paddle_dets(res))
            else:
                canvas_p2 = max(300, 2 * max(ch, cw))
                p2_dets = reader.readtext(crop_proc, detail=1, paragraph=False,
                                          text_threshold=0.2, low_text=0.15,
                                          canvas_size=canvas_p2)
        except Exception:
            continue

        t_hits = [(t, c) for _, t, c in p2_dets if 'T' in t.upper() and any(d.isdigit() for d in t)]
        if t_hits:
            print(f"   [P2 zone {zid}] T+digit: "
                  + " | ".join(f"'{t}' {c:.2f}" for t, c in t_hits[:4]))

        for bbox, text, conf in p2_dets:
            text_clean = re.sub(r"^[\s~'\".\-]+|[\s~'\".\-/:]+$", '', text.strip())
            m = TO_REGEX.match(text_clean) or TO_REGEX_LOOSE.search(text_clean)
            if not m or conf < 0.15 or 'T' not in text_clean.upper():
                continue
            try:
                to_num = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if not (1 <= to_num <= 200):
                continue

            # Convert bbox center về tọa độ ảnh gốc
            bbox_arr = np.array(bbox)
            cx_g = float(bbox_arr[:, 0].mean()) / scale + bx1
            cy_g = float(bbox_arr[:, 1].mean()) / scale + by1
            lx, ly = int(cx_g), int(cy_g)

            # Nhận nếu center trong zone hoặc trong vùng lân cận nhỏ (text có thể
            # nằm gần biên zone, center rơi vào zone kề)
            if 0 <= ly < H and 0 <= lx < W:
                pz = int(labels[ly, lx])
                if pz not in (0, zid):
                    # Tìm xem zone zid có pixel nào gần center không
                    r_near = 30
                    x_lo = max(0, lx - r_near); x_hi = min(W, lx + r_near + 1)
                    y_lo = max(0, ly - r_near); y_hi = min(H, ly + r_near + 1)
                    if not np.any(labels[y_lo:y_hi, x_lo:x_hi] == zid):
                        continue  # center quá xa zone → bỏ qua

            zone_to_dets[zid].append({
                'to_num': to_num, 'cx': cx_g, 'cy': cy_g,
                'text': text_clean, 'conf': float(conf),
            })
            pass2_count += 1
            break  # lấy detection đầu tiên hợp lệ cho zone này

    print(f"   Pass 2 new labels: {pass2_count}")

    # ─── Resolve conflicts: SPLIT zone thành nhiều sub-zone bằng Voronoi ───
    print("\n5. Resolve conflicts via Voronoi split")
    zone_to_num = {}              # zone_id (có thể mới) → detection
    new_zones = []                # zones list mới (giữ + thêm split)
    next_zid = max(z['zone_id'] for z in zones) + 1
    split_count = 0
    new_zone_count = 0

    # Chuẩn bị: index zone gốc theo zone_id để giữ thông tin gốc
    orig_zone_by_id = {z['zone_id']: z for z in zones}

    for zid, dets in zone_to_dets.items():
        if zid not in orig_zone_by_id:
            continue
        if len(dets) == 1:
            zone_to_num[zid] = dets[0]
            new_zones.append(orig_zone_by_id[zid])
            continue

        # ── SPLIT ── nhiều detection → Voronoi
        split_count += 1
        nums = [d['to_num'] for d in dets]
        print(f"   Zone {zid} có {len(dets)} labels {nums} → split")

        zone_mask = (labels == zid)
        ys, xs = np.where(zone_mask)
        if len(ys) == 0:
            continue
        seed_pts = np.array([[d['cx'], d['cy']] for d in dets], dtype=np.float32)
        pixel_pts = np.stack([xs, ys], axis=1).astype(np.float32)
        # Distance pixel → mỗi seed (vectorized, batched để tiết kiệm RAM)
        nearest = np.zeros(len(pixel_pts), dtype=np.int32)
        BATCH = 50000
        for s in range(0, len(pixel_pts), BATCH):
            e = min(s + BATCH, len(pixel_pts))
            d2 = np.sum((seed_pts[None, :, :] - pixel_pts[s:e, None, :]) ** 2, axis=2)
            nearest[s:e] = np.argmin(d2, axis=1)

        # Tạo sub-zone cho mỗi seed
        for k, det in enumerate(dets):
            sub_mask_idx = (nearest == k)
            if sub_mask_idx.sum() < 600:
                # Sub-region quá nhỏ → bỏ qua (merge vào sub khác)
                continue
            sub_ys = ys[sub_mask_idx]
            sub_xs = xs[sub_mask_idx]
            # Cập nhật labels: pixel này thuộc zone mới
            new_zid = next_zid
            next_zid += 1
            labels[sub_ys, sub_xs] = new_zid

            # Tính lại centroid + bbox + dominant color cho sub-zone
            sub_area = int(len(sub_ys))
            sub_cx = int(np.mean(sub_xs))
            sub_cy = int(np.mean(sub_ys))
            sub_x0, sub_y0 = int(sub_xs.min()), int(sub_ys.min())
            sub_x1, sub_y1 = int(sub_xs.max()), int(sub_ys.max())
            # Lấy dominant color và HSV stats
            sub_pix = img[sub_ys, sub_xs]
            sub_pix_hsv = cv2.cvtColor(sub_pix.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            avg_sat = float(np.mean(sub_pix_hsv[:, 1]))
            color_std = float(np.mean(np.std(sub_pix.astype(np.float32), axis=0)))
            pq = (sub_pix // 16) * 16
            cu, cc = np.unique(pq.reshape(-1, 3), axis=0, return_counts=True)
            dom = cu[np.argmax(cc)]
            dom_rgb = [int(dom[2]), int(dom[1]), int(dom[0])]

            # Compute geo coords từ alignment params (lấy từ zone gốc)
            orig = orig_zone_by_id[zid]
            # Scale geo bằng tỉ lệ centroid
            ocx, ocy = orig['centroid_px']
            ogx = orig['centroid_geo']['lon']
            ogy = orig['centroid_geo']['lat']
            # Cần alignment params chính xác — đọc từ file
            from json import load as _jload
            try:
                with open(r'/content/alignment_params.json', 'r') as f:
                    align = _jload(f)
                lon_c = align['lon0'] + sub_cx * align['scale_x']
                lat_c = align['lat0'] - sub_cy * align['scale_y']
            except Exception:
                # Fallback: linear interp từ centroid gốc
                # (không chính xác nếu zone xa centroid)
                lon_c = ogx
                lat_c = ogy

            new_zone = {
                'zone_id': new_zid,
                'area_px': sub_area,
                'centroid_px': [sub_cx, sub_cy],
                'centroid_geo': {'lon': float(lon_c), 'lat': float(lat_c)},
                'bbox_px': [sub_x0, sub_y0, sub_x1, sub_y1],
                'color_rgb': dom_rgb,
                'avg_saturation': round(avg_sat, 1),
                'color_std': round(color_std, 1),
                'split_from': zid,
            }
            new_zones.append(new_zone)
            zone_to_num[new_zid] = det
            new_zone_count += 1

    # Thêm các zone không có detection (giữ nguyên, chưa label)
    labeled_zids = set(zone_to_dets.keys())
    for z in zones:
        if z['zone_id'] not in labeled_zids:
            new_zones.append(z)

    # Cập nhật zones list
    zones = new_zones

    # Update zone_id_to_idx & centroids cho subsequent code
    zone_id_to_idx = {z['zone_id']: i for i, z in enumerate(zones)}

    print(f"   Zones split: {split_count} → {new_zone_count} new sub-zones")
    print(f"   Total zones now: {len(zones)}")
    print(f"   Zones labeled: {len(zone_to_num)}")
    print(f"   Zones WITHOUT label: {len(zones) - len(zone_to_num)}")

    # ─── Step 5b: Adjacency-based label propagation ───
    # BFS từ zone có label → khuếch tán sang zone KỀ NHAU + CÙNG MÀU.
    # Giải quyết "lô rời cùng tổ": cùng màu, không kề label zone, nhưng kề qua chain.
    # An toàn hơn global color propagation: dừng tại ranh giới màu (tổ khác màu khác).
    # Không propagate giữa Voronoi siblings (cùng parent zone = tổ khác nhau).
    print("\n5b. Adjacency-based label propagation")

    # Build adjacency graph từ labels array hiện tại (sau Voronoi split)
    _h_diff = labels[:, :-1] != labels[:, 1:]
    _yb, _xb = np.where(_h_diff)
    _ph = np.stack([labels[_yb, _xb].astype(np.int64),
                    labels[_yb, _xb + 1].astype(np.int64)], axis=1)

    _v_diff = labels[:-1, :] != labels[1:, :]
    _yb2, _xb2 = np.where(_v_diff)
    _pv = np.stack([labels[_yb2, _xb2].astype(np.int64),
                    labels[_yb2 + 1, _xb2].astype(np.int64)], axis=1)

    _all_p = np.vstack([_ph, _pv])
    _valid_p = (_all_p[:, 0] > 0) & (_all_p[:, 1] > 0)
    _sp = np.sort(_all_p[_valid_p], axis=1)
    _sp_uniq = np.unique(_sp, axis=0)

    _adj_map = defaultdict(set)
    for _za, _zb in _sp_uniq:
        _adj_map[int(_za)].add(int(_zb))
        _adj_map[int(_zb)].add(int(_za))
    print(f"   Adjacency: {len(_adj_map)} zones, {len(_sp_uniq)} pairs")

    # Voronoi siblings: sub-zones từ cùng parent → KHÔNG propagate lẫn nhau
    _zone_parent = {z['zone_id']: z.get('split_from') for z in zones}

    # Color lookup
    _zcolor = {z['zone_id']: tuple(z.get('color_rgb', [])) for z in zones}

    # BFS từ tất cả zone đã có label
    from collections import deque as _Deque
    _queue_bfs = _Deque(zone_to_num.keys())
    _visited_bfs = set(zone_to_num.keys())
    _prop_count = 0

    while _queue_bfs:
        _src = _queue_bfs.popleft()
        _src_det = zone_to_num[_src]
        _src_color = _zcolor.get(_src, ())
        _src_parent = _zone_parent.get(_src)

        for _nb in _adj_map.get(_src, set()):
            if _nb in _visited_bfs:
                continue
            # Bỏ qua Voronoi siblings (cùng parent = tổ khác)
            if _src_parent and _src_parent == _zone_parent.get(_nb):
                continue
            # Chỉ propagate sang zone CÙNG MÀU
            _nb_color = _zcolor.get(_nb, ())
            if _src_color and _nb_color:
                # Màu được quantize về bội 16 → exact match; dùng tol=30 cho robust
                _cdist = sum(abs(int(a) - int(b))
                             for a, b in zip(_src_color[:3], _nb_color[:3]))
                if _cdist > 30:
                    continue
            # Propagate
            _nd = _src_det.copy()
            _nd['conf'] = 0.0
            _nd['text'] = f"[adj:{_src_det.get('text', '')}]"
            zone_to_num[_nb] = _nd
            _visited_bfs.add(_nb)
            _queue_bfs.append(_nb)
            _prop_count += 1

    print(f"   Propagated via adjacency+color: +{_prop_count} zones")
    print(f"   Total after propagation: {len(zone_to_num)}/{len(zones)} "
          f"({len(zone_to_num) * 100 // len(zones)}%)")

    # ─── Enrich zones JSON ───
    print("\n6. Enrich zones with TỔ numbers")
    for z in zones:
        zid = z['zone_id']
        if zid in zone_to_num:
            d = zone_to_num[zid]
            z['to_so'] = d['to_num']
            z['ocr_text'] = d['text']
            z['ocr_confidence'] = round(d['conf'], 2)
        else:
            z['to_so'] = None

    # Detect duplicate TỔ numbers across zones (cùng số tổ → có thể merge)
    to_num_to_zones = defaultdict(list)
    for z in zones:
        if z['to_so'] is not None:
            to_num_to_zones[z['to_so']].append(z['zone_id'])
    dup_nums = {k: v for k, v in to_num_to_zones.items() if len(v) > 1}
    if dup_nums:
        print(f"\n   ⚠ TỔ numbers gán cho >1 zone (có thể cần merge):")
        for n, zs in sorted(dup_nums.items()):
            print(f"     TỔ {n}: zones {zs}")

    # ─── Export ───
    output = {
        'metadata': {
            **zones_data['metadata'],
            'ocr_engine': OCR_ENGINE,
            'ocr_detections': len(to_detections),
            'zones_with_label': len(zone_to_num),
            'color_propagated': _prop_count,
            'duplicate_to_numbers': len(dup_nums),
        },
        'summary': {
            **zones_data['summary'],
            'total_zones': len(zones),
            'labeled_zones': len(zone_to_num),
        },
        'zones': zones,
    }
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n   Saved: {OUTPUT_JSON}")

    # Save updated labels (after split)
    np.save(r'/content/v5_labels_split.npy', labels)
    print(f"   Saved: v5_labels_split.npy")

    # ─── Visualization ───
    print("\n7. Visualization")
    vis = img.copy()
    # Draw zone centroids + TỔ number
    font = cv2.FONT_HERSHEY_SIMPLEX
    for z in zones:
        cx, cy = z['centroid_px']
        if z.get('to_so') is not None:
            color = (0, 200, 0)  # green if labeled
            text = f"T{z['to_so']}"
        else:
            color = (0, 0, 255)  # red if unlabeled
            text = "?"
        cv2.circle(vis, (cx, cy), 5, color, -1)
        cv2.putText(vis, text, (cx - 15, cy - 8), font, 0.5, color, 2)

    # Draw OCR detection boxes
    for d in to_detections:
        cx, cy = int(d['cx']), int(d['cy'])
        cv2.circle(vis, (cx, cy), 3, (255, 0, 255), -1)

    cv2.imwrite(OUTPUT_VIS, vis)
    print(f"   Saved: {OUTPUT_VIS}")
    print(f"\nDONE: {len(zone_to_num)}/{len(zones)} zones labeled "
          f"({len(zone_to_num) * 100 / len(zones):.1f}%)")


if __name__ == '__main__':
    main()
