# -*- coding: utf-8 -*-
"""
Pipeline v1 — Boundary-line detection + Flood-fill segmentation (Ensemble)

Triết lý: chạy 4 cấu hình boundary detection khác nhau → union ranh giới →
flood-fill một lần duy nhất → bao trọn mọi trường hợp.

Config A (balanced)   : ranh giới chuẩn, cân bằng
Config B (faint_lines): đường biên nhạt / mực mờ
Config C (dark_heavy) : đường đen rõ + đóng gap lớn
Config D (color_focus): ranh giới = thay đổi màu, không cần dark line
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import json
from collections import defaultdict
from functools import reduce

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ── Paths ──────────────────────────────────────────────────────────────────
DPI          = 300
PDF_IMG      = f'd:\\Programming\\Python\\SON_TRA\\map_{DPI}dpi.png'
GEOJSON_PATH = r'd:\Programming\Python\SON_TRA\phuong_son_tra.geojson'
ALIGN_PARAMS = r'd:\Programming\Python\SON_TRA\alignment_params.json'
OUT_DIR      = r'd:\Programming\Python\SON_TRA\results\v1_dpi_300'

# ── Ensemble boundary configs ───────────────────────────────────────────────
# Mỗi config phát hiện một loại ranh giới khác nhau → union → finer zones
CONFIGS = [
    # A: Balanced — baseline cân bằng
    dict(name='balanced',
         DARK_THRESH=50,  MIN_DARK_AREA=200,
         CANNY_LOW=10,  CANNY_HIGH=50,
         CANNY_AB_LOW=15, CANNY_AB_HIGH=50,
         CLOSE_K=3, CLOSE_ITERS=2),

    # B: Faint lines — đường biên nhạt, mực mờ
    # DARK_THRESH thấp hơn bắt được mực không đủ tối
    # Canny ngưỡng thấp bắt chuyển màu nhẹ giữa zone liền kề
    # dict(name='faint_lines',
    #      DARK_THRESH=45,  MIN_DARK_AREA=200,
    #      CANNY_LOW=5,   CANNY_HIGH=25,
    #      CANNY_AB_LOW=8,  CANNY_AB_HIGH=30,
    #      CLOSE_K=3, CLOSE_ITERS=2),

    # # C: Dark heavy — đường đen rõ nét + đóng gap lớn
    # # DARK_THRESH cao hơn bắt thêm dark pixels xung quanh nét mực
    # # MIN_DARK_AREA nhỏ giữ cả đoạn biên mảnh
    # # CLOSE mạnh hơn để bịt khe hở trong đường biên
    # dict(name='dark_heavy',
    #      DARK_THRESH=80,  MIN_DARK_AREA=150,
    #      CANNY_LOW=20,  CANNY_HIGH=80,
    #      CANNY_AB_LOW=25, CANNY_AB_HIGH=80,
    #      CLOSE_K=5, CLOSE_ITERS=3),

    # # D: Color focus — ranh giới = thay đổi màu, không dựa dark line
    # # CLOSE_ITERS=1: ít đóng → giữ các ranh giới màu mỏng
    # # Canny AB nhạy hơn: bắt chuyển đổi sắc a,b kể cả khi L không thay đổi
    # dict(name='color_focus',
    #      DARK_THRESH=55,  MIN_DARK_AREA=500,
    #      CANNY_LOW=8,   CANNY_HIGH=35,
    #      CANNY_AB_LOW=10, CANNY_AB_HIGH=38,
    #      CLOSE_K=3, CLOSE_ITERS=1),
]

# ── Zone filter params (chia sẻ cho tất cả configs) ────────────────────────
_dpi_k         = (DPI / 150) ** 2        # 1× @ 150, 4× @ 300, 9× @ 450
MIN_ZONE_AREA  = int(200     * _dpi_k)
MAX_ZONE_AREA  = int(200_000 * _dpi_k)
MIN_AVG_SAT    = 6       # HSV saturation — quá thấp = nước/xám (12→6: cho phép dark gray zones)
MAX_COMPACT    = 50      # perimeter²/(4π·area) — quá cao = đường mảnh
MIN_FILL_RATIO = 0.10    # area/bbox — quá thấp = dải đường ngoằn ngoèo

# ── Post-merge 2 tầng ────────────────────────────────────────────────────
N_COLOR_CLUSTERS  = 1
ADJ_INTRA_THRESH  = 5.0    # LAB dist tối đa để merge liền kề (18→10: chặt hơn)
PROXIMITY_GAP_PX  = int(50 * (DPI / 150))
PROXIMITY_LAB     = 4.0     # LAB dist tối đa để merge lô cách đường (10→6: chặt hơn)

# Phase 1 guard: nếu ≥ X% interface là dark pixels → boundary thật → không merge
# Ý nghĩa: đường đen in giữa 2 zone = ý đồ của bản đồ, không được gộp dù cùng màu
DARK_BOUNDARY_RATIO = 0.2 # 15% interface có dark pixel → giữ nguyên

# ── Alignment ───────────────────────────────────────────────────────────────
with open(ALIGN_PARAMS, encoding='utf-8') as f:
    _ap = json.load(f)
ALIGN_DPI  = _ap.get('dpi', 150)
DPI_FACTOR = ALIGN_DPI / DPI
LON0    = _ap['lon0']
LAT0    = _ap['lat0']
SCALE_X = _ap['scale_x'] * DPI_FACTOR
SCALE_Y = _ap['scale_y'] * DPI_FACTOR
print(f"Alignment: lon0={LON0:.8f} lat0={LAT0:.8f}  ({ALIGN_DPI}→{DPI} DPI)")

def geo_to_px(lon, lat):
    return int(round((lon - LON0) / SCALE_X)), int(round((LAT0 - lat) / SCALE_Y))

def px_to_geo(px, py):
    return round(LON0 + px * SCALE_X, 7), round(LAT0 - py * SCALE_Y, 7)

# ── Hàm boundary detection cho 1 config ────────────────────────────────────
def run_boundary(cfg, lab_clean, gray_orig, ward_mask, H, W):
    """Trả về (boundary_mask, dark_mask) — cả 2 đều là uint8 255/0."""
    # Gradient boundary: Canny trên LAB
    canny_l = cv2.Canny(lab_clean[:, :, 0], cfg['CANNY_LOW'],    cfg['CANNY_HIGH'])
    canny_a = cv2.Canny(lab_clean[:, :, 1], cfg['CANNY_AB_LOW'], cfg['CANNY_AB_HIGH'])
    canny_b = cv2.Canny(lab_clean[:, :, 2], cfg['CANNY_AB_LOW'], cfg['CANNY_AB_HIGH'])
    grad = cv2.bitwise_or(canny_l, cv2.bitwise_or(canny_a, canny_b))
    grad = cv2.bitwise_and(grad, ward_mask)

    # Trong run_boundary(), thêm channel HSV value:
    hsv = cv2.cvtColor(img_clean, cv2.COLOR_BGR2HSV)
    canny_v = cv2.Canny(hsv[:, :, 2], cfg['CANNY_LOW'], cfg['CANNY_HIGH'])
    grad = cv2.bitwise_or(grad, canny_v)  # union với LAB gradient

    # Dark-line boundary: phân biệt boundary line vs dark zone interior
    #
    # Vấn đề: dark zone bị text trắng + hatching bên trong PHÁ VỠ thành nhiều blob nhỏ
    # → check shape từng blob riêng lẻ không hiệu quả.
    #
    # Giải pháp: CLOSE lớn để bridge qua text/hatching → dark zone thành 1 blob lớn →
    # check compact/size trong không gian đã đóng → loại blob gốc thuộc dark zone candidate.
    _dz_close  = int(28 * (DPI / 150))            # kernel bridge text (~56px at 300dpi)
    _dz_area   = int(15_000 * (DPI / 150) ** 2)   # diện tích tối thiểu dark zone (post-close)
    _dz_aspect = 5                                  # aspect ratio tối đa để coi là compact

    dark_raw = ((gray_orig < cfg['DARK_THRESH']) & (ward_mask == 255)).astype(np.uint8) * 255

    # Bước 1: tìm dark zone candidates trong không gian đã đóng
    k_dz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_dz_close, _dz_close))
    dark_closed = cv2.morphologyEx(dark_raw, cv2.MORPH_CLOSE, k_dz)
    n_dz, lbl_dz, sts_dz, _ = cv2.connectedComponentsWithStats(dark_closed, connectivity=4)
    dark_zone_map = np.zeros((H, W), dtype=np.uint8)   # 255 = thuộc dark zone
    for i in range(1, n_dz):
        area_c = int(sts_dz[i, cv2.CC_STAT_AREA])
        if area_c < _dz_area:
            continue
        wb = int(sts_dz[i, cv2.CC_STAT_WIDTH])
        hb = int(sts_dz[i, cv2.CC_STAT_HEIGHT])
        if max(wb, hb) / max(min(wb, hb), 1) >= _dz_aspect:
            continue   # elongated → boundary line, không phải dark zone
        dark_zone_map[lbl_dz == i] = 255

    # Bước 2: xử lý từng dark blob gốc — bỏ qua nếu thuộc dark zone candidate
    n_dk, lbl_dk, sts_dk, _ = cv2.connectedComponentsWithStats(dark_raw, connectivity=4)
    dark = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, n_dk):
        area = int(sts_dk[i, cv2.CC_STAT_AREA])
        if area < cfg['MIN_DARK_AREA']:
            continue   # text/noise → bỏ
        # Kiểm tra centroid: nằm trong dark zone candidate → không phải ranh giới
        cx = int(sts_dk[i, cv2.CC_STAT_LEFT] + sts_dk[i, cv2.CC_STAT_WIDTH]  / 2)
        cy = int(sts_dk[i, cv2.CC_STAT_TOP]  + sts_dk[i, cv2.CC_STAT_HEIGHT] / 2)
        if dark_zone_map[cy, cx] > 0:
            continue   # thuộc dark zone → giữ để flood-fill, không đánh boundary
        dark[lbl_dk == i] = 255

    # Kết hợp + đóng khe hở
    combined = cv2.bitwise_or(grad, dark)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg['CLOSE_K'], cfg['CLOSE_K']))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=cfg['CLOSE_ITERS'])
    combined = cv2.dilate(combined, k, iterations=1)

    return combined, dark   # trả về cả dark mask để dùng trong merge validation

# ─────────────────────────────────────────────────────────────
print("\n1. Load data")
# ─────────────────────────────────────────────────────────────
img = cv2.imread(PDF_IMG)
assert img is not None, f"Không đọc được ảnh: {PDF_IMG}"
H, W = img.shape[:2]
print(f"   {W}×{H}")

with open(GEOJSON_PATH, encoding='utf-8') as f:
    gj = json.load(f)

# ─────────────────────────────────────────────────────────────
print("\n2. Ward mask + loại vùng nước")
# ─────────────────────────────────────────────────────────────
ward_mask_raw = np.zeros((H, W), dtype=np.uint8)
for feat in gj['features']:
    geom = feat['geometry']
    polys = (geom['coordinates'] if geom['type'] == 'MultiPolygon'
             else [geom['coordinates']])
    for poly in polys:
        for ring in poly:
            pts = np.array([geo_to_px(c[0], c[1]) for c in ring], dtype=np.int32)
            cv2.fillPoly(ward_mask_raw, [pts], 255)

smooth_pre = cv2.bilateralFilter(img, d=11, sigmaColor=80, sigmaSpace=80)
hsv_pre    = cv2.cvtColor(smooth_pre, cv2.COLOR_BGR2HSV)
sat_ch     = hsv_pre[:, :, 1]
val_ch     = hsv_pre[:, :, 2]
hue_ch     = hsv_pre[:, :, 0]

water_mask = np.zeros((H, W), dtype=np.uint8)

ls_in_ward = cv2.bitwise_and((sat_ch < 20).astype(np.uint8) * 255, ward_mask_raw)
n, lbl, sts, _ = cv2.connectedComponentsWithStats(ls_in_ward, connectivity=4)
for i in range(1, n):
    if sts[i, cv2.CC_STAT_AREA] > 5000:
        water_mask[lbl == i] = 255

db_in_ward = cv2.bitwise_and(
    ((val_ch < 120) & (sat_ch < 30)).astype(np.uint8) * 255, ward_mask_raw)
n, lbl, sts, _ = cv2.connectedComponentsWithStats(db_in_ward, connectivity=4)
for i in range(1, n):
    if sts[i, cv2.CC_STAT_AREA] > 3000:
        water_mask[lbl == i] = 255

ward_boundary = cv2.morphologyEx(ward_mask_raw, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
cyan_mask = cv2.bitwise_and(
    ((hue_ch >= 80) & (hue_ch <= 130) & (sat_ch >= 20)
     & (sat_ch <= 150) & (val_ch >= 140)).astype(np.uint8) * 255,
    ward_mask_raw)
n, lbl, sts, _ = cv2.connectedComponentsWithStats(cyan_mask, connectivity=4)
for i in range(1, n):
    if sts[i, cv2.CC_STAT_AREA] >= 30000:
        comp = (lbl == i)
        if (comp & (ward_boundary > 0)).any():
            water_mask[comp] = 255

k_water    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
water_mask = cv2.dilate(water_mask, k_water)
ward_mask  = cv2.bitwise_and(ward_mask_raw, cv2.bitwise_not(water_mask))
ward_mask[:120, :]    = 0
ward_mask[H - 80:, :] = 0
print(f"   Ward mask: {ward_mask.sum() // 255:,} px  |  loại nước: {water_mask.sum() // 255:,} px")

# ─────────────────────────────────────────────────────────────
print("\n3. Pre-process (làm sạch text/hatching)")
# ─────────────────────────────────────────────────────────────
k_clean   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_clean)
img_med    = cv2.medianBlur(img_closed, 7)
img_clean  = cv2.bilateralFilter(img_med, d=9, sigmaColor=45, sigmaSpace=45)
cv2.imwrite(f'{OUT_DIR}\\v1_img_clean.png', img_clean)

gray_orig = cv2.cvtColor(img,       cv2.COLOR_BGR2GRAY)
lab_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
print("   Saved: v1_img_clean.png")

# ─────────────────────────────────────────────────────────────
print("\n4. Ensemble boundary detection")
# ─────────────────────────────────────────────────────────────
all_boundaries = []
all_dark_maps  = []
for cfg in CONFIGS:
    b, dk = run_boundary(cfg, lab_clean, gray_orig, ward_mask, H, W)
    all_boundaries.append(b)
    all_dark_maps.append(dk)
    print(f"   [{cfg['name']:12s}] boundary px: {b.sum() // 255:,}")
    cv2.imwrite(f'{OUT_DIR}\\v1_boundary_{cfg["name"]}.png', b)

# Union dark maps → dùng trong Phase 1 merge validation
dark_map = reduce(cv2.bitwise_or, all_dark_maps)

# Union: pixel = ranh giới nếu ít nhất 1 config phát hiện
boundary_union = reduce(cv2.bitwise_or, all_boundaries)
print(f"   [union        ] boundary px: {boundary_union.sum() // 255:,}")
cv2.imwrite(f'{OUT_DIR}\\v1_boundary_union.png', boundary_union)

# Consensus (tham khảo): pixel = ranh giới nếu >= 2 configs đồng thuận
boundary_votes    = sum((b > 0).astype(np.int16) for b in all_boundaries)
boundary_consensus = (boundary_votes >= 2).astype(np.uint8) * 255
print(f"   [consensus ≥2 ] boundary px: {boundary_consensus.sum() // 255:,}")
cv2.imwrite(f'{OUT_DIR}\\v1_boundary_consensus.png', boundary_consensus)

# ─────────────────────────────────────────────────────────────
print("\n5. Flood-fill (union boundary)")
# ─────────────────────────────────────────────────────────────
fill_region = np.zeros((H, W), dtype=np.uint8)
fill_region[(boundary_union == 0) & (ward_mask == 255)] = 255
cv2.imwrite(f'{OUT_DIR}\\v1_fill_region.png', fill_region)
n_raw, labels_raw, stats_raw, centroids_raw = cv2.connectedComponentsWithStats(
    fill_region, connectivity=4)
print(f"   Raw flood-fill regions: {n_raw - 1}")
print("   Saved: v1_fill_region.png, v1_boundary_*.png")

# ─────────────────────────────────────────────────────────────
print("\n6. Lọc zones (area / shape / saturation)")
# ─────────────────────────────────────────────────────────────
hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

zones        = []
skip_area    = 0
skip_fill    = 0
skip_compact = 0
skip_sat     = 0

for i in range(1, n_raw):
    area = int(stats_raw[i, cv2.CC_STAT_AREA])
    if area < MIN_ZONE_AREA or area > MAX_ZONE_AREA:
        skip_area += 1
        continue

    x0 = int(stats_raw[i, cv2.CC_STAT_LEFT])
    y0 = int(stats_raw[i, cv2.CC_STAT_TOP])
    wb = int(stats_raw[i, cv2.CC_STAT_WIDTH])
    hb = int(stats_raw[i, cv2.CC_STAT_HEIGHT])
    x1, y1 = x0 + wb, y0 + hb
    cx = int(centroids_raw[i, 0])
    cy = int(centroids_raw[i, 1])

    fill_ratio = area / max(wb * hb, 1)
    if fill_ratio < MIN_FILL_RATIO and area > 2000:
        skip_fill += 1
        continue

    sub_mask = (labels_raw[y0:y1, x0:x1] == i).astype(np.uint8)
    contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and area > 1500:
        perim = sum(cv2.arcLength(c, True) for c in contours)
        if (perim * perim) / (4 * np.pi * area + 1e-6) > MAX_COMPACT:
            skip_compact += 1
            continue

    pix_hsv = hsv_full[y0:y1, x0:x1][sub_mask == 1]
    if len(pix_hsv) == 0:
        continue
    avg_sat = float(np.mean(pix_hsv[:, 1]))
    if avg_sat < MIN_AVG_SAT:
        skip_sat += 1
        continue

    pix_bgr = img[y0:y1, x0:x1][sub_mask == 1]
    pq = (pix_bgr // 16) * 16
    cu, cc = np.unique(pq.reshape(-1, 3), axis=0, return_counts=True)
    dom = cu[np.argmax(cc)]
    dom_rgb = [int(dom[2]), int(dom[1]), int(dom[0])]

    pix_lab = lab_full[y0:y1, x0:x1][sub_mask == 1]
    mean_lab = pix_lab.mean(axis=0).tolist()

    lon_c, lat_c = px_to_geo(cx, cy)
    zones.append({
        'zone_id':        int(i),
        'area_px':        area,
        'centroid_px':    [cx, cy],
        'centroid_geo':   {'lon': lon_c, 'lat': lat_c},
        'bbox_px':        [x0, y0, x1, y1],
        'color_rgb':      dom_rgb,
        'avg_saturation': round(avg_sat, 1),
        '_mean_lab':      mean_lab,
    })

zones.sort(key=lambda z: z['area_px'], reverse=True)
print(f"   Valid zones        : {len(zones)}")
print(f"   Skipped area       : {skip_area}")
print(f"   Skipped fill_ratio : {skip_fill}   (< {MIN_FILL_RATIO})")
print(f"   Skipped compactness: {skip_compact}  (> {MAX_COMPACT})")
print(f"   Skipped saturation : {skip_sat}   (< {MIN_AVG_SAT})")

# ─────────────────────────────────────────────────────────────
print("\n7. Post-merge 2 tầng")
# ─────────────────────────────────────────────────────────────
n_before_merge = len(zones)
zone_ids_list  = [z['zone_id'] for z in zones]
valid_ids      = set(zone_ids_list)
zone_lab       = {z['zone_id']: np.array(z['_mean_lab']) for z in zones}
zone_bbox      = {z['zone_id']: z['bbox_px']              for z in zones}

parent = {zid: zid for zid in valid_ids}

def _find(x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root

def _union(a, b):
    pa, pb = _find(a), _find(b)
    if pa != pb:
        parent[pb] = pa
        return True
    return False

# 7a. K-means cluster màu trên CÁC ZONE ĐÃ SEGMENT
colors_arr = np.array([zone_lab[zid] for zid in zone_ids_list], dtype=np.float32)
k = min(N_COLOR_CLUSTERS, len(zone_ids_list))
criteria_km = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
_, km_labels, km_centers = cv2.kmeans(
    colors_arr, k, None, criteria_km, 5, cv2.KMEANS_PP_CENTERS)
km_labels_flat = km_labels.flatten()
zone_cluster = {zid: int(km_labels_flat[i]) for i, zid in enumerate(zone_ids_list)}
print(f"   K-means: {k} clusters trên {len(zone_ids_list)} zones  "
      f"(center spread: {float(np.std(km_centers)):.1f} LAB)")

# 7b. Tầng 1: liền kề + cùng color cluster + KHÔNG có đường đen giữa
#
# Vấn đề: flood-fill để 0-pixels (ranh giới) giữa các zone → zone không bao giờ
# trực tiếp liền kề nhau trong labels_raw → direct adjacency check luôn = 0 pairs.
#
# Giải pháp: forward-fill label zone vào boundary pixels → mỗi boundary pixel
# biết zone nào ở bên trái/phải/trên/dưới nó → dễ tìm adjacent pairs.
lbl = labels_raw.astype(np.int32)

def _fwd_fill_h(arr):
    """Lan truyền label zone từ trái sang phải qua boundary pixels (=0)."""
    idx = np.where(arr != 0, np.arange(arr.shape[1]), 0).astype(np.int32)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(arr.shape[0])[:, None], idx]

def _fwd_fill_v(arr):
    """Lan truyền label zone từ trên xuống dưới qua boundary pixels (=0)."""
    idx = np.where(arr != 0, np.arange(arr.shape[0])[:, None], 0).astype(np.int32)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return arr[idx, np.arange(arr.shape[1])[None, :]]

lbl_lr = _fwd_fill_h(lbl)                      # zone bên trái tại mỗi pixel
lbl_rl = _fwd_fill_h(lbl[:, ::-1])[:, ::-1]   # zone bên phải
lbl_tb = _fwd_fill_v(lbl)                      # zone phía trên
lbl_bt = _fwd_fill_v(lbl[::-1, :])[::-1, :]   # zone phía dưới

# Boundary pixels trong ward: nơi 2 zone khác nhau gặp nhau
bnd = (lbl == 0) & (ward_mask == 255)
h_bnd = bnd & (lbl_lr != 0) & (lbl_rl != 0) & (lbl_lr != lbl_rl)
v_bnd = bnd & (lbl_tb != 0) & (lbl_bt != 0) & (lbl_tb != lbl_bt)

h_ys, h_xs = np.where(h_bnd)
v_ys, v_xs = np.where(v_bnd)

h_pairs_raw = (np.column_stack([lbl_lr[h_ys, h_xs], lbl_rl[h_ys, h_xs]])
               if len(h_ys) > 0 else np.empty((0, 2), dtype=np.int32))
v_pairs_raw = (np.column_stack([lbl_tb[v_ys, v_xs], lbl_bt[v_ys, v_xs]])
               if len(v_ys) > 0 else np.empty((0, 2), dtype=np.int32))

_stacked = [a for a in [h_pairs_raw, v_pairs_raw] if len(a) > 0]
if _stacked:
    all_pairs    = np.sort(np.vstack(_stacked), axis=1)
    unique_pairs = np.unique(all_pairs, axis=0)
    mv = (np.isin(unique_pairs[:, 0], list(valid_ids)) &
          np.isin(unique_pairs[:, 1], list(valid_ids)))
    unique_pairs = unique_pairs[mv]
else:
    unique_pairs = np.empty((0, 2), dtype=np.int32)
print(f"   Adjacent valid pairs: {len(unique_pairs):,}")

# Precompute dark_ratio: tỉ lệ dark pixels tại boundary pixels giữa mỗi cặp zone
# Dùng để ngăn merge zone bị chia bởi đường đen in thật
_max_id = int(lbl.max()) + 1
_h_a    = lbl_lr[h_ys, h_xs].astype(np.int64)
_h_b    = lbl_rl[h_ys, h_xs].astype(np.int64)
_h_dk   = (dark_map[h_ys, h_xs] > 0).astype(np.int32)
_v_a    = lbl_tb[v_ys, v_xs].astype(np.int64)
_v_b    = lbl_bt[v_ys, v_xs].astype(np.int64)
_v_dk   = (dark_map[v_ys, v_xs] > 0).astype(np.int32)
_all_a  = np.concatenate([_h_a, _v_a])
_all_b  = np.concatenate([_h_b, _v_b])
_all_dk = np.concatenate([_h_dk, _v_dk])
_lo     = np.minimum(_all_a, _all_b)
_hi     = np.maximum(_all_a, _all_b)
_pid    = _lo * _max_id + _hi

pair_dark_ratio = {}
if len(_pid) > 0:
    _sidx   = np.argsort(_pid)
    _spid   = _pid[_sidx]
    _sdk    = _all_dk[_sidx]
    _splits = np.where(np.diff(_spid))[0] + 1
    for gid, gd in zip(np.split(_spid, _splits), np.split(_sdk, _splits)):
        if len(gid) == 0:
            continue
        pk = int(gid[0])
        pair_dark_ratio[(pk // _max_id, pk % _max_id)] = float(gd.sum()) / len(gd)

n_t1 = 0
n_t1_blocked_dark = 0
for z1, z2 in unique_pairs.tolist():
    z1, z2 = int(z1), int(z2)
    dr = pair_dark_ratio.get((min(z1, z2), max(z1, z2)), 0.0)
    if dr >= DARK_BOUNDARY_RATIO:
        n_t1_blocked_dark += 1
        continue
    if zone_cluster[z1] != zone_cluster[z2]:
        continue
    dist = float(np.sqrt(((zone_lab[z1] - zone_lab[z2]) ** 2).sum()))
    if dist <= ADJ_INTRA_THRESH:
        if _union(z1, z2):
            n_t1 += 1
print(f"   Tầng 1 (liền kề + cluster): +{n_t1} merges  (chặn dark boundary: {n_t1_blocked_dark})")

# 7c. Tầng 2: gần nhau + gần như cùng màu (lô cách đường)
n_t2 = 0
zone_ids_arr = sorted(valid_ids)
for i, id1 in enumerate(zone_ids_arr):
    b1 = zone_bbox[id1]
    c1 = zone_lab[id1]
    for id2 in zone_ids_arr[i + 1:]:
        if _find(id1) == _find(id2):
            continue
        if zone_cluster[id1] != zone_cluster[id2]:
            continue
        dist_lab = float(np.sqrt(((c1 - zone_lab[id2]) ** 2).sum()))
        if dist_lab > PROXIMITY_LAB:
            continue
        b2  = zone_bbox[id2]
        dx  = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        dy  = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        if max(dx, dy) <= PROXIMITY_GAP_PX:
            if _union(id1, id2):
                n_t2 += 1
print(f"   Tầng 2 (proximity {PROXIMITY_GAP_PX}px + LAB≤{PROXIMITY_LAB}): +{n_t2} merges")

# 7d. Tái tạo zones sau merge
groups = defaultdict(list)
for z in zones:
    groups[_find(z['zone_id'])].append(z)

zones_merged = []
for root_id, group in groups.items():
    total_area = sum(z['area_px'] for z in group)
    cx = int(sum(z['centroid_px'][0] * z['area_px'] for z in group) / total_area)
    cy = int(sum(z['centroid_px'][1] * z['area_px'] for z in group) / total_area)
    x0 = min(z['bbox_px'][0] for z in group)
    y0 = min(z['bbox_px'][1] for z in group)
    x1 = max(z['bbox_px'][2] for z in group)
    y1 = max(z['bbox_px'][3] for z in group)
    largest  = max(group, key=lambda z: z['area_px'])
    child_ids = sorted([z['zone_id'] for z in group])
    lon_c, lat_c = px_to_geo(cx, cy)
    zones_merged.append({
        'zone_id':        root_id,
        'area_px':        total_area,
        'centroid_px':    [cx, cy],
        'centroid_geo':   {'lon': lon_c, 'lat': lat_c},
        'bbox_px':        [x0, y0, x1, y1],
        'color_rgb':      largest['color_rgb'],
        'avg_saturation': round(largest['avg_saturation'], 1),
        'sub_zones':      len(group),
        'child_zone_ids': child_ids,  # Track which pre-merge zones were merged
    })
zones_merged.sort(key=lambda z: z['area_px'], reverse=True)
print(f"   Zones: {n_before_merge} → {len(zones_merged)}  (t1={n_t1}, t2={n_t2})")

# 7e. Remap labels array (vectorised)
max_lbl = int(labels_raw.max())
remap   = np.arange(max_lbl + 1, dtype=np.int32)
for zid in valid_ids:
    remap[zid] = _find(zid)
labels = remap[labels_raw]

# ─────────────────────────────────────────────────────────────
print("\n8. Export")
# ─────────────────────────────────────────────────────────────
# Build reverse mapping: pre_merge_zone_id -> post_merge_zone_id
child_to_parent = {}
for merged in zones_merged:
    parent_id = merged['zone_id']
    for child_id in merged['child_zone_ids']:
        child_to_parent[child_id] = parent_id

# Save pre-merge zones (original flood-fill results) with parent mapping
zones_pre_merge = []
for z in zones:
    z_copy = z.copy()
    z_copy.pop('_mean_lab', None)
    z_copy['parent_zone_id'] = child_to_parent.get(z['zone_id'], None)  # Reverse mapping
    zones_pre_merge.append(z_copy)

# Save post-merge zones with child relationships
for z in zones_merged:
    z.pop('_mean_lab', None)

output = {
    'metadata': {
        'method':             'boundary_flood_fill_v1_ensemble',
        'image_size':         [W, H],
        'dpi':                DPI,
        'configs':            [c['name'] for c in CONFIGS],
        'n_color_clusters':   N_COLOR_CLUSTERS,
        'adj_intra_thresh':   ADJ_INTRA_THRESH,
        'proximity_gap_px':   PROXIMITY_GAP_PX,
        'proximity_lab':      PROXIMITY_LAB,
        'zones_before_merge': n_before_merge,
        'zones_after_merge':  len(zones_merged),
    },
    'summary': {
        'total_zones_pre_merge': len(zones_pre_merge),
        'total_zones_post_merge': len(zones_merged),
        'min_area':    MIN_ZONE_AREA,
        'max_area':    MAX_ZONE_AREA,
    },
    'zones_pre_merge': zones_pre_merge,   # Original zones before merge
    'zones_post_merge': zones_merged,     # Merged zones with child_zone_ids
}

OUT_JSON   = f'{OUT_DIR}\\to_dan_pho_v1.json'
OUT_LABELS = f'{OUT_DIR}\\v1_labels.npy'
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, cls=NpEncoder)
np.save(OUT_LABELS, labels.astype(np.int32))
print(f"   {OUT_JSON}")
print(f"   {OUT_LABELS}")

# ─────────────────────────────────────────────────────────────
print("\n9. Visualization")
# ─────────────────────────────────────────────────────────────
max_id = max((z['zone_id'] for z in zones_merged), default=0)
np.random.seed(42)
colors_rand = np.random.randint(60, 255, (max_id + 1, 3), dtype=np.uint8)

seg_vis = np.zeros_like(img)
for z in zones_merged:
    seg_vis[labels == z['zone_id']] = colors_rand[z['zone_id']]

overlay = cv2.addWeighted(img, 0.5, seg_vis, 0.5, 0)
for z in zones_merged:
    cx, cy = z['centroid_px']
    cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)

cv2.imwrite(f'{OUT_DIR}\\v1_overlay.png',  overlay)
cv2.imwrite(f'{OUT_DIR}\\v1_segments.png', seg_vis)
print("   Saved: v1_overlay.png, v1_segments.png")

print(f"\n{'='*60}")
print(f"DONE  |  configs={len(CONFIGS)}  flood-fill={n_before_merge}  →  merge={len(zones_merged)} zones")
if zones_merged:
    print(f"Area  :  {zones_merged[-1]['area_px']:,} – {zones_merged[0]['area_px']:,} px")
avg_sub = sum(z['sub_zones'] for z in zones_merged) / max(len(zones_merged), 1)
print(f"Avg sub-zones merged: {avg_sub:.1f}")
print(f"{'='*60}")
print("""
Tune nếu cần:
  Còn nhiều unsegmented → thêm config mới vào CONFIGS hoặc hạ DARK_THRESH / Canny thresholds
  Over-segment → bỏ config B (faint_lines) hoặc tăng MIN_ZONE_AREA
  Merge chưa đủ  → tăng ADJ_INTRA_THRESH (18→25) hoặc PROXIMITY_GAP_PX
  Merge quá mạnh → giảm N_COLOR_CLUSTERS (30→20) hoặc PROXIMITY_LAB (10→6)
""")
