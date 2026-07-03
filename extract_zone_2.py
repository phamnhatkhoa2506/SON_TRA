# -*- coding: utf-8 -*-
"""
Pipeline v2 — Adaptive Boundary + Post-merge

Cải tiến so với v1:
  1. Ngưỡng ranh giới TỰ ĐỘNG từ thống kê ảnh (Otsu + gradient percentile)
  2. Over-segment trước → sau đó POST-MERGE các zone lân cận cùng màu
     (merge threshold tự tính từ phân phối khoảng cách màu thực tế)

Triết lý: "over-segment thì merge được, under-segment thì không cứu được"
  Không cần đặt tham số thủ công — thuật toán tự học từ ảnh.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import json
from collections import defaultdict

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

# ── Paths ──
DPI          = 150
PDF_IMG      = f'd:\\Programming\\Python\\SON_TRA\\map_{DPI}dpi.png'
GEOJSON_PATH = r'd:\Programming\Python\SON_TRA\phuong_son_tra.geojson'
ALIGN_PARAMS = r'd:\Programming\Python\SON_TRA\alignment_params.json'
OUT_DIR      = r'd:\Programming\Python\SON_TRA'

# ─────────────────────────────────────────────────────────────
# Chỉ còn CÁC THAM SỐ KHÔNG THỂ TỰ ĐỘNG HOÁ
# ─────────────────────────────────────────────────────────────
MIN_DARK_AREA  = 300    # dark blob nhỏ hơn = nét chữ đơn lẻ → bỏ
CLOSE_K        = 3      # kernel closing khe hở ranh giới
CLOSE_ITERS    = 2

MIN_ZONE_AREA  = 800    # px² — artifact nhỏ sau flood-fill
MAX_ZONE_AREA  = 600_000
MIN_AVG_SAT    = 12
MAX_COMPACT    = 35
MIN_FILL_RATIO = 0.15

# Post-merge: merge adjacent zones có màu LAB gần nhau
# None = tự tính từ phân phối. Đặt số (vd: 18.0) để override.
MERGE_THRESH_OVERRIDE = None
MERGE_THRESH_PERCENTILE = 20   # dùng khi auto: lấy percentile thứ N của dist
MERGE_THRESH_MAX       = 28.0  # trần: không merge khi dist > này dù auto
# ─────────────────────────────────────────────────────────────

# ── Alignment ──
with open(ALIGN_PARAMS, encoding='utf-8') as f:
    _ap = json.load(f)
ALIGN_DPI  = _ap.get('dpi', 150)
DPI_FACTOR = ALIGN_DPI / DPI
LON0    = _ap['lon0']
LAT0    = _ap['lat0']
SCALE_X = _ap['scale_x'] * DPI_FACTOR
SCALE_Y = _ap['scale_y'] * DPI_FACTOR
print(f"Alignment: lon0={LON0:.8f}  lat0={LAT0:.8f}  ({ALIGN_DPI}→{DPI} DPI)")

def geo_to_px(lon, lat):
    return int(round((lon - LON0) / SCALE_X)), int(round((LAT0 - lat) / SCALE_Y))

def px_to_geo(px, py):
    return round(LON0 + px * SCALE_X, 7), round(LAT0 - py * SCALE_Y, 7)

# ─────────────────────────────────────────────────────────────
print("\n1. Load data")
# ─────────────────────────────────────────────────────────────
img = cv2.imread(PDF_IMG)
assert img is not None, f"Không đọc được: {PDF_IMG}"
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
sat_ch, val_ch, hue_ch = hsv_pre[:,:,1], hsv_pre[:,:,2], hsv_pre[:,:,0]

water_mask = np.zeros((H, W), dtype=np.uint8)
for mask_in, area_thr in [
    (cv2.bitwise_and((sat_ch < 20).astype(np.uint8)*255, ward_mask_raw), 5000),
    (cv2.bitwise_and(((val_ch<120)&(sat_ch<30)).astype(np.uint8)*255, ward_mask_raw), 3000),
]:
    n, lbl, sts, _ = cv2.connectedComponentsWithStats(mask_in, connectivity=4)
    for i in range(1, n):
        if sts[i, cv2.CC_STAT_AREA] > area_thr:
            water_mask[lbl == i] = 255

ward_boundary_edge = cv2.morphologyEx(ward_mask_raw, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
cyan_in_ward = cv2.bitwise_and(
    ((hue_ch>=80)&(hue_ch<=130)&(sat_ch>=20)&(sat_ch<=150)&(val_ch>=140)).astype(np.uint8)*255,
    ward_mask_raw)
n, lbl, sts, _ = cv2.connectedComponentsWithStats(cyan_in_ward, connectivity=4)
for i in range(1, n):
    if sts[i, cv2.CC_STAT_AREA] >= 30000 and ((lbl==i) & (ward_boundary_edge>0)).any():
        water_mask[lbl == i] = 255

water_mask  = cv2.dilate(water_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15)))
ward_mask   = cv2.bitwise_and(ward_mask_raw, cv2.bitwise_not(water_mask))
ward_mask[:120, :]    = 0
ward_mask[H - 80:, :] = 0
print(f"   Ward: {ward_mask.sum()//255:,} px  |  nước: {water_mask.sum()//255:,} px")

# ─────────────────────────────────────────────────────────────
print("\n3. Pre-process")
# ─────────────────────────────────────────────────────────────
k_clean    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_clean)
img_med    = cv2.medianBlur(img_closed, 7)
img_clean  = cv2.bilateralFilter(img_med, d=9, sigmaColor=45, sigmaSpace=45)

# ─────────────────────────────────────────────────────────────
print("\n4. Adaptive boundary detection")
# ─────────────────────────────────────────────────────────────

# ── 4a. Auto dark threshold (Otsu trên vùng ward) ──
gray_orig  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_ward  = gray_orig.copy()
gray_ward[ward_mask == 0] = 200  # điền neutral để Otsu không bị lệch
otsu_val, _ = cv2.threshold(gray_ward, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dark_thresh = int(min(otsu_val * 0.75, 90))  # 75% Otsu: bảo thủ, chỉ lấy rất tối
print(f"   Auto dark threshold : {dark_thresh}  (Otsu={otsu_val:.0f})")

dark_raw = ((gray_orig < dark_thresh) & (ward_mask == 255)).astype(np.uint8) * 255
n_dk, lbl_dk, sts_dk, _ = cv2.connectedComponentsWithStats(dark_raw, connectivity=4)
dark_boundary = np.zeros((H, W), dtype=np.uint8)
for i in range(1, n_dk):
    if sts_dk[i, cv2.CC_STAT_AREA] >= MIN_DARK_AREA:
        dark_boundary[lbl_dk == i] = 255
print(f"   Dark boundary px    : {dark_boundary.sum()//255:,}")

# ── 4b. Auto Canny thresholds (từ phân phối gradient) ──
lab_clean = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
l_ch = lab_clean[:, :, 0].astype(np.float32)
gx   = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
gy   = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
grad_l = np.sqrt(gx*gx + gy*gy)

grad_vals = grad_l[ward_mask == 255]
grad_vals = grad_vals[grad_vals > 0.5]
if len(grad_vals) >= 100:
    p25 = float(np.percentile(grad_vals, 25))
    p65 = float(np.percentile(grad_vals, 65))
    canny_lo_l, canny_hi_l = max(p25 * 0.6, 5.0), max(p65, 20.0)
else:
    canny_lo_l, canny_hi_l = 15.0, 50.0
print(f"   Auto Canny-L thresh : {canny_lo_l:.1f} / {canny_hi_l:.1f}")

canny_l = cv2.Canny(lab_clean[:, :, 0], canny_lo_l, canny_hi_l)
canny_a = cv2.Canny(lab_clean[:, :, 1], max(canny_lo_l*0.6, 5), max(canny_hi_l*0.6, 15))
canny_b = cv2.Canny(lab_clean[:, :, 2], max(canny_lo_l*0.6, 5), max(canny_hi_l*0.6, 15))
grad_boundary = cv2.bitwise_or(canny_l, cv2.bitwise_or(canny_a, canny_b))
grad_boundary = cv2.bitwise_and(grad_boundary, ward_mask)
print(f"   Gradient boundary px: {grad_boundary.sum()//255:,}")

# ── 4c. Kết hợp + đóng khe hở ──
combined = cv2.bitwise_or(grad_boundary, dark_boundary)
k_gap    = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_K, CLOSE_K))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_gap, iterations=CLOSE_ITERS)
combined = cv2.dilate(combined, k_gap, iterations=1)
print(f"   Final boundary px   : {combined.sum()//255:,}")

cv2.imwrite(f'{OUT_DIR}\\v2_boundary.png', combined)
print("   Saved: v2_boundary.png")

# ─────────────────────────────────────────────────────────────
print("\n5. Flood-fill")
# ─────────────────────────────────────────────────────────────
fill_region = np.zeros((H, W), dtype=np.uint8)
fill_region[(combined == 0) & (ward_mask == 255)] = 255

n_raw, labels_raw, stats_raw, centroids_raw = cv2.connectedComponentsWithStats(
    fill_region, connectivity=4)
print(f"   Raw flood-fill regions: {n_raw - 1}")

# ─────────────────────────────────────────────────────────────
print("\n6. Lọc zones")
# ─────────────────────────────────────────────────────────────
hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

raw_zones = []
skip_area = skip_shape = skip_sat = 0

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
        skip_shape += 1
        continue

    sub = (labels_raw[y0:y1, x0:x1] == i).astype(np.uint8)
    if area > 1500:
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            perim = sum(cv2.arcLength(c, True) for c in cnts)
            if (perim * perim) / (4 * np.pi * area + 1e-6) > MAX_COMPACT:
                skip_shape += 1
                continue

    pix_hsv = hsv_full[y0:y1, x0:x1][sub == 1]
    if len(pix_hsv) == 0: continue
    avg_sat = float(np.mean(pix_hsv[:, 1]))
    if avg_sat < MIN_AVG_SAT:
        skip_sat += 1
        continue

    # Mean LAB color (cho adaptive merge)
    pix_lab = lab_full[y0:y1, x0:x1][sub == 1]
    mean_lab = pix_lab.mean(axis=0).tolist()

    # Dominant BGR color
    pix_bgr = img[y0:y1, x0:x1][sub == 1]
    pq = (pix_bgr // 16) * 16
    cu, cc = np.unique(pq.reshape(-1, 3), axis=0, return_counts=True)
    dom = cu[np.argmax(cc)]

    lon_c, lat_c = px_to_geo(cx, cy)
    raw_zones.append({
        'zone_id':      int(i),
        'area_px':      area,
        'centroid_px':  [cx, cy],
        'centroid_geo': {'lon': lon_c, 'lat': lat_c},
        'bbox_px':      [x0, y0, x1, y1],
        'color_rgb':    [int(dom[2]), int(dom[1]), int(dom[0])],
        'avg_saturation': round(avg_sat, 1),
        '_mean_lab':    mean_lab,   # dùng nội bộ cho merge, sẽ bỏ khi export
    })

print(f"   Valid zones (trước merge): {len(raw_zones)}")
print(f"   Skip area={skip_area}  shape={skip_shape}  sat={skip_sat}")

# ─────────────────────────────────────────────────────────────
print("\n7. Adaptive post-merge (gộp zone lân cận cùng màu)")
# ─────────────────────────────────────────────────────────────

valid_ids = {z['zone_id'] for z in raw_zones}

# ── 7a. Tìm tất cả cặp zone liền kề (numpy diff — không cần loop pixel) ──
lbl = labels_raw
h_mask = (lbl[:-1, :] != lbl[1:, :]) & (lbl[:-1, :] > 0) & (lbl[1:, :] > 0)
v_mask = (lbl[:, :-1] != lbl[:, 1:]) & (lbl[:, :-1] > 0) & (lbl[:, 1:] > 0)

ys, xs = np.where(h_mask)
h_pairs = np.column_stack([lbl[ys, xs], lbl[ys + 1, xs]])
ys, xs = np.where(v_mask)
v_pairs = np.column_stack([lbl[ys, xs], lbl[ys, xs + 1]])

all_pairs = np.sort(np.vstack([h_pairs, v_pairs]), axis=1)
unique_pairs = np.unique(all_pairs, axis=0)

# Chỉ giữ cặp mà cả 2 đều là valid zone
mask_valid = np.isin(unique_pairs[:, 0], list(valid_ids)) & \
             np.isin(unique_pairs[:, 1], list(valid_ids))
unique_pairs = unique_pairs[mask_valid]
print(f"   Adjacent valid pairs: {len(unique_pairs):,}")

# ── 7b. Tính khoảng cách LAB cho từng cặp ──
zone_lab = {z['zone_id']: np.array(z['_mean_lab']) for z in raw_zones}

dists = []
for z1, z2 in unique_pairs:
    c1, c2 = zone_lab[int(z1)], zone_lab[int(z2)]
    dists.append(float(np.sqrt(((c1 - c2)**2).sum())))

dists_arr = np.array(dists)

# ── 7c. Adaptive merge threshold ──
if MERGE_THRESH_OVERRIDE is not None:
    merge_thresh = float(MERGE_THRESH_OVERRIDE)
    print(f"   Merge threshold     : {merge_thresh:.1f}  (manual override)")
else:
    pct_val = float(np.percentile(dists_arr, MERGE_THRESH_PERCENTILE))
    merge_thresh = min(pct_val, MERGE_THRESH_MAX)
    print(f"   Merge threshold     : {merge_thresh:.1f}  "
          f"(p{MERGE_THRESH_PERCENTILE}={pct_val:.1f}, "
          f"median={np.median(dists_arr):.1f}, "
          f"max={MERGE_THRESH_MAX})")

# ── 7d. Union-Find merge ──
parent = {z['zone_id']: z['zone_id'] for z in raw_zones}

def find(x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root

n_merged = 0
for (z1, z2), dist in zip(unique_pairs.tolist(), dists):
    if dist <= merge_thresh:
        p1, p2 = find(int(z1)), find(int(z2))
        if p1 != p2:
            parent[p2] = p1
            n_merged += 1

print(f"   Merged pairs        : {n_merged}")

# ── 7e. Tái tạo zones sau merge ──
groups = defaultdict(list)
for z in raw_zones:
    groups[find(z['zone_id'])].append(z)

zones = []
for root_id, group in groups.items():
    total_area = sum(z['area_px'] for z in group)
    cx = int(sum(z['centroid_px'][0] * z['area_px'] for z in group) / total_area)
    cy = int(sum(z['centroid_px'][1] * z['area_px'] for z in group) / total_area)
    x0 = min(z['bbox_px'][0] for z in group)
    y0 = min(z['bbox_px'][1] for z in group)
    x1 = max(z['bbox_px'][2] for z in group)
    y1 = max(z['bbox_px'][3] for z in group)
    largest = max(group, key=lambda z: z['area_px'])
    lon_c, lat_c = px_to_geo(cx, cy)
    zones.append({
        'zone_id':      root_id,
        'area_px':      total_area,
        'centroid_px':  [cx, cy],
        'centroid_geo': {'lon': lon_c, 'lat': lat_c},
        'bbox_px':      [x0, y0, x1, y1],
        'color_rgb':    largest['color_rgb'],
        'avg_saturation': round(largest['avg_saturation'], 1),
        'sub_zones':    len(group),
    })
zones.sort(key=lambda z: z['area_px'], reverse=True)
print(f"   Zones sau merge     : {len(zones)}  (từ {len(raw_zones)})")

# ── 7f. Cập nhật labels array ──
max_lbl = int(labels_raw.max())
remap   = np.arange(max_lbl + 1, dtype=np.int32)
for z in raw_zones:
    oid = z['zone_id']
    remap[oid] = find(oid)
labels = remap[labels_raw]  # vectorised — O(H×W), không cần loop

# ─────────────────────────────────────────────────────────────
print("\n8. Export")
# ─────────────────────────────────────────────────────────────
output = {
    'metadata': {
        'method':                'adaptive_boundary_flood_fill_v2',
        'image_size':            [W, H],
        'dpi':                   DPI,
        'auto_dark_thresh':      dark_thresh,
        'auto_canny_l':          [round(canny_lo_l,1), round(canny_hi_l,1)],
        'merge_thresh':          round(merge_thresh, 2),
        'merge_thresh_percentile': MERGE_THRESH_PERCENTILE,
        'zones_before_merge':    len(raw_zones),
    },
    'summary': {
        'total_zones': len(zones),
        'min_area':    MIN_ZONE_AREA,
        'max_area':    MAX_ZONE_AREA,
    },
    'zones': zones,
}

OUT_JSON   = f'{OUT_DIR}\\to_dan_pho_v2.json'
OUT_LABELS = f'{OUT_DIR}\\v2_labels.npy'

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, cls=NpEncoder)
np.save(OUT_LABELS, labels.astype(np.int32))
print(f"   {OUT_JSON}")
print(f"   {OUT_LABELS}")

# ─────────────────────────────────────────────────────────────
print("\n9. Visualization")
# ─────────────────────────────────────────────────────────────
max_id = max((z['zone_id'] for z in zones), default=0)
np.random.seed(42)
colors_rand = np.random.randint(60, 255, (max_id + 1, 3), dtype=np.uint8)

seg_vis = np.zeros_like(img)
for z in zones:
    seg_vis[labels == z['zone_id']] = colors_rand[z['zone_id']]

overlay = cv2.addWeighted(img, 0.5, seg_vis, 0.5, 0)
for z in zones:
    cx, cy = z['centroid_px']
    cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)

cv2.imwrite(f'{OUT_DIR}\\v2_overlay.png',  overlay)
cv2.imwrite(f'{OUT_DIR}\\v2_segments.png', seg_vis)
print("   Saved: v2_overlay.png, v2_segments.png")

print(f"\n{'='*55}")
print(f"DONE  |  {len(raw_zones)} → merge → {len(zones)} zones")
if zones:
    print(f"Area  :  {zones[-1]['area_px']:,} – {zones[0]['area_px']:,} px")
avg_sub = sum(z['sub_zones'] for z in zones) / max(len(zones), 1)
print(f"Avg sub-zones/zone: {avg_sub:.1f}")
print(f"{'='*55}")
print("""
Tune nếu cần:
  Quá nhiều zone → tăng MERGE_THRESH_PERCENTILE (20→30) hoặc MERGE_THRESH_MAX (28→40)
  Quá ít zone   → giảm MERGE_THRESH_PERCENTILE (20→10) hoặc set MERGE_THRESH_OVERRIDE=8
  Boundary yếu  → giảm MIN_DARK_AREA (300→150)
""")
