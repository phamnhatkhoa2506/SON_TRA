# -*- coding: utf-8 -*-
"""
Pipeline v5 — Tinh chỉnh segmentation
Cải thiện:
- Loại bỏ text/ký hiệu nhỏ trước khi detect boundary
- Dùng ONLY color gradient (Sobel) thay vì adaptive threshold
- Không merge → giữ nguyên 126+ zones
- Tìm morphological sweet spot  
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

DPI = 150
PDF_IMG      = f'd:\Programming\Python\SON_TRA\map_{DPI}dpi.png'
GEOJSON_PATH = r'd:\Programming\Python\SON_TRA\phuong_son_tra.geojson'
ALIGN_PARAMS = r'd:\Programming\Python\SON_TRA\alignment_params.json'


# ── Load alignment params (calibrated from manual_align.py) ──
with open(ALIGN_PARAMS, encoding='utf-8') as f:
    _ap = json.load(f)
# Alignment được calibrate ở 150 DPI; scale lại theo DPI hiện tại
ALIGN_DPI = _ap.get('dpi', 150)
DPI_FACTOR = ALIGN_DPI / DPI         # 150/300 = 0.5  → pixel giờ nhỏ hơn → deg/pixel giảm
LON0    = _ap['lon0']                # pixel (0,0) không đổi giữa các DPI
LAT0    = _ap['lat0']
SCALE_X = _ap['scale_x'] * DPI_FACTOR
SCALE_Y = _ap['scale_y'] * DPI_FACTOR
print(f"Alignment: lon0={LON0:.8f}, lat0={LAT0:.8f}  (calibrated @ {ALIGN_DPI} DPI → scaled to {DPI} DPI)")
print(f"           scale_x={SCALE_X:.10f}, scale_y={SCALE_Y:.10f}")

def geo_to_px(lon, lat):
    x = (lon - LON0) / SCALE_X
    y = (LAT0 - lat) / SCALE_Y
    return int(round(x)), int(round(y))

def px_to_geo(px, py):
    lon = LON0 + px * SCALE_X
    lat = LAT0 - py * SCALE_Y
    return round(lon, 7), round(lat, 7)

# ─── Load ───
print("1. Load data")
img = cv2.imread(PDF_IMG)
H, W = img.shape[:2]
print(f"   {W}x{H}")

with open(GEOJSON_PATH, encoding='utf-8') as f:
    gj = json.load(f)

ward_mask_raw = np.zeros((H, W), dtype=np.uint8)
for feat in gj['features']:
    geom = feat['geometry']
    polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
    for poly in polys:
        for ring in poly:
            pts = np.array([geo_to_px(c[0], c[1]) for c in ring], dtype=np.int32)
            cv2.fillPoly(ward_mask_raw, [pts], 255)
print(f"   Ward mask (raw from GeoJSON): {ward_mask_raw.sum()//255:,} px")

# ─── Loại bỏ vùng nước/phi đô thị khỏi ward mask ───
print("\n1b. Loại bỏ vùng nước (vịnh, biển) khỏi ward mask")
# Dùng ảnh smooth để phân tích màu (tránh text/ký hiệu nhỏ)
smooth_pre = cv2.bilateralFilter(img, d=11, sigmaColor=80, sigmaSpace=80)
hsv_pre = cv2.cvtColor(smooth_pre, cv2.COLOR_BGR2HSV)
sat_ch = hsv_pre[:,:,1]  # Saturation channel
val_ch = hsv_pre[:,:,2]  # Value channel

# Vùng nước/nền trên bản đồ có saturation rất thấp (xám, trắng)
# Vùng tô màu (TDP) có saturation cao hơn
# Dùng S < 20 để detect vùng không tô màu (nước, nền)
low_sat = (sat_ch < 20).astype(np.uint8) * 255
low_sat_in_ward = cv2.bitwise_and(low_sat, ward_mask_raw)

# Chỉ loại bỏ các vùng low-saturation LỚN (> 5000 px) — tránh xóa đường biên nhỏ
n_ls, labels_ls, stats_ls, _ = cv2.connectedComponentsWithStats(low_sat_in_ward, connectivity=4)
water_mask = np.zeros((H, W), dtype=np.uint8)
for i in range(1, n_ls):
    if stats_ls[i, cv2.CC_STAT_AREA] > 5000:
        water_mask[labels_ls == i] = 255

# Cũng detect background ngoài map content (nền xám-xanh)
# Pixels với V < 120 và S < 30 thường là nền map hoặc vùng ngoài
dark_bg = ((val_ch < 120) & (sat_ch < 30)).astype(np.uint8) * 255
dark_bg_in_ward = cv2.bitwise_and(dark_bg, ward_mask_raw)
n_db, labels_db, stats_db, _ = cv2.connectedComponentsWithStats(dark_bg_in_ward, connectivity=4)
for i in range(1, n_db):
    if stats_db[i, cv2.CC_STAT_AREA] > 3000:
        water_mask[labels_db == i] = 255

# Detect biển/vịnh — màu XANH CYAN/TEAL VÀ chạm biên ward (mở ra biển)
# Zone cyan hợp lệ nằm GỌN bên trong ward, không chạm biên ngoài
hue_ch = hsv_pre[:,:,0]
water_cyan = (
    (hue_ch >= 80) & (hue_ch <= 130) &
    (sat_ch >= 20) & (sat_ch <= 150) &
    (val_ch >= 140)
).astype(np.uint8) * 255
water_cyan_in_ward = cv2.bitwise_and(water_cyan, ward_mask_raw)

# Tạo mask đường biên ward (1 pixel ở rìa ward polygon)
ward_boundary = cv2.morphologyEx(
    ward_mask_raw, cv2.MORPH_GRADIENT,
    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
)

n_cy, labels_cy, stats_cy, _ = cv2.connectedComponentsWithStats(water_cyan_in_ward, connectivity=4)
print(f"   Cyan/teal components in ward: {n_cy - 1}")

water_kept = 0
for i in range(1, n_cy):
    area_cy = stats_cy[i, cv2.CC_STAT_AREA]
    # Yêu cầu: cyan phải LỚN (>30k) VÀ chạm biên ward
    if area_cy < 30000:
        continue
    comp_mask = (labels_cy == i)
    # Có pixel nào chạm biên ward không?
    touches_boundary = (comp_mask & (ward_boundary > 0)).any()
    if touches_boundary:
        water_mask[comp_mask] = 255
        water_kept += 1
print(f"   Cyan components removed as water (large + touches boundary): {water_kept}")

# Dilate water mask để tạo buffer zone
k_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
water_mask = cv2.dilate(water_mask, k_water, iterations=1)

# Refined ward mask = loại bỏ nước
ward_mask = cv2.bitwise_and(ward_mask_raw, cv2.bitwise_not(water_mask))

# Loại header/legend area
ward_mask[:120, :] = 0
ward_mask[H-80:, :] = 0

print(f"   Water mask: {water_mask.sum()//255:,} px loại bỏ")
print(f"   Ward mask (refined): {ward_mask.sum()//255:,} px")

# ─── Segmentation by K-means color clustering ───
# Triết lý: zone NÀO có MÀU KHÁC NHAU → tự tách. Không cần detect boundary.
# Bước 1: Closing để lấp text + hatching bằng màu zone xung quanh.
# Bước 2: Bilateral để smooth nhẹ, giữ edge giữa các zone.
# Bước 3: K-means quantize ảnh thành N màu chủ đạo.
# Bước 4: Mỗi cluster màu → connected components → zones.
print("\n2. Pre-process: closing + bilateral để chuẩn hóa zone fills")

# Closing 17×17: đủ lấp text + hatching dày 5-8px (sọc tím trên nền trắng v.v.)
k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k_clean)

# Median blur 11×11: triệt tiêu sọc còn sót (mỗi pixel = median 11×11 quanh nó)
# Hatching alternating với chu kỳ 10-15px → median sẽ chọn màu CHIẾM ĐA SỐ → đồng nhất
img_med = cv2.medianBlur(img_closed, 11)

# Bilateral: smooth zone fills nhưng preserve edges giữa các zone
img_smooth = cv2.bilateralFilter(img_med, d=11, sigmaColor=50, sigmaSpace=50)

# Lưu debug
cv2.imwrite(r'd:\Programming\Python\SON_TRA\v5_img_clean.png', img_smooth)

# ── 2a: K-means clustering trên pixels TRONG ward_mask ──
print("\n3. K-means color clustering")
ward_pixels = img_smooth[ward_mask == 255]   # (N, 3) BGR
print(f"   Pixels to cluster: {len(ward_pixels):,}")

# Convert sang LAB color space — Euclidean distance ≈ perceptual difference
ward_pixels_lab = cv2.cvtColor(
    ward_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
).reshape(-1, 3).astype(np.float32)

# K-means: thường ~30-40 màu fills khác nhau trên bản đồ admin
N_CLUSTERS = 35
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
# Subsample để chạy nhanh hơn nếu quá nhiều pixel
if len(ward_pixels_lab) > 500000:
    idx_sub = np.random.choice(len(ward_pixels_lab), 500000, replace=False)
    sample = ward_pixels_lab[idx_sub]
else:
    sample = ward_pixels_lab
_, _, centers_lab = cv2.kmeans(sample, N_CLUSTERS, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

# Gán nhãn cho TOÀN BỘ pixels: nearest cluster
# Tính distance từ mỗi pixel đến các center
print(f"   Assigning labels to all pixels...")
# Tính theo batch để tiết kiệm RAM (mảng (N,K,3) full = 6.8 GB cho 17M pixel)
batch = 100000
labels_all = np.zeros(len(ward_pixels_lab), dtype=np.int32)
for s in range(0, len(ward_pixels_lab), batch):
    e = min(s + batch, len(ward_pixels_lab))
    d = ward_pixels_lab[s:e, None, :] - centers_lab[None, :, :]
    labels_all[s:e] = np.argmin(np.sum(d * d, axis=2), axis=1)

# Tạo cluster map size (H, W): 0 = ngoài ward, 1..K = cluster idx + 1
cluster_map = np.zeros((H, W), dtype=np.int32)
cluster_map[ward_mask == 255] = labels_all + 1

# ── 2c: Reassign DARK clusters (text artifacts) → cluster lân cận chiếm đa số ──
# Centers_lab[:, 0] là L channel (0-255). L < 60 ≈ rất tối (text)
dark_cluster_idx = np.where(centers_lab[:, 0] < 60)[0]
print(f"   Dark clusters detected (text): {len(dark_cluster_idx)} → reassigning")

if len(dark_cluster_idx) > 0:
    # Mask các pixel trong dark clusters
    dark_mask = np.isin(cluster_map - 1, dark_cluster_idx) & (ward_mask == 255)
    n_dark = dark_mask.sum()
    print(f"   Dark pixels: {n_dark:,}")

    # Với mỗi dark pixel, lấy nhãn từ neighbor non-dark gần nhất
    # Cách hiệu quả: dùng cv2.dilate trên cluster_map đã loại dark, lặp lại
    cluster_clean = cluster_map.copy()
    cluster_clean[dark_mask] = 0  # Đánh dấu cần fill

    # Lặp dilate để propagate label từ vùng non-dark sang dark
    # Mỗi iteration mở rộng label ra 1 pixel
    k_prop = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(15):  # ~15 px max propagation
        # Find dark pixels có neighbor non-zero
        unfilled = (cluster_clean == 0) & dark_mask
        if not unfilled.any():
            break
        # Dilate cluster_clean (treats 0 như background): non-zero spreads
        dilated = cv2.dilate(cluster_clean.astype(np.uint16), k_prop).astype(np.int32)
        # Chỉ update pixels unfilled
        cluster_clean = np.where(unfilled & (dilated > 0), dilated, cluster_clean)

    cluster_map = cluster_clean
    print(f"   Dark pixels reassigned (15 iterations)")

# Lưu debug: visualize clusters
centers_bgr = cv2.cvtColor(
    centers_lab.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR
).reshape(-1, 3)
cluster_vis = np.zeros((H, W, 3), dtype=np.uint8)
for k in range(N_CLUSTERS):
    cluster_vis[cluster_map == (k + 1)] = centers_bgr[k]
cv2.imwrite(r'd:\Programming\Python\SON_TRA\v5_clusters.png', cluster_vis)
print(f"   Saved: v5_img_clean.png, v5_clusters.png")

# ── 2b: Mỗi cluster → connected components → zones ──
# Gộp tất cả components từ mọi cluster vào 1 label map duy nhất
print("\n4. Connected components per cluster")
labels = np.zeros((H, W), dtype=np.int32)
stats_list = []
centroids_list = []
next_label = 1

for k in range(1, N_CLUSTERS + 1):
    cluster_mask = (cluster_map == k).astype(np.uint8)
    if cluster_mask.sum() < 600:
        continue
    # Opening nhẹ để loại noise pixel-level
    k_op = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, k_op)

    n_c, lbl_c, stats_c, cent_c = cv2.connectedComponentsWithStats(cluster_mask, connectivity=4)
    for j in range(1, n_c):
        area_j = stats_c[j, cv2.CC_STAT_AREA]
        if area_j < 600 or area_j > 200000:
            continue
        labels[lbl_c == j] = next_label
        stats_list.append(stats_c[j])
        centroids_list.append(cent_c[j])
        next_label += 1

n_labels = next_label
if len(stats_list) > 0:
    stats = np.vstack([np.zeros((1, 5), dtype=np.int32), np.array(stats_list)])
    centroids = np.vstack([np.zeros((1, 2)), np.array(centroids_list)])
else:
    stats = np.zeros((1, 5), dtype=np.int32)
    centroids = np.zeros((1, 2))
print(f"   Total zone candidates: {n_labels - 1}")

# ─── Filter & extract zones ───
MIN_AREA = 600
MAX_AREA = 200000

# Pre-compute HSV cho phân tích sau
hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

zones = []
skipped_low_sat = 0
skipped_low_var = 0
for i in range(1, n_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area < MIN_AREA or area > MAX_AREA:
        continue
    cx = int(centroids[i][0])
    cy = int(centroids[i][1])
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    mask_i = (labels == i)
    pix = img[mask_i]
    if len(pix) == 0: continue

    # Shape filter: loại STREET NETWORK (mạng lưới đường nâu)
    # Street: bbox khổng lồ, area nhỏ → fill_ratio thấp + perimeter² / area cao
    bbox_area = max(w * h, 1)
    fill_ratio = area / bbox_area
    if fill_ratio < 0.25 and area > 2000:
        skipped_low_var += 1
        continue
    # Compactness metric: perimeter² / (4π * area). Circle=1, snake>>10
    contours, _ = cv2.findContours(mask_i.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perim = sum(cv2.arcLength(c, True) for c in contours)
        compactness = (perim * perim) / (4 * np.pi * area + 1e-6)
        if compactness > 25 and area > 1500:
            # Quá nhiều khúc cong → street network
            skipped_low_var += 1
            continue

    # Post-filter: loại bỏ vùng nước/nền còn sót
    pix_hsv = hsv_full[mask_i]
    avg_sat = float(np.mean(pix_hsv[:, 1]))
    if avg_sat < 15:
        skipped_low_sat += 1
        continue

    # Color variance check
    color_std = float(np.mean(np.std(pix.astype(np.float32), axis=0)))
    if color_std < 6 and area > 3000:
        skipped_low_var += 1
        continue

    # Dominant color
    pq = (pix // 16) * 16
    cu, cc = np.unique(pq.reshape(-1,3), axis=0, return_counts=True)
    dom = cu[np.argmax(cc)]
    dom_rgb = [int(dom[2]), int(dom[1]), int(dom[0])]

    lon_c, lat_c = px_to_geo(cx, cy)
    zones.append({
        'zone_id': int(i),
        'area_px': int(area),
        'centroid_px': [cx, cy],
        'centroid_geo': {'lon': lon_c, 'lat': lat_c},
        'bbox_px': [int(x), int(y), int(x+w), int(y+h)],
        'color_rgb': dom_rgb,
        'avg_saturation': round(avg_sat, 1),
        'color_std': round(color_std, 1),
    })

zones.sort(key=lambda z: z['area_px'], reverse=True)
print(f"   Valid zones: {len(zones)}")
print(f"   Skipped (low saturation / water): {skipped_low_sat}")
print(f"   Skipped (low color variance): {skipped_low_var}")

# ─── Export + Visualization ───
print("5. Export")
output = {
    'metadata': {'method': f'K-means color clustering (LAB, K=35)',
                 'image_size': [W, H], 'dpi': DPI},
    'summary': {'total_zones': len(zones), 'min_area': MIN_AREA, 'max_area': MAX_AREA},
    'zones': zones,
}
OUT_JSON = r'd:\Programming\Python\SON_TRA\to_dan_pho_v5.json'
with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, cls=NpEncoder)

# Visualization: colored components
print("6. Visualization")
np.random.seed(42)
colors_rand = np.random.randint(60, 255, (max(z['zone_id'] for z in zones)+1, 3))
seg_vis = np.zeros_like(img)
for z in zones:
    mask_z = (labels == z['zone_id'])
    seg_vis[mask_z] = colors_rand[z['zone_id']]

# Overlay: 50% original + 50% segmentation
overlay = cv2.addWeighted(img, 0.5, seg_vis, 0.5, 0)
# Draw centroids
for z in zones:
    cx, cy = z['centroid_px']
    cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)

# Scale down
VIZ_S = 1.0
overlay_s = cv2.resize(overlay, (int(W*VIZ_S), int(H*VIZ_S)))
cv2.imwrite(r'd:\Programming\Python\SON_TRA\v5_overlay.png', overlay_s)

# Pure segmentation map
seg_s = cv2.resize(seg_vis, (int(W*VIZ_S), int(H*VIZ_S)))
cv2.imwrite(r'd:\Programming\Python\SON_TRA\v5_segments.png', seg_s)

cv2.imwrite(r'd:\Programming\Python\SON_TRA\v5_water_mask.png', water_mask)
cv2.imwrite(r'd:\Programming\Python\SON_TRA\v5_ward_refined.png', ward_mask)

# Lưu labels array để OCR script dùng
np.save(r'd:\Programming\Python\SON_TRA\v5_labels.npy', labels.astype(np.int32))
print("   Saved: v5_labels.npy (zone label map for OCR matching)")

print(f"\nDONE: {len(zones)} zones")
print(f"Area range: {zones[-1]['area_px']:,} - {zones[0]['area_px']:,} px")
