# -*- coding: utf-8 -*-
"""
Manual Alignment Tool v3 — Improved UX
Features:
  - Mouse drag to pan (lon0, lat0)
  - Mouse scroll to zoom (scale)
  - Entry fields for precise value input
  - Step multiplier (×1, ×10, ×100, ×1000)
  - Boundary line clipping (no int32 overflow)
  - Zoom-to-fit buttons (map / GeoJSON)
  - Arrow keys for fine adjustment
  - Lock aspect ratio option

Tham số:
  lon0, lat0  : tọa độ địa lý tại pixel (0,0) của ảnh
  scale_x     : độ kinh/pixel (×1e-5)
  scale_y     : độ vĩ/pixel  (×1e-5)

Nhấn [Lưu tham số] → alignment_params.json
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2, json, numpy as np
from PIL import Image, ImageTk

# ── Paths ──────────────────────────────────────────────────────
MAP_IMG_PATH = r'd:\Programming\Python\SON_TRA\map_150dpi.png'
GEOJSON_PATH = r'd:\Programming\Python\SON_TRA\phuong_son_tra.geojson'
OUT_PARAMS   = r'd:\Programming\Python\SON_TRA\alignment_params.json'

PREVIEW_W, PREVIEW_H = 1000, 680

# ── Giá trị ban đầu (từ calibration extract_zones_v6.py) ───
INIT = dict(lon0=108.21833, lat0=16.10927,
            sx=1.072, sy=1.046)   # sx/sy in units of 1e-5 deg/px

# ── Load ───────────────────────────────────────────────────────
# Load saved params if available
if os.path.exists(OUT_PARAMS):
    with open(OUT_PARAMS, encoding='utf-8') as f:
        saved = json.load(f)
    INIT['lon0'] = saved['lon0']
    INIT['lat0'] = saved['lat0']
    INIT['sx']   = saved['scale_x'] * 1e5
    INIT['sy']   = saved['scale_y'] * 1e5
    print(f"Loaded params from {OUT_PARAMS}")

print("Loading map..."); sys.stdout.flush()
img_full = cv2.imread(MAP_IMG_PATH)
if img_full is None:
    raise FileNotFoundError(MAP_IMG_PATH)
IMG_H, IMG_W = img_full.shape[:2]
print(f"  Image: {IMG_W}×{IMG_H}")

sc = min(PREVIEW_W/IMG_W, PREVIEW_H/IMG_H)
PW, PH = int(IMG_W*sc), int(IMG_H*sc)
base = cv2.resize(img_full, (PW, PH))
PREVIEW_SCALE_X = PW / IMG_W
PREVIEW_SCALE_Y = PH / IMG_H

print("Loading GeoJSON..."); sys.stdout.flush()
with open(GEOJSON_PATH, encoding='utf-8') as f:
    gj = json.load(f)

rings = []
def collect(geom):
    if geom['type'] == 'MultiPolygon':
        for poly in geom['coordinates']:
            for ring in poly: rings.append(ring)
    elif geom['type'] == 'Polygon':
        for ring in geom['coordinates']: rings.append(ring)
for feat in gj['features']: collect(feat['geometry'])

all_lons = [c[0] for r in rings for c in r]
all_lats = [c[1] for r in rings for c in r]
GJ = dict(lon_min=min(all_lons), lon_max=max(all_lons),
          lat_min=min(all_lats), lat_max=max(all_lats))
print(f"  GeoJSON: {len(rings)} rings, "
      f"lon=[{GJ['lon_min']:.5f},{GJ['lon_max']:.5f}] "
      f"lat=[{GJ['lat_min']:.5f},{GJ['lat_max']:.5f}]")


# ── Cohen-Sutherland line clipping ────────────────────────────
def _cs_code(x, y, xmin, ymin, xmax, ymax):
    c = 0
    if x < xmin: c |= 1
    elif x > xmax: c |= 2
    if y < ymin: c |= 4
    elif y > ymax: c |= 8
    return c

def clip_line(x0, y0, x1, y1, xmin, ymin, xmax, ymax):
    """Clip segment to rect. Returns (x0,y0,x1,y1) or None."""
    c0 = _cs_code(x0, y0, xmin, ymin, xmax, ymax)
    c1 = _cs_code(x1, y1, xmin, ymin, xmax, ymax)
    for _ in range(20):
        if not (c0 | c1): return (int(x0), int(y0), int(x1), int(y1))
        if c0 & c1: return None
        c = c0 or c1
        dx, dy = x1 - x0, y1 - y0
        if c & 8:   x, y = x0 + dx*(ymax-y0)/dy, ymax
        elif c & 4: x, y = x0 + dx*(ymin-y0)/dy, ymin
        elif c & 2: x, y = xmax, y0 + dy*(xmax-x0)/dx
        elif c & 1: x, y = xmin, y0 + dy*(xmin-x0)/dx
        if c == c0: x0, y0, c0 = x, y, _cs_code(x, y, xmin, ymin, xmax, ymax)
        else:       x1, y1, c1 = x, y, _cs_code(x, y, xmin, ymin, xmax, ymax)
    return None


# ── Draw ────────────────────────────────────────────────────────
def geo_to_preview(lon, lat, lon0, lat0, sx, sy):
    """geo → preview pixel (float). sx,sy in deg/px (full scale)."""
    xi = (lon - lon0) / sx
    yi = (lat0 - lat) / sy
    return xi * PREVIEW_SCALE_X, yi * PREVIEW_SCALE_Y


def render(lon0, lat0, sx, sy, grid=True):
    canvas = base.copy()
    MARGIN = 50  # clipping margin px

    # Grid
    if grid:
        lon_span = sx * IMG_W;  lat_span = sy * IMG_H
        for n in range(6):
            lon_g = lon0 + lon_span * n/5
            x, _ = geo_to_preview(lon_g, lat0, lon0, lat0, sx, sy)
            ix = int(x)
            if 0 <= ix <= PW:
                cv2.line(canvas, (ix,0),(ix,PH),(160,160,160),1)
                cv2.putText(canvas, f"{lon_g:.4f}", (ix+2,14),
                            cv2.FONT_HERSHEY_SIMPLEX,.32,(60,60,60),1)
        for n in range(6):
            lat_g = lat0 - lat_span * n/5
            _, y = geo_to_preview(lon0, lat_g, lon0, lat0, sx, sy)
            iy = int(y)
            if 0 <= iy <= PH:
                cv2.line(canvas,(0,iy),(PW,iy),(160,160,160),1)
                cv2.putText(canvas, f"{lat_g:.4f}", (2,iy-2),
                            cv2.FONT_HERSHEY_SIMPLEX,.32,(60,60,60),1)

    # GeoJSON boundary — clip each segment to prevent int32 overflow
    for ring in rings:
        n = len(ring)
        for i in range(n):
            j = (i + 1) % n
            x0, y0 = geo_to_preview(ring[i][0], ring[i][1], lon0, lat0, sx, sy)
            x1, y1 = geo_to_preview(ring[j][0], ring[j][1], lon0, lat0, sx, sy)
            seg = clip_line(x0, y0, x1, y1, -MARGIN, -MARGIN, PW+MARGIN, PH+MARGIN)
            if seg:
                cv2.line(canvas, (seg[0],seg[1]), (seg[2],seg[3]), (0,0,210), 2)

    # Status bar
    cv2.rectangle(canvas,(0,PH-22),(PW,PH),(20,20,20),-1)
    info = (f"lon0={lon0:.5f}  lat0={lat0:.5f}  "
            f"sx={sx*1e5:.3f}e-5  sy={sy*1e5:.3f}e-5 deg/px")
    cv2.putText(canvas, info,(4,PH-6),
                cv2.FONT_HERSHEY_SIMPLEX,.42,(0,230,120),1)
    return canvas


# ── GUI ───────────────────────────────────────────────────────
class App:
    PARAMS = [
        # (label, key, fmt_fn, step)
        ('lon0', 'lon0', lambda v: f"{v:.6f}", 0.00001),
        ('lat0', 'lat0', lambda v: f"{v:.6f}", 0.00001),
        ('sx (×1e-5)', 'sx', lambda v: f"{v:.4f}", 0.001),
        ('sy (×1e-5)', 'sy', lambda v: f"{v:.4f}", 0.001),
    ]
    STEP_MULTS = [1, 10, 100, 1000]

    def __init__(self, root):
        self.root = root
        root.title("Manual Alignment Tool v3")
        root.configure(bg='#1e1e2e')
        self._aid = None
        self._drag_start = None
        self._drag_lon0 = None
        self._drag_lat0 = None

        self.vars = {k: tk.DoubleVar(value=INIT[k]) for k in INIT}
        self.grid_var = tk.BooleanVar(value=True)
        self.lock_aspect = tk.BooleanVar(value=True)
        self.step_mult = tk.IntVar(value=1)
        self.entries = {}
        self._build()
        self._redraw()

    # ── Build UI ──────────────────────────────────────────
    def _build(self):
        # Canvas
        self.cnv = tk.Canvas(self.root, width=PW, height=PH,
                             bg='#000', highlightthickness=0, cursor='fleur')
        self.cnv.grid(row=0, column=0, columnspan=4, padx=6, pady=(6,2))
        self.cnv.bind('<ButtonPress-1>', self._on_drag_start)
        self.cnv.bind('<B1-Motion>', self._on_drag_motion)
        self.cnv.bind('<ButtonRelease-1>', self._on_drag_end)
        self.cnv.bind('<MouseWheel>', self._on_scroll)
        self.cnv.bind('<Button-4>', lambda e: self._on_scroll_linux(e, 1))
        self.cnv.bind('<Button-5>', lambda e: self._on_scroll_linux(e, -1))

        # Parameter rows: label | entry | − | +
        frm = tk.Frame(self.root, bg='#1e1e2e')
        frm.grid(row=1, column=0, columnspan=4, padx=6, sticky='ew')

        for r, (lbl, key, fmt_fn, step) in enumerate(self.PARAMS):
            tk.Label(frm, text=lbl, bg='#1e1e2e', fg='#cdd6f4',
                     font=('Consolas',10), width=12, anchor='e'
                     ).grid(row=r, column=0, padx=(6,2), pady=2)

            ent = tk.Entry(frm, width=16, font=('Consolas',11),
                           bg='#313244', fg='#a6e3a1', insertbackground='#a6e3a1',
                           relief='flat', justify='right')
            ent.grid(row=r, column=1, padx=2, pady=2)
            ent.insert(0, fmt_fn(INIT[key]))
            ent.bind('<Return>', lambda e, k=key: self._entry_commit(k))
            ent.bind('<FocusOut>', lambda e, k=key: self._entry_commit(k))
            self.entries[key] = ent

            fb = tk.Frame(frm, bg='#1e1e2e')
            fb.grid(row=r, column=2, padx=2)
            tk.Button(fb, text='−', width=2, bg='#f38ba8', fg='white',
                      relief='flat', font=('Arial',10,'bold'),
                      command=lambda k=key, s=step: self._fine(k, -1, s)
                      ).pack(side='left', padx=1)
            tk.Button(fb, text='+', width=2, bg='#a6e3a1', fg='#1e1e2e',
                      relief='flat', font=('Arial',10,'bold'),
                      command=lambda k=key, s=step: self._fine(k, +1, s)
                      ).pack(side='left', padx=1)

            # Slider (coarse)
            rmin = INIT[key] * 0.95 if key in ('sx','sy') else INIT[key] - 0.02
            rmax = INIT[key] * 1.50 if key in ('sx','sy') else INIT[key] + 0.02
            sl = ttk.Scale(frm, from_=rmin, to=rmax,
                           variable=self.vars[key], orient='horizontal',
                           length=360,
                           command=lambda v, k=key: self._slide(k))
            sl.grid(row=r, column=3, padx=4, sticky='ew')

        # Step multiplier row
        sm_frm = tk.Frame(self.root, bg='#1e1e2e')
        sm_frm.grid(row=2, column=0, columnspan=4, pady=(4,2))
        tk.Label(sm_frm, text='Step:', bg='#1e1e2e', fg='#cdd6f4',
                 font=('Consolas',10)).pack(side='left', padx=(8,4))
        for mult in self.STEP_MULTS:
            tk.Radiobutton(sm_frm, text=f'×{mult}', value=mult,
                           variable=self.step_mult,
                           bg='#1e1e2e', fg='#cdd6f4', selectcolor='#89b4fa',
                           activebackground='#1e1e2e', activeforeground='#89b4fa',
                           font=('Consolas',10,'bold'),
                           indicatoron=0, width=5, relief='flat',
                           ).pack(side='left', padx=2)

        tk.Checkbutton(sm_frm, text='Lock aspect', variable=self.lock_aspect,
                       bg='#1e1e2e', fg='#cdd6f4', selectcolor='#313244',
                       activebackground='#1e1e2e',
                       font=('Consolas',9)).pack(side='left', padx=(16,4))
        tk.Checkbutton(sm_frm, text='Grid', variable=self.grid_var,
                       bg='#1e1e2e', fg='#cdd6f4', selectcolor='#313244',
                       activebackground='#1e1e2e',
                       font=('Consolas',9),
                       command=self._now).pack(side='left', padx=4)

        # Action buttons
        ctrl = tk.Frame(self.root, bg='#1e1e2e')
        ctrl.grid(row=3, column=0, columnspan=4, pady=4)

        for txt, cmd, bg_c in [
            ('Zoom Map', self._zoom_map, '#313244'),
            ('Zoom GeoJSON', self._zoom_geojson, '#313244'),
            ('Reset', self._reset, '#313244'),
            ('In tham so', self._print, '#313244'),
            ('Luu tham so', self._save, '#89b4fa'),
        ]:
            tk.Button(ctrl, text=txt, bg=bg_c,
                      fg='#1e1e2e' if bg_c=='#89b4fa' else '#cdd6f4',
                      relief='flat', padx=10, font=('Arial',9,'bold'),
                      command=cmd).pack(side='left', padx=4)

        # Help label
        tk.Label(self.root,
                 text='Drag: pan | Scroll: zoom | Entry: type exact value + Enter | Arrow keys: fine adjust',
                 bg='#181825', fg='#6c7086', font=('Consolas',8), anchor='w', padx=8
                 ).grid(row=4, column=0, columnspan=4, sticky='ew')

        # Status bar
        self.status = tk.StringVar(value='')
        tk.Label(self.root, textvariable=self.status,
                 bg='#181825', fg='#f9e2af',
                 font=('Consolas',9), anchor='w', padx=8
                 ).grid(row=5, column=0, columnspan=4, sticky='ew')

        # Keyboard bindings
        self.root.bind('<Left>',  lambda e: self._arrow('lon0', -1))
        self.root.bind('<Right>', lambda e: self._arrow('lon0', +1))
        self.root.bind('<Up>',    lambda e: self._arrow('lat0', +1))
        self.root.bind('<Down>',  lambda e: self._arrow('lat0', -1))

    # ── Mouse drag to pan ─────────────────────────────────
    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_lon0 = self.vars['lon0'].get()
        self._drag_lat0 = self.vars['lat0'].get()

    def _on_drag_motion(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        sx = self.vars['sx'].get() * 1e-5
        sy = self.vars['sy'].get() * 1e-5
        # Convert pixel drag to geo shift
        dlon = -dx / PREVIEW_SCALE_X * sx
        dlat = dy / PREVIEW_SCALE_Y * sy
        self.vars['lon0'].set(self._drag_lon0 + dlon)
        self.vars['lat0'].set(self._drag_lat0 + dlat)
        self._sync_entries('lon0')
        self._sync_entries('lat0')
        self._schedule()

    def _on_drag_end(self, event):
        self._drag_start = None

    # ── Mouse scroll to zoom ───────────────────────────────
    def _on_scroll(self, event):
        factor = 1.05 if event.delta > 0 else 1/1.05
        self._zoom(factor, event.x, event.y)

    def _on_scroll_linux(self, event, direction):
        factor = 1.05 if direction > 0 else 1/1.05
        self._zoom(factor, event.x, event.y)

    def _zoom(self, factor, px, py):
        """Zoom around cursor position (px, py in preview coords)."""
        sx_old = self.vars['sx'].get() * 1e-5
        sy_old = self.vars['sy'].get() * 1e-5
        lon0 = self.vars['lon0'].get()
        lat0 = self.vars['lat0'].get()

        # Geo coord under cursor
        cursor_lon = lon0 + (px / PREVIEW_SCALE_X) * sx_old
        cursor_lat = lat0 - (py / PREVIEW_SCALE_Y) * sy_old

        sx_new = sx_old * factor
        sy_new = sy_old * factor if self.lock_aspect.get() else sy_old * factor

        # Adjust origin so cursor stays fixed
        new_lon0 = cursor_lon - (px / PREVIEW_SCALE_X) * sx_new
        new_lat0 = cursor_lat + (py / PREVIEW_SCALE_Y) * sy_new

        self.vars['sx'].set(sx_new * 1e5)
        self.vars['sy'].set(sy_new * 1e5)
        self.vars['lon0'].set(new_lon0)
        self.vars['lat0'].set(new_lat0)
        for k in ('lon0','lat0','sx','sy'):
            self._sync_entries(k)
        self._schedule()

    # ── Arrow keys ────────────────────────────────────────
    def _arrow(self, key, sign):
        step = 0.00001 * self.step_mult.get()
        v = self.vars[key].get() + sign * step
        self.vars[key].set(v)
        self._sync_entries(key)
        self._now()

    # ── Entry fields ──────────────────────────────────────
    def _entry_commit(self, key):
        try:
            v = float(self.entries[key].get())
            self.vars[key].set(v)
            self._now()
        except ValueError:
            self._sync_entries(key)

    def _sync_entries(self, key):
        """Update entry widget to match current var value."""
        ent = self.entries[key]
        fmt_fn = None
        for _, k, fn, _ in self.PARAMS:
            if k == key:
                fmt_fn = fn
                break
        ent.delete(0, tk.END)
        ent.insert(0, fmt_fn(self.vars[key].get()))

    # ── Slider ───────────────────────────────────────────
    def _slide(self, key):
        self._sync_entries(key)
        self._schedule()

    # ── Fine +/− ─────────────────────────────────────────
    def _fine(self, key, sign, base_step):
        mult = self.step_mult.get()
        old_val = self.vars[key].get()
        new_val = old_val + sign * base_step * mult
        self.vars[key].set(new_val)
        if self.lock_aspect.get() and key == 'sx' and old_val != 0:
            ratio = self.vars['sy'].get() / old_val
            self.vars['sy'].set(new_val * ratio)
            self._sync_entries('sy')
        self._sync_entries(key)
        self._now()

    # ── Scheduling ───────────────────────────────────────
    def _schedule(self):
        if self._aid: self.root.after_cancel(self._aid)
        self._aid = self.root.after(50, self._now)

    def _now(self):
        self._aid = None
        self._redraw()

    def _redraw(self):
        p = self._params()
        img = render(p['lon0'], p['lat0'], p['sx'] * 1e-5, p['sy'] * 1e-5,
                     self.grid_var.get())
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.cnv.delete('all')
        self.cnv.create_image(0, 0, anchor='nw', image=self._tk)

        # Map extent in geo
        sx, sy = p['sx']*1e-5, p['sy']*1e-5
        map_lon_max = p['lon0'] + sx * IMG_W
        map_lat_min = p['lat0'] - sy * IMG_H
        self.status.set(
            f"Map: lon[{p['lon0']:.5f}, {map_lon_max:.5f}] "
            f"lat[{map_lat_min:.5f}, {p['lat0']:.5f}]  |  "
            f"GeoJSON: lon[{GJ['lon_min']:.4f}, {GJ['lon_max']:.4f}] "
            f"lat[{GJ['lat_min']:.4f}, {GJ['lat_max']:.4f}]"
        )

    def _params(self):
        return {k: self.vars[k].get() for k in self.vars}

    # ── Zoom to fit ───────────────────────────────────────
    def _zoom_map(self):
        """Reset to initial values (map fills preview)."""
        for k in INIT:
            self.vars[k].set(INIT[k])
            self._sync_entries(k)
        self._now()

    def _zoom_geojson(self):
        """Zoom out so entire GeoJSON boundary fits in preview."""
        margin = 1.1  # 10% margin
        lon_span = (GJ['lon_max'] - GJ['lon_min']) * margin
        lat_span = (GJ['lat_max'] - GJ['lat_min']) * margin
        sx = lon_span / IMG_W
        sy = lat_span / IMG_H
        # Use the larger scale to ensure everything fits
        s = max(sx, sy)
        cx = (GJ['lon_min'] + GJ['lon_max']) / 2
        cy = (GJ['lat_min'] + GJ['lat_max']) / 2
        lon0 = cx - s * IMG_W / 2
        lat0 = cy + s * IMG_H / 2
        self.vars['lon0'].set(lon0)
        self.vars['lat0'].set(lat0)
        self.vars['sx'].set(s * 1e5)
        self.vars['sy'].set(s * 1e5)
        for k in ('lon0','lat0','sx','sy'):
            self._sync_entries(k)
        self._now()

    def _reset(self):
        """Reset to initial calibration values."""
        for k in INIT:
            self.vars[k].set(INIT[k])
            self._sync_entries(k)
        self._now()

    # ── Export ───────────────────────────────────────────
    def _print(self):
        p = self._params()
        sx = p['sx']*1e-5; sy = p['sy']*1e-5
        print("\n" + "═"*55)
        print("Tham so alignment hien tai:")
        print(f"  lon0    = {p['lon0']:.8f}")
        print(f"  lat0    = {p['lat0']:.8f}")
        print(f"  scale_x = {sx:.10f}  # deg lon / px")
        print(f"  scale_y = {sy:.10f}  # deg lat / px")
        print(f"\n# Dung trong draw_boundary.py:")
        print(f"  def geo_to_pdf(lon, lat):")
        print(f"      xi = (lon - {p['lon0']:.8f}) / {sx:.10f}")
        print(f"      yi = ({p['lat0']:.8f} - lat) / {sy:.10f}")
        print(f"      return xi * 72/150, yi * 72/150")
        print("═"*55)

    def _save(self):
        p = self._params()
        sx = p['sx']*1e-5; sy = p['sy']*1e-5
        out = {'lon0': p['lon0'], 'lat0': p['lat0'],
               'scale_x': sx, 'scale_y': sy,
               'note': 'lon at px x = lon0 + x*scale_x; lat at px y = lat0 - y*scale_y'}
        with open(OUT_PARAMS,'w',encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        self._print()
        messagebox.showinfo("Da luu",
            f"Da luu -> {OUT_PARAMS}\n\n"
            f"lon0    = {p['lon0']:.8f}\n"
            f"lat0    = {p['lat0']:.8f}\n"
            f"scale_x = {sx:.10f}\n"
            f"scale_y = {sy:.10f}")


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
