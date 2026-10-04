"""Aerial image chips (MassGIS 2025 imagery) with the parcel and state building outlines drawn on top, for eyeballing flagged homes."""
import io, math, pickle, sys, concurrent.futures as cf
import requests
from PIL import Image, ImageDraw, ImageFont
TILE = "https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/Massachusetts_Aerial_Imagery_2025/MapServer/tile/{z}/{y}/{x}"
R = 6378137.0
def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
_cache = {}
def tile(z, x, y):
    k = (z, x, y)
    if k not in _cache:
        for a in range(3):
            try:
                r = requests.get(TILE.format(z=z, x=x, y=y), timeout=30)
                if r.status_code == 200 and len(r.content) > 500:
                    _cache[k] = Image.open(io.BytesIO(r.content)).convert("RGB"); break
            except Exception:
                pass
        else:
            _cache[k] = Image.new("RGB", (256, 256), (40, 40, 40))
    return _cache[k]
def chip(lat, lon, half_m, size=340, rings=(), label=""):
    """rings: list of (list of (lon,lat), color, width)"""
    cx, cy = merc(lon, lat)
    k = 1 / math.cos(math.radians(lat))                       # mercator meters per ground meter
    hm = half_m * k
    z = 20
    while z > 14 and (2 * hm) / (156543.03392804097 / 2 ** z) > 3 * size: z -= 1
    res = 156543.03392804097 / 2 ** z
    x0, x1 = cx - hm, cx + hm; y0, y1 = cy - hm, cy + hm
    ox = -20037508.342787; oy = 20037508.342787
    tx0, tx1 = int((x0 - ox) / (256 * res)), int((x1 - ox) / (256 * res))
    ty0, ty1 = int((oy - y1) / (256 * res)), int((oy - y0) / (256 * res))
    W, Hh = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    img = Image.new("RGB", (W, Hh))
    jobs = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    with cf.ThreadPoolExecutor(8) as ex:
        for (tx, ty), im in zip(jobs, ex.map(lambda t: tile(z, t[0], t[1]), jobs)):
            img.paste(im, ((tx - tx0) * 256, (ty - ty0) * 256))
    px = lambda mx, my: ((mx - ox) / res - tx0 * 256, (oy - my) / res - ty0 * 256)
    d = ImageDraw.Draw(img)
    for ring, color, w in rings:
        pts = [px(*merc(lo, la)) for lo, la in ring]
        if len(pts) > 2: d.line(pts + [pts[0]], fill=color, width=w)
    a, b = px(x0, y1), px(x1, y0)
    img = img.crop((int(a[0]), int(a[1]), int(b[0]), int(b[1]))).resize((size, size))
    d = ImageDraw.Draw(img)
    # scale bar: 20 m
    bar = size * 20 / (2 * half_m)
    d.rectangle([8, size - 14, 8 + bar, size - 10], fill=(255, 255, 255))
    if label:
        d.rectangle([0, 0, size, 30], fill=(0, 0, 0))
        for n, line in enumerate(label.split("\n")[:2]):
            d.text((5, 2 + 13 * n), line, fill=(255, 255, 255))
    return img
def grid(chips, cols=3, path="grid.png"):
    size = chips[0].size[0]
    rows = math.ceil(len(chips) / cols)
    g = Image.new("RGB", (cols * size + (cols - 1) * 6, rows * size + (rows - 1) * 6), (255, 255, 255))
    for n, c in enumerate(chips):
        g.paste(c, ((n % cols) * (size + 6), (n // cols) * (size + 6)))
    g.save(path)
    return path
