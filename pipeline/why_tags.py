"""Step 10: why a home is worth what it is: split its value into land and house, each against what's typical in its town.

  acreage   land a typical lot of this size would carry, above a typical house lot
  location  land value above what the lot's size explains (waterfront, views, a prime street)
  size      house value a typical house of this size would carry, above a typical house
  quality   house value above what its size explains (new or luxury construction, historic, extra buildings)

The biggest part is the home's tag (WHY1); a second part at least half as big is a second tag (WHY2).
Waterfront: the lot is within 12 m of open water (ocean, bay, harbor, tidal flat, lake or pond of 5+ acres,
reservoir) or 8 m of a named river, per MassDEP 1:25,000 hydrography.
"""
import json, math, pickle, sys
import numpy as np, pandas as pd
from shapely.geometry import shape
from shapely.ops import transform
from shapely.strtree import STRtree
from paths import D
from towns import TOWNS
S = pickle.load(open(D + "final6.pkl", "rb"))
H, P, curves = S["H"].copy(), S["P"], S["curves"]
LON0, LAT0, MX, MY = S["LON0"], S["LAT0"], S["MX"], S["MY"]
pd.set_option("display.width", 320); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 22)
F_LAND, F_MULTI, F_SALE, F_HIGHV = 1, 4, 256, 16384
FYEAR = 2026

# ---------- waterfront ----------
# The 1:25,000 shoreline sits a few dozen meters off many rocky or marshy lot lines, so a lot counts as waterfront
# within 35 m of open water (15 m of a river line) when no mapped road runs between the lot and the water; otherwise
# a lot within 250 m of the ocean, a bay, or a harbor is "near the ocean" (across the road, or a street back).
from shapely.geometry import LineString
from shapely.ops import nearest_points
to_m = lambda x, y, z=None: ((np.asarray(x) - LON0) * MX, (np.asarray(y) - LAT0) * MY)
W = json.load(open(D + "water.json"))
polys, pcode, lines = [], [], []
for t, feats in W.items():
    for f in feats:
        g = transform(to_m, shape(f["geom"]))
        if f["code"] == "river": lines.append(g)
        else: polys.append(g); pcode.append(f["code"])
roads = []
for f in json.load(open(D + "roads.json")):
    for pth in (f.get("geometry") or {}).get("paths", []):
        if len(pth) >= 2: roads.append(LineString([to_m(x, y) for x, y in pth]))
tp, tl, tr = STRtree(polys), STRtree(lines), STRtree(roads)
def road_between(g, w):
    a, b = nearest_points(g, w)
    seg = LineString([a, b])
    if seg.length < 4: return False
    seg = LineString([seg.interpolate(3), b])                     # skip a road that only grazes the lot line
    return any(roads[j].intersects(seg) for j in tr.query(seg))
water, near_ocean, wdist = np.zeros(len(H), bool), np.zeros(len(H), bool), np.full(len(H), np.nan)
for k, h in enumerate(H.itertuples()):
    if h.PIDX < 0: continue
    g = P[int(h.PIDX)]["geom"]
    cands = [(polys[j].distance(g), polys[j], pcode[j]) for j in tp.query(g.buffer(250))]
    cands += [(lines[j].distance(g) + 20, lines[j], "river") for j in tl.query(g.buffer(15))]   # rivers: 15 m counts like 35 m
    cands.sort(key=lambda c: c[0])
    if cands: wdist[k] = cands[0][0]
    for d, w, c in cands:
        if d > 35: break
        if not road_between(g, w):
            water[k] = True; break
    if not water[k] and any(c in (7, 8) and d <= 250 for d, w, c in cands):
        near_ocean[k] = True
H["WATER"], H["NEAR_OCEAN"], H["WDIST"] = water, near_ocean, wdist
print("waterfront homes:", int(water.sum()), H[water].groupby("TOWN").size().to_dict())
print("near the ocean:", int(near_ocean.sum()), H[near_ocean].groupby("TOWN").size().to_dict())

# ---------- town baselines ----------
def expected_land(town, acres):
    ac, lv, _ = curves[town]
    if acres <= ac[0]: return float(lv[0])
    if acres <= ac[-1]: return float(np.interp(math.log(acres), np.log(ac), lv))
    k = max(len(ac) - 3, 0)
    rate = (lv[-1] - lv[k]) / max(ac[-1] - ac[k], 1e-9)
    return float(lv[-1] + min(max(rate, 0), 40_000) * (acres - ac[-1]))
SZ = [0, 1200, 1800, 2500, 3500, 5000, 7000, 1e9]
SIZE = np.where((H.FLAGS & 32768) > 0, H.CSF, H.RES_AREA).astype(float)
H["SIZE"] = SIZE
base = {}
for t in TOWNS:
    sf = H[(H.TOWN == t) & (H.TYPE == "Single family") & (H.SIZE > 400) & (H.BLDG_VAL > 20000) & (H.LOT_SIZE > 0)]
    a0, s0 = float(sf.LOT_SIZE.median()), float(sf.SIZE.median())
    band = pd.cut(sf.SIZE, SZ, labels=False)
    g = sf.assign(b=band, psf=sf.BLDG_VAL / sf.SIZE).groupby("b").agg(s=("SIZE", "median"), psf=("psf", "median"), n=("psf", "size"))
    g = g[g.n >= 12]
    cond = H[(H.TOWN == t) & (H.TYPE == "Condo") & (H.SIZE > 300) & (H.TOTAL_VAL > 0)]
    base[t] = {"A0": a0, "L0": expected_land(t, a0), "S0": s0, "curve_s": g.s.tolist(), "curve_psf": g.psf.tolist(),
               "C0": float(cond.TOTAL_VAL.median()) if len(cond) >= 20 else np.nan,
               "CS0": float(cond.SIZE.median()) if len(cond) >= 20 else np.nan,
               "CPSF": float((cond.TOTAL_VAL / cond.SIZE).median()) if len(cond) >= 20 else np.nan}
    b = base[t]; b["B0"] = float(np.interp(math.log(s0), np.log(b["curve_s"]), b["curve_psf"])) * s0
def house_at(t, s):
    b = base[t]
    return float(np.interp(math.log(max(s, 300)), np.log(b["curve_s"]), b["curve_psf"])) * s
print({t: {k: (round(v) if isinstance(v, float) and v == v else v) for k, v in b.items() if k not in ("curve_s", "curve_psf")} for t, b in base.items()})

# ---------- value parts ----------
LABELS = ["", "Waterfront", "Prime location", "Acreage", "Big house", "New build", "High-end build", "Historic", "Several houses",
          "Extra buildings", "Big unit", "Typical for town", "Near the ocean", "Sold high"]
L = {n: i for i, n in enumerate(LABELS)}
out = []
for h in H.itertuples():
    rec = (0, 0, 0.0, 0.0, 0.0, 0.0)
    if h.BEST < 1_000_000:
        out.append(rec); continue
    b, t = base[h.TOWN], h.TOWN
    if h.TYPE == "Condo":
        if b["CPSF"] != b["CPSF"] or not h.SIZE > 0:
            out.append((L["Typical for town"], 0, 0, 0, 0, 0)); continue
        expect = b["CPSF"] * h.SIZE
        size = max(expect - b["C0"], 0.0)
        loc = max(h.TOTAL_VAL - max(expect, b["C0"]), 0.0)
        parts = {"size": size, "loc": loc}
        lab = {"size": "Big unit", "loc": "Waterfront" if h.WATER else "Near the ocean" if h.NEAR_OCEAN else ("New build" if (h.YEAR_BUILT or 0) >= FYEAR - 11 else "Prime location")}
        acre = locp = sizep = qual = 0.0; sizep, locp = size, loc
    else:
        land = h.EXPLAND if (h.FLAGS & F_LAND) and h.EXPLAND == h.EXPLAND else h.LAND_VAL
        ac = h.LOT_SIZE if h.LOT_SIZE == h.LOT_SIZE and h.LOT_SIZE > 0 else b["A0"]
        exp_land = expected_land(t, ac)
        if ac < max(2.0, 3 * b["A0"]):
            # on a lot that isn't big, extra land value is about where it is (in coastal towns the bigger lots are the
            # prime ones, so the lot-size curve itself carries location)
            acre, locp = 0.0, max(land - b["L0"], 0.0)
        elif land >= exp_land: acre, locp = exp_land - b["L0"], land - exp_land
        else: acre, locp = max(land - b["L0"], 0.0), 0.0
        acre = max(acre, 0.0)
        bldg = h.BEST - land - (h.OTHER_VAL or 0) if h.FLAGS & 65536 else h.BLDG_VAL     # unfinished: its estimated finished value
        if h.SIZE > 0:
            exp_house = house_at(t, h.SIZE)
            if bldg >= exp_house: sizep, qual = exp_house - b["B0"], bldg - exp_house
            else: sizep, qual = max(bldg - b["B0"], 0.0), 0.0
            sizep = max(sizep, 0.0)
        else:
            sizep, qual = 0.0, max(bldg - b["B0"], 0.0)
        extra = (h.FLAGS & F_HIGHV and h.HIGHV_WHY == 1) or (h.LOCM == "bldg" and h.ROOF_LOT > 1.8 * h.FP and h.ROOF_LOT - h.FP >= 1500)
        yb = h.YEAR_BUILT if h.YEAR_BUILT == h.YEAR_BUILT else 0
        # a lot coded as several houses where one house is most of the roof on the lot (a mansion and its carriage house):
        # the value is about that house, so it's labeled like one
        several = bool(h.FLAGS & F_MULTI) and not (h.LOCM == "bldg" and h.ROOF_LOT > 0 and h.FP >= 0.7 * h.ROOF_LOT)
        qlab = ("Several houses" if several else "Extra buildings" if extra else "New build" if yb >= FYEAR - 11
                else "Historic" if 1630 <= yb < 1850 else "High-end build")
        parts = {"acre": acre, "loc": locp, "size": sizep, "qual": qual}
        # on 10+ acres, land value beyond what the size explains is more likely buildable land than a prime spot
        lab = {"acre": "Acreage", "loc": "Waterfront" if h.WATER else "Near the ocean" if h.NEAR_OCEAN else "Acreage" if ac >= 10 else "Prime location",
               "size": "Several houses" if several else "Big house", "qual": qlab}
    order = sorted(parts, key=lambda k: -parts[k])
    top = order[0]
    basis = h.BEST if h.FLAGS & 65536 else h.TOTAL_VAL
    pv = [parts.get(k, 0.0) for k in ("acre", "loc", "size", "qual")]
    if parts[top] < 0.15 * basis:
        # nothing stands out in the assessment: either its sale put it here, or it's what homes cost in this town
        sold = (h.FLAGS & F_SALE) and h.BEST - h.TOTAL_VAL >= 0.2 * h.BEST
        out.append((L["Sold high"] if sold else L["Typical for town"], 0, *pv)); continue
    second = order[1] if len(order) > 1 and parts[order[1]] >= 0.5 * parts[top] and parts[order[1]] >= 150000 and lab[order[1]] != lab[top] else None
    w2 = None
    # a waterfront lot says so even when the land isn't the biggest part
    if lab[top] != "Waterfront" and h.WATER and parts.get("loc", 0) >= 150000:
        w2 = "Waterfront"
    elif lab[top] != "Near the ocean" and h.NEAR_OCEAN and parts.get("loc", 0) >= 150000:
        w2 = "Near the ocean"
    out.append((L[lab[top]], L[w2] if w2 else (L[lab[second]] if second else 0), *pv))
H["WHY1"], H["WHY2"] = [o[0] for o in out], [o[1] for o in out]
H["P_ACRE"], H["P_LOC"], H["P_SIZE"], H["P_QUAL"] = [o[2] for o in out], [o[3] for o in out], [o[4] for o in out], [o[5] for o in out]
m = H.BEST >= 1_000_000
print("\nhomes at $1M+:", int(m.sum()))
print(pd.crosstab(H[m].WHY1.map(lambda i: LABELS[i]), H[m].TOWN).to_string())
print("\nsecond tags:", H[m & (H.WHY2 > 0)].WHY2.map(lambda i: LABELS[i]).value_counts().to_dict())
cols = ["TOWN", "ADDRESS", "TYPE", "SIZE", "LOT_SIZE", "YEAR_BUILT", "LAND_VAL", "BLDG_VAL", "BEST", "WATER", "P_ACRE", "P_LOC", "P_SIZE", "P_QUAL"]
for i, name in enumerate(LABELS):
    if not i: continue
    d = H[m & (H.WHY1 == i)].sort_values("BEST", ascending=False)
    if len(d):
        print(f"\n=== {name}: {len(d)}")
        print(d[cols + ["WHY2"]].head(8).round(0).to_string(index=False))
pickle.dump({**S, "H": H, "base": base, "why_labels": LABELS}, open(D + "final7.pkl", "wb"))
print("\nsaved final7.pkl")
