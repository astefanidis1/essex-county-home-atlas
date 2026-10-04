"""Step 4: homes, house pins on their building outlines, the land-value model, and records that aren't really homes.
Writes data/processed.pkl and prints diagnostics."""
import json, math, re, sys, pickle
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

import town_home_values as T

from paths import D
from towns import TOWNS, slug
SQFT_PER_M2 = 10.7639

raw = {t: json.load(open(D + f"{slug(t)}_parcels.json")) for t in TOWNS}
# lot sizes in acres everywhere (Newburyport's assessor reports square feet)
for t in TOWNS:
    sqft_town = sum(f["attributes"].get("LOT_UNITS") == "Sq. Ft." for f in raw[t]) > len(raw[t]) / 2
    for f in raw[t]:
        a = f["attributes"]
        if a.get("LOT_SIZE") is not None and (a.get("LOT_UNITS") == "Sq. Ft." or (a.get("LOT_UNITS") is None and sqft_town)):
            a["LOT_SIZE"], a["LOT_UNITS"] = a["LOT_SIZE"] / 43560.0, "Acres"
bld = {t: json.load(open(D + f"{slug(t)}_bldgs.json")) for t in TOWNS}

# ---------- one regional projection (meters) ----------
lons = [p[0] for t in TOWNS for f in raw[t] if f.get("geometry") for r in f["geometry"]["rings"] for p in r]
lats = [p[1] for t in TOWNS for f in raw[t] if f.get("geometry") for r in f["geometry"]["rings"] for p in r]
LON0, LAT0 = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
MX = 111412.84 * math.cos(math.radians(LAT0)) - 93.5 * math.cos(math.radians(3 * LAT0))
MY = 111132.954 - 559.822 * math.cos(math.radians(2 * LAT0)) + 1.175 * math.cos(math.radians(4 * LAT0))
to_m = lambda lon, lat: ((lon - LON0) * MX, (lat - LAT0) * MY)
to_ll = lambda x, y: (LON0 + x / MX, LAT0 + y / MY)

def rings_to_geom(rings):
    polys = []
    for r in rings:
        pts = [to_m(*p) for p in r]
        if len(pts) >= 4:
            pg = Polygon(pts)
            if not pg.is_valid:
                pg = pg.buffer(0)
            if not pg.is_empty:
                polys.append((pg, pg.exterior.is_ccw if hasattr(pg, "exterior") else True))
    if not polys:
        return None
    # ArcGIS: exterior rings clockwise, holes counter-clockwise (in y-up space)
    ext = [p for p, ccw in polys if not ccw] or [p for p, _ in polys]
    holes = [p for p, ccw in polys if ccw and any(e.contains(p.representative_point()) for e in ext)]
    g = unary_union(ext)
    if holes:
        g = g.difference(unary_union(holes))
    return g if not g.is_empty else None

rows_all, parcels_all, blds_all = [], [], []
for town in TOWNS:
    feats = raw[town]
    df = pd.DataFrame([f["attributes"] for f in feats])
    geom_by_loc = {}
    for f in feats:
        a, g = f["attributes"], f.get("geometry")
        k = a.get("LOC_ID") or f"_obj{a['OBJECTID']}"
        if g and k not in geom_by_loc:
            gg = rings_to_geom(g["rings"])
            if gg is not None:
                geom_by_loc[k] = gg
    for k, g in geom_by_loc.items():
        parcels_all.append({"town": town, "loc": k, "geom": g})
    homes = T.rank_homes(town, feats)
    homes["TOWN"] = town
    fy = int(df.FY.dropna().mode()[0])
    homes["FYV"] = fy
    # deed group sizes (all parcel records in town, distinct LOC_IDs)
    df["DEED"] = (df.LS_BOOK.fillna("").astype(str).str.lstrip("0") + "/" + df.LS_PAGE.fillna("").astype(str).str.lstrip("0"))
    df.loc[df.LS_BOOK.isna() | df.LS_PAGE.isna() | (df.LS_BOOK.astype(str).str.strip() == "") , "DEED"] = ""
    homes = homes.merge(df[["PROP_ID", "DEED"]], on="PROP_ID", how="left")
    rows_all.append(homes)
    for b in bld[town]:
        g = b.get("geometry")
        if not g:
            continue
        ring = max(g["rings"], key=len)
        pts = [to_m(*p) for p in ring]
        if len(pts) < 4:
            continue
        pg = Polygon(pts)
        if not pg.is_valid:
            pg = pg.buffer(0)
        if pg.is_empty:
            continue
        blds_all.append({"id": b["attributes"]["STRUCT_ID"], "geom": pg, "sqft": pg.area * SQFT_PER_M2, "town": town})

H = pd.concat(rows_all, ignore_index=True)
print("homes:", len(H), H.groupby("TOWN").size().to_dict(), "| parcels:", len(parcels_all), "| buildings:", len(blds_all))

# ---------- towns whose "living area" field holds gross area ----------
# MassGIS has living area (RES_AREA) and gross building area (BLD_AREA). Some towns fill both with one number. Usually
# that's living area, but Salem's is gross area (basement, attic and porches counted): about 460 sq ft per room against
# about 300 elsewhere. There, living area is estimated from the gross area, using the living-to-gross ratio of the same
# kind of house (one or several units, story count, age) in the towns that record both.
def floors_of(v):
    m = re.match(r"^(\d+(?:\.\d+)?)([AHTB]?)$", str(v).strip().upper())
    if not m or float(m.group(1)) <= 0: return np.nan
    return min(4.0, float(m.group(1)) + {"A": 0.5, "H": 0.5, "T": 0.75, "B": 0, "": 0}[m.group(2)])
_fl = H.STORIES.map(floors_of)
_yb = H.YEAR_BUILT.fillna(0)
KEYS = pd.DataFrame({"g": np.select([H.TYPE == "Multiple houses", H.TYPE.isin(["Two-family", "Three-family"])], ["several", "multi"], "single"),
                     "f": np.select([_fl.isna(), _fl <= 1.25, _fl <= 1.75, _fl <= 2.25], ["?", "1", "1.5", "2"], "3"),
                     "e": np.select([_yb <= 0, _yb < 1850, _yb < 1900, _yb < 1940, _yb < 1980], ["?", "<1850", "1850-99", "1900-39", "1940-79"], "1980+")},
                    index=H.index)
_has = (H.RES_AREA.fillna(0) > 0) & (H.TYPE != "Condo")
copy_t = ((H.BLD_AREA == H.RES_AREA) & _has).groupby(H.TOWN).sum() / _has.groupby(H.TOWN).sum().clip(lower=1) >= 0.9
_sf = H[(H.TYPE == "Single family") & (H.RES_AREA > 0) & (H.NUM_ROOMS > 0)]
per_room = (_sf.RES_AREA / _sf.NUM_ROOMS).groupby(_sf.TOWN).median()
typ_room = per_room[[not copy_t.get(t, False) for t in per_room.index]].median()
GROSS_ONLY = [t for t in TOWNS if copy_t.get(t, False) and per_room.get(t, 0) > 1.3 * typ_room]
print("sq ft per room (single family):", per_room.round(0).to_dict(), "| towns recording gross area as living area:", GROSS_ONLY)
ref = _has & (H.BLD_AREA > H.RES_AREA) & ~H.TOWN.map(copy_t).fillna(False).astype(bool)
kr = (H.RES_AREA / H.BLD_AREA)[ref]
kk = pd.Series(np.nan, index=H.index)
for ks in (["g", "f", "e"], ["g", "f"], ["g"]):
    t = kr.groupby([KEYS.loc[ref, k] for k in ks]).agg(["median", "size"])
    med = t["median"].where(t["size"] >= 40)
    kk = kk.fillna(pd.Series([med.get(tuple(r) if len(ks) > 1 else r[0], np.nan) for r in KEYS[ks].itertuples(index=False)], index=H.index))
gm = H.TOWN.isin(GROSS_ONLY) & _has
H["GROSS"] = H.RES_AREA.where(gm)
H.loc[gm, "BLD_AREA"] = H.loc[gm, "GROSS"]
H.loc[gm, "RES_AREA"] = (H.loc[gm, "GROSS"] * kk[gm] / 10).round() * 10
H["GROSS_K"] = kk.where(gm)
if gm.any():
    g2 = H[gm & (H.TYPE == "Single family") & (H.NUM_ROOMS > 0)]
    print("  converted", int(gm.sum()), "records; living share used: median", round(float(kk[gm].median()), 3), "range", round(float(kk[gm].min()), 3), "-", round(float(kk[gm].max()), 3),
          "| sq ft per room now", round(float((g2.RES_AREA / g2.NUM_ROOMS).median())))

# de-duplicate buildings that appear in two towns' pulls (border buildings)
seen, B = set(), []
for b in blds_all:
    if b["id"] in seen:
        continue
    seen.add(b["id"]); B.append(b)
print("unique buildings:", len(B))

# ---------- assign buildings to parcels ----------
P = parcels_all
ptree = STRtree([p["geom"] for p in P])
bld_by_parcel = {}
for bi, b in enumerate(B):
    rp = b["geom"].representative_point()
    for pi in ptree.query(rp, predicate="within"):
        bld_by_parcel.setdefault(int(pi), []).append(bi)
        break
ploc = {(p["town"], p["loc"]): i for i, p in enumerate(P)}

def stories(v):
    try:
        s = float(v)
        return s if s >= 1 else None
    except (TypeError, ValueError):
        return None

house_x, house_y, fp_sqft, method, pidx, pin_bi = [], [], [], [], [], []
for h in H.itertuples():
    pi = ploc.get((h.TOWN, h.LOC_ID))
    pidx.append(pi if pi is not None else -1)
    cands = [bi for bi in bld_by_parcel.get(pi, []) if B[bi]["sqft"] >= 400] if pi is not None else []
    if cands:
        if h.RES_AREA and not pd.isna(h.RES_AREA) and h.RES_AREA > 300 and h.TYPE != "Condo":
            st = stories(h.STORIES) or (2 if h.RES_AREA > 3000 else 1.5)
            expfp = h.RES_AREA / st * 1.25
            bi = min(cands, key=lambda i: abs(math.log(B[i]["sqft"] / expfp)))
        else:
            bi = max(cands, key=lambda i: B[i]["sqft"])
        g = B[bi]["geom"]
        c = g.centroid if g.contains(g.centroid) else g.representative_point()
        house_x.append(c.x); house_y.append(c.y); fp_sqft.append(round(B[bi]["sqft"])); method.append("bldg"); pin_bi.append(bi)
    elif pi is not None:
        c = P[pi]["geom"].representative_point()
        house_x.append(c.x); house_y.append(c.y); fp_sqft.append(0); method.append("parcel"); pin_bi.append(-1)
    else:
        house_x.append(np.nan); house_y.append(np.nan); fp_sqft.append(0); method.append("none"); pin_bi.append(-1)
H["PIDX"], H["HX"], H["HY"], H["FP"], H["LOCM"] = pidx, house_x, house_y, fp_sqft, method

# lots whose outline misses the house: parcel lines and building outlines don't always line up, so a house can
# straddle its lot line with its center just outside. Take the building that overlaps the lot most, as long as
# no other home's lot already claims it.
btree = STRtree([b["geom"] for b in B])
claimed = set()
for pi, bl in bld_by_parcel.items():
    claimed.update(bl)
homes_on = set(int(p) for p in H.PIDX if p >= 0)
home_claimed = set(b for pi, bl in bld_by_parcel.items() if pi in homes_on for b in bl)
fixed = 0
for i in [i for i, m in enumerate(method) if m == "parcel"]:
    pi = pidx[i]
    if pi is None or pi < 0:
        continue
    g = P[pi]["geom"]
    cands = []
    for bi in btree.query(g):
        bi = int(bi)
        if bi in home_claimed or B[bi]["sqft"] < 400:
            continue
        inter = B[bi]["geom"].intersection(g).area * SQFT_PER_M2
        if inter >= 250 and inter >= 0.15 * B[bi]["sqft"]:
            cands.append((bi, inter))
    if not cands:
        continue
    h = H.iloc[i]
    if h.RES_AREA and not pd.isna(h.RES_AREA) and h.RES_AREA > 300 and h.TYPE != "Condo":
        st = stories(h.STORIES) or (2 if h.RES_AREA > 3000 else 1.5)
        expfp = h.RES_AREA / st * 1.25
        bi = min(cands, key=lambda c: abs(math.log(B[c[0]]["sqft"] / expfp)))[0]
    else:
        bi = max(cands, key=lambda c: B[c[0]]["sqft"])[0]
    gb = B[bi]["geom"]; c = gb.centroid if gb.contains(gb.centroid) else gb.representative_point()
    H.iat[i, H.columns.get_loc("HX")], H.iat[i, H.columns.get_loc("HY")] = c.x, c.y
    H.iat[i, H.columns.get_loc("FP")], H.iat[i, H.columns.get_loc("LOCM")] = round(B[bi]["sqft"]), "bldg"
    bld_by_parcel.setdefault(pi, []).append(bi)          # so later steps see it as this lot's building
    home_claimed.add(bi); fixed += 1; pin_bi[i] = bi
print("pins moved onto a house straddling its lot line:", fixed)

# homes with no parcel shape: LOC_ID encodes the parcel centroid in state-plane units ("F_" feet / "M_" meters);
# fit a local affine transform from other parcels' LOC_IDs to our projection, then snap to the nearest building
miss = [i for i, m in enumerate(method) if m == "none"]
if miss:
    rows = []
    for pp in P:
        m = re.match(r"^([FM])_(\d+)_(\d+)$", pp["loc"] or "")
        if m:
            c = pp["geom"].centroid
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)), c.x, c.y))
    for i in miss:
        m = re.match(r"^([FM])_(\d+)_(\d+)$", H.at[i, "LOC_ID"] or "")
        if not m:
            continue
        pts = np.array([(a, b, x, y) for u, a, b, x, y in rows if u == m.group(1)])
        X = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        cx, *_ = np.linalg.lstsq(X, pts[:, 2], rcond=None); cy, *_ = np.linalg.lstsq(X, pts[:, 3], rcond=None)
        gx, gy = np.array([float(m.group(2)), float(m.group(3)), 1.0]) @ cx, np.array([float(m.group(2)), float(m.group(3)), 1.0]) @ cy
        near = sorted(((b["geom"].centroid.distance(Point(gx, gy)), bi) for bi, b in enumerate(B) if b["sqft"] >= 400))[:1]
        if near and near[0][0] < 60:
            g = B[near[0][1]]["geom"]; c = g.centroid if g.contains(g.centroid) else g.representative_point()
            H.at[i, "HX"], H.at[i, "HY"], H.at[i, "FP"], H.at[i, "LOCM"] = c.x, c.y, round(B[near[0][1]]["sqft"]), "bldg"; pin_bi[i] = near[0][1]
        else:
            H.at[i, "HX"], H.at[i, "HY"], H.at[i, "LOCM"] = gx, gy, "parcel"
        print("placed from LOC_ID:", H.at[i, "ADDRESS"], "->", H.at[i, "LOCM"], "| nearest building m:", round(near[0][0], 1) if near else None)
print("house location method:", H.LOCM.value_counts().to_dict())
# a house that runs across its lot line (a double house, or lot lines that don't match the outlines) can have its middle in
# the neighbor's lot: put the pin on the part of the house inside its own lot
moved = 0
for i, bi in enumerate(pin_bi):
    pi = pidx[i]
    if bi < 0 or pi is None or pi < 0 or H.at[i, "LOCM"] != "bldg":
        continue
    lot = P[pi]["geom"]
    if lot.contains(Point(H.at[i, "HX"], H.at[i, "HY"])):
        continue
    part = B[bi]["geom"].intersection(lot)
    if not part.is_empty and part.area * SQFT_PER_M2 >= 100:
        c = part.representative_point(); H.at[i, "HX"], H.at[i, "HY"] = c.x, c.y; moved += 1
print("pins moved onto the part of the house inside its own lot:", moved)

# ---------- land-value model per town ----------
def code3(c):
    """3-digit DOR class: '0101' (zero-padded home) -> 101, '0130'/'1010' (4th-digit suffix) -> 013/101."""
    c = (c or "").strip().upper() if isinstance(c, str) else ""
    return c[1:4] if (len(c) == 4 and c[0] == "0" and c[1:4] in T.HOME_CODES) else c[:3]

EDGES = [0.05, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 25, 40, 70, 120, 250]
curves = {}
for town in TOWNS:
    df = pd.DataFrame([f["attributes"] for f in raw[town]])
    c3 = df.USE_CODE.map(code3)
    ref = df[(c3.isin(["101", "130"])) & (df.LOT_SIZE > 0.05) & (df.LAND_VAL > 0)].copy()
    pts = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        b = ref[(ref.LOT_SIZE >= lo) & (ref.LOT_SIZE < hi)]
        if len(b) >= 6:
            pts.append((b.LOT_SIZE.median(), b.LAND_VAL.median(), len(b)))
    ac = np.array([p[0] for p in pts]); lv = np.maximum.accumulate(np.array([p[1] for p in pts]))
    curves[town] = (ac, lv, pts)

def expected_land(town, acres):
    ac, lv, _ = curves[town]
    if acres <= ac[0]:
        return float(lv[0])
    if acres <= ac[-1]:
        return float(np.interp(math.log(acres), np.log(ac), lv))
    k = max(len(ac) - 3, 0)          # marginal $/acre from the upper part of the curve
    rate = (lv[-1] - lv[k]) / max(ac[-1] - ac[k], 1e-9)
    # past the town's own data, extra acres add at most $40K each: where big lots are all prime spots (Marblehead Neck),
    # the curve's top end reflects location, not acreage
    return float(lv[-1] + min(max(rate, 0), 40_000) * (acres - ac[-1]))

for town in TOWNS:
    ac, lv, pts = curves[town]
    print(f"\n{town} land curve (median acres -> median land $, n):")
    print("  " + " | ".join(f"{a:.2f}ac ${v/1e3:,.0f}K n{n}" for (a, _, n), v in zip(pts, lv)))
    print("  extrapolated: 50ac $%.2fM, 100ac $%.2fM, 135ac $%.2fM" % tuple(expected_land(town, x) / 1e6 for x in (50, 100, 135)))

# records coded residential but with no building at all are land, not homes
nb = (H.BLDG_VAL.fillna(0) <= 0) & (H.RES_AREA.fillna(0) <= 0)
print("dropping building-less 'homes':", H[nb][["TOWN", "ADDRESS", "USE_CODE", "TOTAL_VAL", "LOT_SIZE"]].to_dict("records"))
H = H[~nb].reset_index(drop=True)

# mixed-use records that are apartment blocks or commercial buildings, not houses (common in Lynn), and commercial condos
NONHOUSE = r"APR?TMNT|APT-|APT |MULTI 111|MULTI-GRDN|MULT APT|WAREHOUSE|STORE|OFFICE|RESTAURANT|COMM|GARAGE|REPAIR|FUNERAL|RETAIL|INDUST"
sty = H.STYLE.fillna("").astype(str).str.upper()
mixed = H.TYPE.isin(["House + business", "House + other use"])
nh = sty.str.contains(r"CLUB", regex=True) | \
     (mixed & ((H.RES_AREA.fillna(0) >= 8000) | sty.str.contains(NONHOUSE, regex=True))) | \
     ((H.TYPE == "Condo") & (sty.str.contains(r"RETAIL|COMM|OFFICE|OFC|STORE|WAREHOUSE|INDUST", regex=True) | (H.RES_AREA.fillna(0) >= 10000)))
# a "multiple houses" record covering a colony of many small cottages (no single owner-occupied house) isn't one home
def colony(h):
    if h.TYPE != "Multiple houses" or h.PIDX < 0: return False
    fps = [B[b]["sqft"] for b in bld_by_parcel.get(int(h.PIDX), []) if B[b]["sqft"] >= 400]
    return len(fps) >= 10 and (h.RES_AREA or 0) >= 8 * max(fps)
col = pd.Series([colony(h) for h in H.itertuples()], index=H.index)
print("dropping cottage colonies:", H[col][["TOWN", "ADDRESS", "RES_AREA", "LOT_SIZE", "TOTAL_VAL"]].to_dict("records"))
# "multiple houses" records whose building style is an apartment block (Salem codes some that way)
aptblock = (H.TYPE == "Multiple houses") & sty.str.contains(r"\bAPT|APARTMENT", regex=True)
print("dropping apartment blocks coded as multiple houses:", H[aptblock][["TOWN", "ADDRESS", "STYLE", "RES_AREA", "UNITS"]].to_dict("records"))
nh = nh | col | aptblock
print("dropping apartment/commercial buildings:", int(nh.sum()), H[nh].groupby("TOWN").size().to_dict())
print(H[nh].sort_values("TOTAL_VAL", ascending=False)[["TOWN", "ADDRESS", "USE_CODE", "STYLE", "RES_AREA", "TOTAL_VAL"]].head(12).to_string(index=False))
H = H[~nh].reset_index(drop=True)

OPEN_CODES = {"012", "016", "017", "018"}
exp_land, adj, flag_land = [], [], []
for h in H.itertuples():
    acres = h.LOT_SIZE if (h.LOT_SIZE and not pd.isna(h.LOT_SIZE)) else 0
    land = h.LAND_VAL if not pd.isna(h.LAND_VAL) else 0
    if h.TYPE == "Condo" or acres < 5:          # farm/forest/recreation programs need 5+ acres (forest: 10)
        exp_land.append(np.nan); adj.append(h.TOTAL_VAL); flag_land.append(False); continue
    e = expected_land(h.TOWN, acres)
    coded = code3(h.USE_CODE) in OPEN_CODES
    f = (coded and land < e) or (land < 0.45 * e and (e - land) >= 150_000)
    exp_land.append(e); flag_land.append(f)
    adj.append(h.TOTAL_VAL - land + e if f else h.TOTAL_VAL)
H["EXPLAND"], H["ADJ"], H["F_LAND"] = exp_land, adj, flag_land
H["ROUGH"] = H.F_LAND & (H.LOT_SIZE > H.TOWN.map(lambda t: curves[t][0][-1]))

print("\nLand-flagged homes:", int(H.F_LAND.sum()), H[H.F_LAND].groupby("TOWN").size().to_dict())
cols = ["TOWN", "RANK", "ADDRESS", "USE_CODE", "TOTAL_VAL", "LAND_VAL", "EXPLAND", "ADJ", "LOT_SIZE", "RES_AREA", "LS_DATE", "LS_PRICE"]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(H[H.F_LAND].sort_values("ADJ", ascending=False)[cols].head(25).to_string(index=False))
print("\nOpen-land-coded homes NOT adjusted:")
print(H[(H.TYPE == "House + open land") & ~H.F_LAND][cols].to_string(index=False))

pickle.dump({"H": H, "P": P, "B": B, "bld_by_parcel": bld_by_parcel, "LON0": LON0, "LAT0": LAT0, "MX": MX, "MY": MY,
             "curves": curves, "raw_attrs": {t: pd.DataFrame([f["attributes"] for f in raw[t]]) for t in TOWNS}},
            open(D + "processed.pkl", "wb"))
print("\nsaved processed.pkl")
