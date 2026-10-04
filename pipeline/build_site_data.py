"""Step 11: page data for the atlas: one compact home index plus per-town parcel and footprint files.

Map units are meters: X to the east, Y downward (screen orientation), origin at the region's NW corner.
Every data file is gzip-compressed JSON wrapped as {"z": "<base64>"} so it downloads small on any host.
"""
import base64, gzip, json, math, pickle, re, sys
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from paths import D, SITE as OUT
from towns import TOWNS, slug

S = pickle.load(open(D + "final7.pkl", "rb"))
H, P, B, E, A = S["H"].copy(), S["P"], S["B"], S["E"], S["raw_attrs"]
LON0, LAT0, MX, MY = S["LON0"], S["LAT0"], S["MX"], S["MY"]
TI = {t: i for i, t in enumerate(TOWNS)}

# ---------- map frame ----------
allb = [p["geom"].bounds for p in P]
minx, miny = min(b[0] for b in allb), min(b[1] for b in allb)
maxx, maxy = max(b[2] for b in allb), max(b[3] for b in allb)
W, HH = math.ceil(maxx - minx), math.ceil(maxy - miny)
MXf = lambda x: int(round(x - minx))
MYf = lambda y: int(round(maxy - y))
to_ll = lambda x, y: (LON0 + x / MX, LAT0 + y / MY)

# per-town parcel numbering
local_pi, town_parcels = {}, {t: [] for t in TOWNS}
for gi, p in enumerate(P):
    local_pi[gi] = len(town_parcels[p["town"]])
    town_parcels[p["town"]].append(gi)

# ---------- homes in global value order ----------
H = H.sort_values(["BEST", "TOTAL_VAL"], ascending=False).reset_index(drop=False).rename(columns={"index": "OLDIDX"})
H["REG"] = np.arange(1, len(H) + 1)
H["TRANK"] = H.groupby("TOWN").BEST.rank(ascending=False, method="first").astype(int)
CODES = {"Single family": "S", "Condo": "C", "Mobile home": "H", "Two-family": "2", "Three-family": "3",
         "Multiple houses": "M", "House + business": "B", "House + open land": "F", "House + other use": "O"}

deed_n = {}
for town in TOWNS:
    df = A[town]
    k = df.LS_BOOK.fillna("").astype(str) + "/" + df.LS_PAGE.fillna("").astype(str)
    k = k.where(df.LS_BOOK.notna() & df.LS_PAGE.notna(), "nodeed:" + df.PROP_ID.fillna("").astype(str))
    n = df.assign(k=k).drop_duplicates("LOC_ID").groupby("k").size()
    deed_n.update({(town, pid): n.get(kk, 1) for pid, kk in zip(df.PROP_ID, k)})

BULK = S.get("bulk", set())
def sale(h):
    p, d = h.LS_PRICE, h.LS_DATE
    if pd.isna(p) or p < 50000 or not isinstance(d, str) or len(d) < 6 or deed_n.get((h.TOWN, h.PROP_ID), 1) > 1 or (h.TOWN, h.LOC_ID) in BULK:
        return 0, 0
    return int(d[:4]) * 12 + int(d[4:6]) - 1, int(p)          # months since year 0

LM_LIST, LM_IDX = [], {}
for lm in H.LANDMARK:
    if lm is not None and lm[0] + lm[1] not in LM_IDX:
        LM_IDX[lm[0] + lm[1]] = len(LM_LIST); LM_LIST.append([lm[0], lm[1], lm[2]])

styles, style_idx = [], {}
ucodes, ucode_idx = [], {}
def intern(v, arr, idx):
    if v not in idx:
        idx[v] = len(arr); arr.append(v)
    return idx[v]

n0 = lambda v: 0 if pd.isna(v) else int(round(v))
WORDS = {"street": "st", "road": "rd", "avenue": "ave", "av": "ave", "lane": "ln", "la": "ln", "drive": "dr", "circle": "cir",
         "ci": "cir", "cr": "cir", "court": "ct", "terrace": "ter", "place": "pl", "wy": "way", "highway": "hwy", "parkway": "pkwy",
         "turnpike": "tpke", "square": "sq", "mount": "mt", "cg": "crossing", "ph": "path", "rw": "row", "lndg": "landing",
         "cres": "crescent", "north": "n", "south": "s", "east": "e", "west": "w"}
def norm(t):
    t = t.lower().replace("`", "").replace("'", "").replace("½", " 1/2").replace("-", "").replace("#", " ").replace(".", "")
    return " ".join(WORDS.get(w, w) for w in t.split())
F_SIMP, F_NOPARCEL = 1024, 2048
F_LOW, F_SIZEBIG, F_SIZESML, F_HIGHV, F_SIZEFIX, F_VALEST = 64, 4096, 8192, 16384, 32768, 65536
SIZEF = F_LOW | F_SIZEBIG | F_SIZESML | F_HIGHV | F_SIZEFIX
sz_fix = lambda h: int(h.FLAGS) & F_SIZEFIX
cols = {k: [] for k in ["tw", "ad", "ty", "as", "dv", "bl", "la", "sf", "ac", "yb", "st", "sy", "sp", "pi", "uc", "pid",
                        "fl", "x", "y", "fp", "oy", "sb"]}
sparse = {"alt": {}, "exl": {}, "cr": {}, "line": {}, "lm": {}, "sz": {}, "gr": {}}     # gr: gross area where that's all the assessor records (Salem)
other_bldg = []                                        # building value includes barns or other houses on the lot
for n, h in enumerate(H.itertuples()):
    sy, sp = sale(h)
    cols["tw"].append(TI[h.TOWN])
    cols["ad"].append(h.ADDRESS)
    cols["ty"].append(CODES[h.TYPE])
    cols["as"].append(n0(h.TOTAL_VAL))
    cols["dv"].append(n0(h.BEST) - n0(h.TOTAL_VAL))
    cols["bl"].append(n0(h.BLDG_VAL)); cols["la"].append(n0(h.LAND_VAL))
    cols["sf"].append(n0(h.CSF) if sz_fix(h) else n0(h.RES_AREA))     # corrected size where the evidence agrees
    cols["ac"].append(round(float(h.LOT_SIZE), 2) if not pd.isna(h.LOT_SIZE) else 0)
    cols["yb"].append(n0(h.YEAR_BUILT) if n0(h.YEAR_BUILT) >= 1630 else 0)   # "1600" is an assessor placeholder, not a date
    other_bldg.append(bool(int(h.FLAGS) & F_HIGHV) and int(h.HIGHV_WHY) == 1)
    cols["st"].append(intern((h.STYLE if isinstance(h.STYLE, str) else "").strip().title(), styles, style_idx))
    cols["sy"].append(sy); cols["sp"].append(sp)
    cols["pi"].append(local_pi[int(h.PIDX)] if h.PIDX >= 0 else -1)
    cols["uc"].append(intern(h.USE_CODE if isinstance(h.USE_CODE, str) else "", ucodes, ucode_idx))
    cols["pid"].append(h.PROP_ID if isinstance(h.PROP_ID, str) else "")
    simp = not pd.isna(h.SALE_IMPL) and h.SALE_IMPL > 0
    if simp:
        r = S["ratio"][h.TOWN][sy // 12]
        assert sp > 0 and abs(sp / r - h.SALE_IMPL) < 1, (h.ADDRESS, sp, r, h.SALE_IMPL)
    cols["fl"].append(int(h.FLAGS) | (F_SIMP if simp else 0) | (F_NOPARCEL if h.PIDX < 0 else 0))
    cols["x"].append(MXf(h.HX)); cols["y"].append(MYf(h.HY))
    cols["fp"].append(n0(h.FP))
    cols["oy"].append(int(h.OUTYR) if h.FP > 0 else 0)
    if int(h.FLAGS) & SIZEF:
        # [typical house $/sq ft for similar homes, roof outlines on the lot, stories used, outline year, value agrees with a smaller house]
        sparse["sz"][n] = [n0(h.EXP_PSF) if not pd.isna(h.EXP_PSF) else 0, n0(h.ROOF_LOT), round(float(h.FLOORS), 2), int(h.OUTYR),
                           1 if bool(h.SIZEBIG_VALUE_AGREES) else 0, n0(h.FP),
                           # recorded living area, gross-area clue, outline clue, value clue, gross building area,
                           # correction kind (0 none, 1 all clues agree, 2 record predates the building), why the value looks high (1 other buildings, 2 clues say bigger, 3 no outline)
                           n0(h.RES_AREA), n0(h.C_BLD), n0(h.C_OUT), n0(h.C_VAL), n0(h.BLD_AREA),
                           (1 if h.FIXKIND == "R1" else 2) if sz_fix(h) else 0, int(h.HIGHV_WHY)]
    cols["sb"].append(int(round(h.SETBACK_M * 3.28084)) if h.SETBACK_M * 3.28084 >= 150 else 0)
    if not pd.isna(h.GROSS) and h.GROSS > 0:
        sparse["gr"][n] = int(h.GROSS)
    assessor = re.sub(r"\s+", " ", f"{h.ADDR_NUM if isinstance(h.ADDR_NUM, str) else ''} {h.FULL_STR if isinstance(h.FULL_STR, str) else ''}").strip().lower()
    if assessor and norm(assessor) != norm(h.ADDRESS):
        sparse["alt"][n] = norm(assessor)
    if h.F_LAND:
        sparse["exl"][n] = n0(h.EXPLAND)
    if h.CR_FRAC > 0.004:
        sparse["cr"][n] = [int(round(h.CR_FRAC * 100)), h.CR_NAME or ""]
    if h.LINE_TOWN:
        sparse["line"][n] = h.LINE_TOWN
    if h.LANDMARK is not None:
        sparse["lm"][n] = LM_IDX[h.LANDMARK[0] + h.LANDMARK[1]]
cols["ty"] = "".join(cols["ty"])
cols["wy"] = "".join(chr(97 + int(a)) + chr(97 + int(b)) for a, b in zip(H.WHY1, H.WHY2))   # why tags for $1M+ homes, 'a' = none
street_tab, street_idx, an, sn, sx = [], {}, [], [], {}
for n, a in enumerate(cols["ad"]):
    m = re.match(r"^(\S+)\s+(.*?)((?:\s+#.*|\s+\((?:lot|rear)\b.*\))?)$", a)
    num, stn, suf = (m.group(1), m.group(2), m.group(3).strip()) if (m and re.match(r"^\d", m.group(1))) else ("", a, "")
    assert (num + " " if num else "") + stn + (" " + suf if suf else "") == a, a
    an.append(num); sn.append(intern(stn, street_tab, street_idx))
    if suf:
        sx[n] = suf
homes_n = len(H)
V = lambda i: cols["as"][i] + cols["dv"][i]            # value used for ranking

# ---------- estates ----------
old_to_new = {int(o): n for n, o in enumerate(H.OLDIDX)}
estates, est_of = [], {}
for e in E.itertuples():
    members = sorted(old_to_new[int(oi)] for oi in e.homes)
    eid = len(estates)
    for m in members:
        est_of[m] = eid
    ym = (int(e.sale_date[:4]) * 12 + int(e.sale_date[4:6]) - 1) if isinstance(e.sale_date, str) and len(e.sale_date) >= 6 else 0
    estates.append([TI[e.town], old_to_new[int(e.main)], members, [local_pi[p] for p in e.pidx if p >= 0], int(e.n_parcels),
                    round(e.acres, 1), int(e.assessed), int(round(e.best)), ym, int(e.sale_price or 0), bool(e.rough),
                    int(round(e.impl)) if not pd.isna(e.impl) else 0])
cols["es"] = [est_of.get(i, -1) for i in range(homes_n)]

# ---------- towns ----------
towns, outlines = [], []
for t in TOWNS:
    geoms = [P[gi]["geom"] for gi in town_parcels[t]]
    u = unary_union([g.buffer(30) for g in geoms]).buffer(-30)
    polys = [u] if isinstance(u, Polygon) else sorted(u.geoms, key=lambda g: -g.area)
    polys = [p for p in polys if p.area >= 0.03 * polys[0].area]
    rings = []
    for pg in polys:
        ring = Polygon(pg.exterior).simplify(10).exterior.coords
        rings.append([v for x, y in ring for v in (MXf(x), MYf(y))])
    outlines.append(rings)
    b = unary_union(polys).bounds
    lp = polys[0].representative_point()
    idx = [i for i in range(homes_n) if cols["tw"][i] == TI[t]]
    vals = sorted(V(i) for i in idx)
    fy = int(A[t].FY.dropna().mode()[0])
    s_prev = [cols["sp"][i] / cols["as"][i] for i in idx if cols["sy"][i] and cols["sy"][i] // 12 == fy - 2 and cols["as"][i] > 0]
    towns.append({"name": t, "fy": fy, "valDate": f"Jan 1, {fy - 1}", "n": len(idx), "median": int(np.median(vals)),
                  "n2m": sum(v >= 2e6 for v in vals), "n3m": sum(v >= 3e6 for v in vals),
                  "bbox": [MXf(b[0]), MYf(b[3]), MXf(b[2]), MYf(b[1])], "label": [MXf(lp.x), MYf(lp.y)],
                  "ratio": round(float(np.median(s_prev)), 3) if len(s_prev) >= 20 else None,
                  "newPsf": round(S["new_psf"][t]) if S["new_psf"][t] == S["new_psf"][t] else None,
                  "lastSale": max((cols["sy"][i] for i in idx), default=0), "slug": slug(t),
                  "why": {k: (round(v) if v == v else 0) for k, v in (("a0", 0), ("l0", S["base"][t]["L0"]), ("s0", S["base"][t]["S0"]),
                                                                          ("cs0", S["base"][t]["CS0"]), ("cpsf", S["base"][t]["CPSF"]))} |
                         {"a0": round(S["base"][t]["A0"], 2), "lac": [round(float(x), 3) for x in S["curves"][t][0]],
                          "llv": [round(float(x)) for x in S["curves"][t][1]], "ps": [round(float(x)) for x in S["base"][t]["curve_s"]],
                          "pp": [round(float(x), 1) for x in S["base"][t]["curve_psf"]]}})

# ---------- records per scope ----------
def street_of(addr):
    s = re.sub(r"^\S+\s+", "", addr)
    s = re.sub(r"\s+#.*$", "", s)
    return re.sub(r"\s+\(lot.*\)$", "", s)
NONC = set("SMFOB23")
HOUSES = set("SMF")
def in_scope(i, sc):
    return sc is None or cols["tw"][i] == sc

def records(sc):
    idx = [i for i in range(homes_n) if in_scope(i, sc)]
    nonc = [i for i in idx if cols["ty"][i] in NONC]
    houses = [i for i in idx if cols["ty"][i] in HOUSES]
    out = [{"k": "value", "h": idx[0]}, {"k": "house", "h": max([i for i in nonc if not other_bldg[i]], key=lambda i: cols["bl"][i])},
           {"k": "size", "h": max([i for i in houses if not cols["fl"][i] & F_SIZEBIG], key=lambda i: cols["sf"][i])}]
    es = [e for e, er in enumerate(estates) if sc is None or er[0] == sc]
    if es:
        e = max(es, key=lambda e: estates[e][5])
        out.append({"k": "estate", "e": e, "h": estates[e][1]})
    old = [i for i in nonc if cols["yb"][i] >= 1600]          # houses: a condo unit's year is its building's
    out.append({"k": "oldest", "h": min(old, key=lambda i: (cols["yb"][i], -V(i)))})
    big = [i for i in houses if cols["sf"][i] >= 5000 and cols["yb"][i] > 0]
    if big:
        out.append({"k": "newest", "h": max(big, key=lambda i: (cols["yb"][i], cols["sf"][i]))})
    st = {}
    for i in idx:
        st.setdefault((cols["tw"][i], street_of(cols["ad"][i])), []).append(V(i))
    st = {k: v for k, v in st.items() if len(v) >= 5}
    (t, name), v = max(st.items(), key=lambda kv: np.median(kv[1]))
    first = min(i for i in idx if cols["tw"][i] == t and street_of(cols["ad"][i]) == name)
    out.append({"k": "street", "street": name, "town": t, "median": int(np.median(v)), "n": len(v), "h": first})
    return out
SCOPES = [None] + list(range(len(TOWNS)))
key = lambda sc: "all" if sc is None else str(sc)
recs = {key(sc): records(sc) for sc in SCOPES}

# ---------- collections per scope ----------
def first_caveat(f):
    for bit, lab in [(256, "Valued by sale"), (1, "Land corrected"), (128, "Protected land"), (65536, "Finished value estimated"), (16, "Unfinished when assessed"),
                     (32768, "Size corrected"), (4096, "Size looks overstated"), (8192, "Size looks understated"), (16384, "Value high for size"),
                     (64, "Value low for size"), (2, "Estate"), (512, "On town line"), (32, "Pin on lot")]:
        if f & bit: return lab
    return ""
last_sale = {t["name"]: t["lastSale"] for t in towns}
MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# each town's sales run up to its own data pull, so say where they stop when that differs
def joinand(a): return a[0] if len(a) == 1 else ", ".join(a[:-1]) + " and " + a[-1]
ends = {}
for t in TOWNS:
    ends.setdefault(last_sale[t], []).append(t)
sales_note = "The priciest single-home sales since January 2024."
early = [(m, ts) for m, ts in sorted(ends.items()) if m < 2025 * 12]          # towns whose records stop in 2024
if early:
    sales_note += " Records end in " + joinand([MON[m % 12] + " 2024 for " + joinand(ts) for m, ts in early]) + \
                  "; the other towns' run into " + ("2025 or 2026" if max(ends) >= 2026 * 12 else "2025") + "."
print("sales note:", sales_note)
COLLS = [
    ("landmarks", "Landmarks", "Historic estates with a story, researched and sourced.",
     lambda i: sparse["lm"].get(i) is not None, lambda i: -V(i), 30),
    ("hidden", "Hidden estates", "Big houses (4,000+ sq ft) set 500+ feet back from the nearest road. You'd drive right past them.",
     lambda i: cols["ty"][i] in NONC and cols["sf"][i] >= 4000 and cols["sb"][i] >= 492, lambda i: -cols["sb"][i], 60),
    ("mansions", "Mansions", "Houses with 7,000+ sq ft of living area, biggest first.",
     lambda i: cols["ty"][i] in NONC and cols["sf"][i] >= 7000 and not cols["fl"][i] & F_SIZEBIG, lambda i: -cols["sf"][i], 100),
    ("oldest", "Oldest homes", "Houses built before 1750, per the assessor's records.",
     lambda i: 1600 <= cols["yb"][i] <= 1750 and cols["ty"][i] != "C", lambda i: (cols["yb"][i], -V(i)), 100),
    ("land", "Most land", "The houses with the most acreage on their own lot.",
     lambda i: cols["ty"][i] in NONC, lambda i: -cols["ac"][i], 40),
    ("newbig", "New big builds", "Houses of 5,000+ sq ft built in 2015 or later.",
     lambda i: cols["ty"][i] in NONC and cols["yb"][i] >= 2015 and cols["sf"][i] >= 5000, lambda i: (-cols["yb"][i], -cols["sf"][i]), 100),
    ("sales", "Recent sales", sales_note,
     lambda i: cols["sy"][i] >= 2024 * 12 and cols["sp"][i] >= 300000 and not (cols["yb"][i] and cols["sy"][i] // 12 < cols["yb"][i]),
     lambda i: -cols["sp"][i], 60),
    ("traps", "Trap cases", "Homes where the assessor's number needed a second look.",
     lambda i: cols["fl"][i] & (1 | 2 | 16 | 64 | 128 | 256 | 512 | SIZEF | F_VALEST), lambda i: -V(i), 100),
    ("sizecheck", "Size doesn't add up", "Homes whose recorded size doesn't match the building, corrected where the evidence agrees, plus houses valued way off for their size. Biggest corrections first.",
     lambda i: cols["fl"][i] & SIZEF, lambda i: (0, -abs(math.log(cols["sf"][i] / sparse["sz"][i][6]))) if cols["fl"][i] & F_SIZEFIX
                                               else (1 if cols["fl"][i] & (F_SIZEBIG | F_SIZESML) else 2, -V(i)), 150),
]
colls = []
for k, t, d, keep, order, lim in COLLS:
    items = {}
    base = sorted([i for i in range(homes_n) if keep(i)], key=order)
    for sc in SCOPES:
        items[key(sc)] = [i for i in base if in_scope(i, sc)][:lim]
    colls.append({"k": k, "t": t, "d": d, "items": items})
st = {}
for i in range(homes_n):
    st.setdefault((cols["tw"][i], street_of(cols["ad"][i])), []).append(i)
st_rows = sorted([(k, v) for k, v in st.items() if len(v) >= 5], key=lambda kv: -np.median([V(i) for i in kv[1]]))
streets = {}
for sc in SCOPES:
    streets[key(sc)] = [[name, t, int(np.median([V(i) for i in v])), len(v), min(v)] for (t, name), v in st_rows if sc is None or t == sc][:25]
colls.append({"k": "streets", "t": "Priciest streets", "d": "Streets with 5+ homes, ranked by median value.", "streets": streets})

# ---------- cross-town note: how recent $1.5M+ sales compare with assessments ----------
hi = {}
for t in TOWNS:
    rs = [cols["sp"][i] / cols["as"][i] for i in range(homes_n) if cols["tw"][i] == TI[t] and cols["sy"][i] >= 2024 * 12
          and cols["sp"][i] >= 1_500_000 and cols["as"][i] > 0 and not (cols["yb"][i] and cols["sy"][i] // 12 < cols["yb"][i])]
    hi[t] = (round(float(np.median(rs)), 3), len(rs)) if len(rs) >= 5 else (None, len(rs))
print("recent $1.5M+ sale/assessment medians:", hi)

# ---------- sources ----------
SOURCES = [["MIT News, “Topsfield estate” (Oct 2000)", "https://news.mit.edu/2000/topsfield-1018"],
           ["Forbes, “Cool Estate” (Jan 2002)", "https://www.forbes.com/2002/01/11/0111how.html"],
           ["Wikipedia, River Road–Cross Street Historic District", "https://en.wikipedia.org/wiki/River_Road%E2%80%93Cross_Street_Historic_District"],
           ["Meredith Farm, History", "https://www.meredithfarm.com/history"],
           ["Wikipedia, Arden (Andover, Massachusetts)", "https://en.wikipedia.org/wiki/Arden_(Andover,_Massachusetts)"],
           ["Andover Historic Preservation, 276 North Main Street", "https://preservation.mhl.org/276-north-main-street"],
           ["Andover Historic Preservation, 24 Phillips Street", "https://preservation.mhl.org/24-phillips-street"],
           ["Andover News, “Andover Estate Sells For Record $8.5M”", "https://andovermanews.com/andover-estate-sells-for-record-8-5m/"],
           ["Andover Historic Preservation, 26 Phillips Street", "https://preservation.mhl.org/26-phillips-street"],
           ["Wikipedia, Kittredge Mansion", "https://en.wikipedia.org/wiki/Kittredge_Mansion"]]

lazy = {k: cols.pop(k) for k in ["pi", "fp", "pid", "oy"]}
full_ad = cols.pop("ad")
cols["an"], cols["sn"] = an, sn
summary = {"v": 2, "W": W, "H": HH, "proj": {"lon0": LON0, "lat0": LAT0, "mx": MX, "my": MY, "minx": minx, "maxy": maxy},
           "towns": towns, "outlines": outlines, "n": homes_n, "cols": cols, "sparse": {**sparse, "sx": sx}, "streets": street_tab, "styles": styles, "ucodes": ucodes,
           "ratio": {TI[t]: {str(y): r for y, r in S["ratio"][t].items()} for t in TOWNS}, "words": WORDS,
           "estates": estates, "recs": recs, "colls": colls, "landmarks": LM_LIST, "sources": SOURCES,
           "hiRatio": {t: v for t, v in hi.items()}, "lineCount": int(((H.FLAGS & 512) > 0).sum()), "whyLabels": S["why_labels"],
           "parcels": {t: len(town_parcels[t]) for t in TOWNS}}

import os
os.makedirs(OUT, exist_ok=True)
def write_z(name, obj):
    raw = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode()
    z = gzip.compress(raw, 9)
    b = base64.b64encode(z).decode()
    open(OUT + name, "w").write(json.dumps({"z": b}))
    return len(raw), len(z), len(b)
sizes = {"homes.json": write_z("homes.json", summary)}
tw_list = cols["tw"]

# ---------- per-town parcels ----------
def rings_of(g):
    polys = [g] if isinstance(g, Polygon) else list(getattr(g, "geoms", []))
    for pg in polys:
        if isinstance(pg, Polygon) and not pg.is_empty:
            yield pg.exterior.coords
            for r in pg.interiors:
                yield r.coords
for t in TOWNS:
    arr = []
    for gi in town_parcels[t]:
        g = P[gi]["geom"].simplify(0.8, preserve_topology=True)
        rings = []
        for ring in rings_of(g):
            pts, last = [], None
            for x, y in ring:
                q = (MXf(x), MYf(y))
                if q != last:
                    pts.append(q); last = q
            if len(pts) > 1 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) >= 3:
                flat = [pts[0][0], pts[0][1]]
                for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                    flat += [bx - ax, by - ay]
                rings.append(flat)
        arr.append(rings)
    mine = [i for i in range(homes_n) if tw_list[i] == TI[t]]
    sizes[f"p-{slug(t)}.json"] = write_z(f"p-{slug(t)}.json", {"r": arr, "pi": [lazy["pi"][i] for i in mine],
                                                               "fp": [lazy["fp"][i] for i in mine], "pid": [lazy["pid"][i] for i in mine],
                                                               "oy": [lazy["oy"][i] for i in mine]})

# ---------- per-town footprints (0.5 m units relative to the house point), in each town's home order ----------
bbp = S["bld_by_parcel"]
fps = {t: [] for t in TOWNS}
for h in H.itertuples():
    geom = None
    if h.LOCM == "bldg" and h.PIDX >= 0:
        c = [B[b] for b in bbp.get(int(h.PIDX), []) if abs(B[b]["sqft"] - h.FP) < 1]
        if c:
            geom = c[0]["geom"]
    if geom is None:
        fps[h.TOWN].append([]); continue
    g = geom.simplify(0.4)
    coords = list(g.exterior.coords)[:-1]
    fps[h.TOWN].append([v for x, y in coords for v in (int(round((x - h.HX) / 0.5)), int(round((h.HY - y) / 0.5)))])
for t in TOWNS:
    sizes[f"f-{slug(t)}.json"] = write_z(f"f-{slug(t)}.json", fps[t])

for k, (raw, z, b) in sizes.items():
    print(f"{k:<24} raw {raw/1024:8.0f} KB  gz {z/1024:7.0f} KB  b64 {b/1024:7.0f} KB")
print("homes", homes_n, "| estates", len(estates), "| frame", W, "x", HH, "m")
print("towns:", [(t["name"], t["n"], t["median"], t["n2m"], t["n3m"], t["ratio"], t["lastSale"] // 12) for t in towns])
names = {"value": "Most valuable", "house": "Priciest house", "size": "Biggest house", "estate": "Biggest estate",
         "oldest": "Oldest home", "newest": "Newest mansion", "street": "Priciest street"}
for sc, rs in recs.items():
    print(f"records[{sc}]: " + " | ".join(f"{names[r['k']]}: {full_ad[r['h']]} ({TOWNS[cols['tw'][r['h']]]})" for r in rs))
for c in colls:
    if "items" in c:
        print(f"collection {c['k']:<10} " + " ".join(f"{k}:{len(v)}" for k, v in c["items"].items()))
