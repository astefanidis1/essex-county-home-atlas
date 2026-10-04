"""Step 6: conservation restrictions, a per-town sales index and sale-based values, town-line houses, landmarks."""
import json, math, pickle, re, sys, time
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from paths import D
S = pickle.load(open(D + "final.pkl", "rb"))
H, P, B, E, A, curves = S["H"].copy(), S["P"], S["B"], S["E"].copy(), S["raw_attrs"], S["curves"]
LON0, LAT0, MX, MY = S["LON0"], S["LAT0"], S["MX"], S["MY"]
to_m = lambda lon, lat: ((lon - LON0) * MX, (lat - LAT0) * MY)
from towns import TOWNS, TID as TIDS
TID = {v: k for k, v in TIDS.items()}
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
F_LAND, F_EST, F_MULTI, F_CONDO, F_NEW, F_PIN, F_LOW = 1, 2, 4, 8, 16, 32, 64
F_CR, F_SALE, F_LINE = 128, 256, 512

# ---------- 1. permanent conservation restrictions (MassGIS open space) ----------
os_ = json.load(open(D + "openspace.json"))
restr = []
for tid, feats in os_.items():
    for f in feats:
        a, g = f["attributes"], f.get("geometry")
        ints = {a.get("OLI_1_INT"), a.get("OLI_2_INT"), a.get("OLI_3_INT")}
        if not g or a.get("LEV_PROT") != "P":
            continue
        if not (ints & {"CR", "APR", "CAPR", "WR", "HR"}) and a.get("OWNER_TYPE") != "P":
            continue
        polys = [Polygon([to_m(*p) for p in r]).buffer(0) for r in g["rings"] if len(r) >= 4]
        u = unary_union([p for p in polys if not p.is_empty])
        if not u.is_empty:
            restr.append((u, (a.get("SITE_NAME") or "").strip()))
tree = STRtree([r[0] for r in restr])
cr_frac, cr_name = [], []
for h in H.itertuples():
    if h.PIDX < 0 or h.TYPE == "Condo":
        cr_frac.append(0.0); cr_name.append(""); continue
    g = P[int(h.PIDX)]["geom"]
    tot, names = 0.0, set()
    for j in tree.query(g):
        inter = g.intersection(restr[j][0]).area
        if inter > 0:
            tot += inter; names.add(restr[j][1])
    cr_frac.append(min(1.0, tot / g.area) if g.area else 0.0)
    cr_name.append("; ".join(sorted(n for n in names if n))[:90])
H["CR_FRAC"], H["CR_NAME"] = cr_frac, cr_name
restricted = (H.CR_FRAC >= 0.25) & (H.LOT_SIZE.fillna(0) >= 2)
undo = H.F_LAND & restricted
print("land corrections withdrawn (land is permanently restricted):", H[undo][["TOWN", "ADDRESS", "TOTAL_VAL", "ADJ", "CR_FRAC"]].to_dict("records"))
H.loc[undo, "ADJ"] = H.loc[undo, "TOTAL_VAL"]
H.loc[undo, "F_LAND"] = False
H["FLAGS"] = np.where(undo, H.FLAGS & ~F_LAND, H.FLAGS)
H["FLAGS"] = np.where(restricted, H.FLAGS | F_CR, H.FLAGS)

# ---------- 2. market index: median sale / FY2026 assessment by town and year ----------
# bulk transfers: 25+ parcels changing hands on one day (Ipswich's Little Neck condo conversion in August 2012,
# where each cottage owner bought their land share for ~$180K) aren't market sales of the homes
bulk = set()
for town in TOWNS:
    df = A[town]
    s0 = df[(df.LS_PRICE.fillna(0) >= 1000) & df.LS_DATE.notna()].drop_duplicates("LOC_ID")
    days = s0.groupby("LS_DATE").LOC_ID.nunique()
    for d in days[days >= 25].index:
        ids = set(s0[s0.LS_DATE == d].LOC_ID)
        bulk |= {(town, i) for i in ids}
        print(f"bulk transfer: {town} {d}: {len(ids)} parcels (median ${s0[s0.LS_DATE == d].LS_PRICE.median():,.0f}) - not used as sales")
ratio = {}
deed_n = {}
for town in TOWNS:
    df = A[town]
    k = df.LS_BOOK.fillna("").astype(str) + "/" + df.LS_PAGE.fillna("").astype(str)
    n = df.assign(k=k).groupby("k").LOC_ID.transform("nunique")
    deed_n.update({(town, pid): nn for pid, nn in zip(df.PROP_ID, n)})
    s = df.assign(n=n, bulk=[(town, i) in bulk for i in df.LOC_ID])
    s = s[(s.LS_PRICE >= 100000) & (s.n == 1) & ~s.bulk & (s.TOTAL_VAL > 0) & s.USE_CODE.fillna("").str.contains(r"^0?10[1249]|^101", regex=True)
          & (s.YEAR_BUILT.fillna(0) < s.LS_DATE.fillna("0").str[:4].astype(int))]
    s = s.assign(y=s.LS_DATE.str[:4].astype(int), r=s.LS_PRICE / s.TOTAL_VAL)
    g = s[s.y >= 2010].groupby("y").r.agg(["median", "size"])
    ratio[town] = {int(y): float(m) for y, (m, c) in g.iterrows() if c >= 8}
    # a year far off both neighbors is a data artifact, not the market: use the neighbors' average
    for y in sorted(ratio[town]):
        a, b = ratio[town].get(y - 1), ratio[town].get(y + 1)
        if a and b and abs(math.log(ratio[town][y]) - (math.log(a) + math.log(b)) / 2) > 0.25:
            print(f"ratio index {town} {y}: {ratio[town][y]:.3f} replaced with {math.sqrt(a * b):.3f}")
            ratio[town][y] = math.sqrt(a * b)

def implied(town, date, price):
    if not isinstance(date, str) or len(date) < 4 or not price or price < 100000:
        return None
    y = int(date[:4])
    r = ratio[town].get(y)
    return price / r if r else None

# ---------- 3. sale evidence for single homes whose assessment is known to be distorted ----------
distorted = H.F_LAND | ((H.FLAGS & (F_CR | F_LOW)) > 0) | (H.TYPE == "House + open land")
H["SALE_IMPL"] = np.nan
H["BEST"] = H.ADJ.astype(float)
H["SALE_RECENT"] = False
for i, h in H.iterrows():
    single = deed_n.get((h.TOWN, h.PROP_ID), 1) == 1
    built_before = isinstance(h.LS_DATE, str) and (h.YEAR_BUILT if not pd.isna(h.YEAR_BUILT) else 0) < int(h.LS_DATE[:4] or 0)
    v = implied(h.TOWN, h.LS_DATE, h.LS_PRICE) if (single and built_before and (h.TOWN, h.LOC_ID) not in bulk) else None
    if v:
        H.at[i, "SALE_IMPL"] = v                   # shown as context for every home
        # a sale more than ten years before the valuation date is too old to re-price a house reliably (and a conservation
        # restriction is often granted after the purchase), so it's context only
        if distorted.loc[i] and v > h.ADJ * 1.05 and int(h.LS_DATE[:4]) >= h.FYV - 11:
            H.at[i, "BEST"] = v
            H.at[i, "FLAGS"] = H.at[i, "FLAGS"] | F_SALE
        elif int(h.LS_DATE[:4]) >= h.FYV - 2 and v > h.ADJ * 1.15:
            # sold since the valuation date's prior year for well above the assessment: the assessor hasn't caught up
            H.at[i, "BEST"] = v
            H.at[i, "FLAGS"] = H.at[i, "FLAGS"] | F_SALE
            H.at[i, "SALE_RECENT"] = True

# ---------- 4. estates: re-value with the sale when it says the package is worth more ----------
E["base"] = 0.0; E["impl"] = np.nan; E["best"] = 0.0
for ei, e in E.iterrows():
    town = e.town
    df = A[town]
    mem = df[df.LOC_ID.isin(e.locs)].drop_duplicates("LOC_ID")
    home_idx = e.homes
    base = float(H.loc[home_idx, "BEST"].sum()) + float(mem[~mem.LOC_ID.isin(H.loc[home_idx, "LOC_ID"])].TOTAL_VAL.fillna(0).sum())
    dyear = int(str(e.sale_date)[:4]) if isinstance(e.sale_date, str) and e.sale_date[:4].isdigit() else 0
    built_after = (H.loc[home_idx, "YEAR_BUILT"].fillna(0) >= dyear).any() if dyear else True
    fyv = int(H.loc[home_idx, "FYV"].iloc[0]) if len(home_idx) else 0
    v = implied(town, e.sale_date, e.sale_price) if (not built_after and dyear >= fyv - 11) else None
    dist = bool(distorted.loc[home_idx].any()) or bool(mem.USE_CODE.fillna("").str.match(r"^[678]").any())
    E.at[ei, "base"], E.at[ei, "impl"] = base, v if v else np.nan
    best = base
    if v and dist and v > base * 1.05:
        best = v
        excess = v - base
        ac_total = float(mem.LOT_SIZE.fillna(0).sum()) or 1.0
        for hi in home_idx:
            share = float(H.at[hi, "LOT_SIZE"] or 0) / ac_total
            H.at[hi, "BEST"] = H.at[hi, "BEST"] + excess * share
            H.at[hi, "FLAGS"] = H.at[hi, "FLAGS"] | F_SALE
    E.at[ei, "best"] = best

# ---------- 5. houses on a town line ----------
BU = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/Building_Structures/FeatureServer/0/query"
ids = ",".join(str(v) for v in TIDS.values())
feats, off = [], 0
while True:
    for attempt in range(5):
        j = requests.get(BU, params=dict(where=f"(TOWN_ID IN ({ids}) OR TOWN_ID2 IN ({ids})) AND TOWN_ID2 IS NOT NULL",
                                         outFields="STRUCT_ID,TOWN_ID,TOWN_ID2", returnGeometry="false", orderByFields="OBJECTID",
                                         resultOffset=off, resultRecordCount=2000, f="json"), timeout=120).json()
        if "error" not in j: break
        time.sleep(4)
    feats += j["features"]
    if not j.get("exceededTransferLimit"): break
    off += len(j["features"])
cross = {f["attributes"]["STRUCT_ID"]: (f["attributes"]["TOWN_ID"], f["attributes"]["TOWN_ID2"]) for f in feats}
print("buildings crossing a town line (region):", len(cross))
# names for neighboring towns: ask the parcel layer for one record of each unknown town id
PU = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/Massachusetts_Property_Tax_Parcels/FeatureServer/0/query"
for tid in sorted({t for ab in cross.values() for t in ab if t} - set(TID)):
    try:
        r = requests.get(PU, params=dict(where=f"TOWN_ID={tid} AND CITY IS NOT NULL", outFields="CITY", returnGeometry="false",
                                          resultRecordCount=50, f="json"), timeout=60).json()
        names = [f["attributes"]["CITY"] for f in r.get("features", []) if f["attributes"].get("CITY")]
        if names:
            TID[tid] = max(set(names), key=names.count).title()
    except Exception as e:
        print("town name lookup failed", tid, e)
print("neighbor names:", {k: v for k, v in TID.items() if v not in TOWNS})
bbp = S["bld_by_parcel"]
H["LINE_TOWN"] = ""
for i, h in H.iterrows():
    if h.LOCM != "bldg" or h.PIDX < 0:
        continue
    for b in bbp.get(int(h.PIDX), []):
        if abs(B[b]["sqft"] - h.FP) < 1 and B[b]["id"] in cross:
            a, bb = cross[B[b]["id"]]
            other = TID.get(bb if TID.get(a) == h.TOWN else a, "a neighboring town")
            H.at[i, "LINE_TOWN"] = other
            H.at[i, "FLAGS"] = H.at[i, "FLAGS"] | F_LINE

# ---------- 6. duplicate addresses get their lot number ----------
dup = H.duplicated(["TOWN", "ADDRESS"], keep=False) & (H.TYPE != "Condo")
maps = {(t, pid): mp for t in TOWNS for pid, mp in zip(A[t].PROP_ID, A[t].MAP_PAR_ID)}
H["ADDRESS"] = [f"{a} (lot {maps.get((t, pid), pid)})" if d else a for a, t, pid, d in zip(H.ADDRESS, H.TOWN, H.PROP_ID, dup)]

# ---------- 7. landmarks (researched; sources cited on the page) ----------
LANDMARKS = {
    ("Topsfield", "68 River Rd"): ("The Coolidge estate mansion",
        "A 24-room Georgian mansion (14 bedrooms) designed in 1921 by Philip Richardson for John Lee Saltonstall and later owned by "
        "William Appleton Coolidge, who left his River Road lands to MIT. MIT marketed it with a building restriction on the "
        "land, and it sold in 2002 for $4.2 million. The assessor now values the house itself at only $547,800, far below what a house this size "
        "normally carries, so treat its value here as uncertain.", [0, 1, 2]),
    ("Topsfield", "41 Cross St"): ("Meredith Farm",
        "An Arts and Crafts estate: J. Morris Meredith built the main house in 1900, and the farm grew to about 250 acres. "
        "William Appleton Coolidge bought it in 1970 and deeded it to MIT, which put permanent restrictions on the land. "
        "It's still a private working farm.", [2, 3]),
    ("Topsfield", "29 Cross St"): ("Meredith Farm",
        "The second house of Meredith Farm, sold together with 41 Cross St.", [2, 3]),
    ("Topsfield", "82 River Rd"): ("The Cummings Estate",
        "Tudor Revival buildings designed by architect Charles K. Cummings, with grounds by landscape architect Arthur Shurcliff.", [2]),
    ("Topsfield", "47-49 Salem Rd"): ("The Perkins-Pingree House",
        "A Federal-style house from about 1807, described as one of the finest of its type in Topsfield. The assessor records it "
        "as a condominium, which is why it shows up as a condo.", [2]),
    ("Andover", "276 North Main St"): ("Arden",
        "The estate of William M. Wood, founder of the American Woolen Company, who bought it in 1891 and added many outbuildings. "
        "The Gothic Revival house was built in the 1840s for flax-mill owner John Dove by local builder Jacob Chickering. It's on the "
        "National Register, and its 63.7 acres carry a permanent conservation restriction, so the land's low assessment is real.", [4, 5]),
    ("Andover", "24 Phillips St"): ("The Dove-Hayes House",
        "A Colonial Revival mansion built in the 1890s for Susan and George W. W. Dove, near Phillips Academy. It sold for $8.5 million "
        "in July 2025, the highest price ever paid for an Andover home and more than double the previous record.", [6, 7]),
    ("Andover", "26 Phillips St"): ("The Byers House",
        "A Georgian Revival house built around 1930 by Marjorie S. Hayes next to the Dove-Hayes House, part of a Hayes family "
        "compound on Andover Hill. It's named for anthropologist Douglas S. Byers, later director of Phillips Academy's Peabody "
        "archaeology museum, and his wife Dorothy.", [8]),
    ("North Andover", "56 Academy Rd"): ("The Kittredge Mansion",
        "A three-story Federal mansion built in 1784 for the Kittredges, a long line of local doctors, with its design attributed to "
        "Salem builder Samuel McIntire (the assessor lists 1820). It's on the National Register.", [9]),
}
H["LANDMARK"] = [LANDMARKS.get((t, a)) for t, a in zip(H.TOWN, H.ADDRESS)]
print("landmarks matched:", int(H.LANDMARK.notna().sum()), "of", len(LANDMARKS))

print("\nsale-based values applied:", int(((H.FLAGS & F_SALE) > 0).sum()), "| of which recent sales well above assessment:", int(H.SALE_RECENT.sum()), H[H.SALE_RECENT].groupby("TOWN").size().to_dict())
print(H[H.SALE_RECENT].assign(R=lambda d: d.LS_PRICE / d.TOTAL_VAL).sort_values("BEST", ascending=False)[["TOWN", "ADDRESS", "TYPE", "TOTAL_VAL", "BEST", "LS_DATE", "LS_PRICE", "R", "YEAR_BUILT", "RES_AREA"]].head(40).to_string(index=False))
print("ratio index:", {t: {y: round(r, 3) for y, r in ratio[t].items() if y >= 2022} for t in TOWNS})
print(H[(H.FLAGS & F_SALE) > 0][["TOWN", "ADDRESS", "TOTAL_VAL", "ADJ", "BEST", "LS_DATE", "LS_PRICE"]].sort_values("BEST", ascending=False).head(12).to_string(index=False))
print("\nestates where the sale raised the value:")
print(E[E.best > E.base * 1.001][["town", "main_addr", "acres", "assessed", "base", "impl", "best", "sale_date", "sale_price"]].to_string(index=False))
print("\ntown-line houses:", H[(H.FLAGS & F_LINE) > 0][["TOWN", "ADDRESS", "LINE_TOWN"]].to_dict("records"))
print("\nNEW regional top 15 by best estimate:")
top = H.sort_values("BEST", ascending=False).head(15)
print(top[["TOWN", "ADDRESS", "TOTAL_VAL", "BEST", "FLAGS", "CR_FRAC"]].to_string(index=False))

pickle.dump({**S, "H": H, "E": E, "ratio": ratio, "bulk": bulk}, open(D + "final3.pkl", "wb"))
print("\nsaved final3.pkl")
