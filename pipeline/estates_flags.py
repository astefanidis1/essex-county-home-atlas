"""Step 5: estates (parcels sold together on one deed), flags, unfinished new builds."""
import math, pickle, re, sys
import numpy as np
import pandas as pd
from shapely.geometry import Point

from paths import D
S = pickle.load(open(D + "processed.pkl", "rb"))
H, P, B, curves = S["H"], S["P"], S["B"], S["curves"]
A = S["raw_attrs"]
from towns import TOWNS
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
ploc = {(p["town"], p["loc"]): i for i, p in enumerate(P)}

def expected_land(town, acres):
    ac, lv, _ = curves[town]
    if acres <= ac[0]: return float(lv[0])
    if acres <= ac[-1]: return float(np.interp(math.log(acres), np.log(ac), lv))
    k = max(len(ac) - 3, 0)
    rate = (lv[-1] - lv[k]) / max(ac[-1] - ac[k], 1e-9)
    return float(lv[-1] + max(rate, 0) * (acres - ac[-1]))

HOME3 = {"101", "102", "103", "104", "105", "109"}
def code3(c):
    c = (c or "").strip().upper() if isinstance(c, str) else ""
    return c[1:4] if (len(c) == 4 and c[0] == "0" and c[1:4] in HOME3) else c[:3]

# ---------------- estates: parcels whose latest sale was the same deed ----------------
H["ESTATE"] = -1
estates = []
skipped = {}
for town in TOWNS:
    df = A[town].copy()
    df["DEED"] = (df.LS_BOOK.fillna("").astype(str).str.strip().str.lstrip("0") + "/" +
                  df.LS_PAGE.fillna("").astype(str).str.strip().str.lstrip("0"))
    df = df[df.LS_BOOK.notna() & df.LS_PAGE.notna() & (df.DEED != "/") & df.LOC_ID.notna()]
    th = H[H.TOWN == town]
    home_by_loc = {}
    for i, h in th.iterrows():
        home_by_loc.setdefault(h.LOC_ID, []).append(i)
    for deed, g in df.groupby("DEED"):
        g = g.drop_duplicates("LOC_ID")
        if not (2 <= len(g) <= 8):
            continue
        hidx = [i for loc in g.LOC_ID for i in home_by_loc.get(loc, [])]
        if not hidx:
            continue
        hs = H.loc[hidx]
        if (hs.TYPE == "Condo").any():
            continue
        # an estate is houses plus their land and outbuildings, not an investment package or a subdivision
        acres_all = float(g.LOT_SIZE.fillna(0).sum())
        other = g[~g.LOC_ID.isin(set(hs.LOC_ID))]
        oc = other.USE_CODE.map(code3)
        commercial = oc.str.match(r"^(11[12]|12\d|[349]\d\d|0[34]\d)$") & (other.LOT_SIZE.fillna(0) < 5)
        heavy = other.BLDG_VAL.fillna(0).sum() > max(250_000, 0.5 * hs.BLDG_VAL.fillna(0).sum()) and acres_all < 10
        lots = (oc.str.match(r"^1[03]\d$") & (other.LOT_SIZE.fillna(0) < 2.5)).sum() >= 3
        portfolio = len(hidx) >= 3 and acres_all < 10
        if commercial.any() or heavy or lots or portfolio:
            skipped[("commercial" if commercial.any() else "heavy" if heavy else "lots" if lots else "portfolio", town)] = \
                skipped.get(("commercial" if commercial.any() else "heavy" if heavy else "lots" if lots else "portfolio", town), 0) + 1
            continue
        dyear = int(str(g.LS_DATE.dropna().iloc[0])[:4]) if g.LS_DATE.notna().any() else 0
        if dyear and (hs.YEAR_BUILT.fillna(0) > dyear).sum() >= 2:      # a builder's package, not an estate
            continue
        main = hs.ADJ.idxmax()
        mx, my = H.at[main, "HX"], H.at[main, "HY"]
        far = False
        for loc in g.LOC_ID:
            pi = ploc.get((town, loc))
            if pi is None or P[pi]["geom"].representative_point().distance(Point(mx, my)) > 1500:
                far = True; break
        if far:
            continue
        acres = float(g.LOT_SIZE.fillna(0).sum())
        land = float(g.LAND_VAL.fillna(0).sum())
        bldg_other = float(g.BLDG_VAL.fillna(0).sum() + g.OTHER_VAL.fillna(0).sum())
        assessed = float(g.TOTAL_VAL.fillna(0).sum())
        exp_l = expected_land(town, acres) if acres >= 5 else land
        adjusted = bldg_other + max(land, exp_l)
        eid = len(estates)
        estates.append({"id": eid, "town": town, "deed": deed, "main": int(main), "homes": [int(i) for i in hidx],
                        "locs": list(g.LOC_ID), "pidx": [ploc.get((town, l), -1) for l in g.LOC_ID],
                        "n_parcels": len(g), "n_homes": len(hidx), "acres": acres, "assessed": assessed,
                        "adjusted": adjusted, "sale_date": str(g.LS_DATE.dropna().iloc[0]) if g.LS_DATE.notna().any() else "",
                        "sale_price": float(g.LS_PRICE.dropna().iloc[0]) if g.LS_PRICE.notna().any() else 0,
                        "rough": acres > curves[town][0][-1] and max(land, exp_l) > land})
        for i in hidx:
            H.at[i, "ESTATE"] = eid
E = pd.DataFrame(estates)
print("deed groups rejected as estates:", skipped)
print("estates:", len(E), E.groupby("town").size().to_dict() if len(E) else {})
if len(E):
    E["main_addr"] = E.main.map(H.ADDRESS)
    print(E.sort_values("acres", ascending=False)[["town", "main_addr", "n_parcels", "n_homes", "acres", "assessed", "adjusted", "sale_date", "sale_price"]].head(15).to_string(index=False))

# ---------------- flags ----------------
F_LAND, F_ESTATE, F_MULTI, F_CONDO, F_NEW, F_PIN, F_LOWHOUSE = 1, 2, 4, 8, 16, 32, 64
flags = np.zeros(len(H), dtype=int)
flags |= np.where(H.F_LAND, F_LAND, 0)
flags |= np.where(H.ESTATE >= 0, F_ESTATE, 0)
flags |= np.where(H.TYPE == "Multiple houses", F_MULTI, 0)
flags |= np.where(H.TYPE == "Condo", F_CONDO, 0)
new_psf = {}
for town in TOWNS:
    fy = int(H[H.TOWN == town].FYV.iloc[0])
    ref = H[(H.TOWN == town) & (H.TYPE != "Condo") & H.YEAR_BUILT.between(fy - 11, fy - 3) & (H.RES_AREA >= 2500)]
    new_psf[town] = float((ref.BLDG_VAL / ref.RES_AREA).median()) if len(ref) >= 5 else float("nan")
    print(town, "recent-build reference homes:", len(ref))
psf = H.BLDG_VAL / H.RES_AREA.where(H.RES_AREA > 0)
recent = H.YEAR_BUILT.fillna(0) >= (H.FYV - 2)
unfinished = recent & (H.TYPE != "Condo") & ((psf < 0.5 * H.TOWN.map(new_psf)) | (H.YEAR_BUILT.fillna(0) >= H.FYV - 1))
flags |= np.where(unfinished, F_NEW, 0)
print("recent-build $/sqft norms:", {t: (round(v) if v == v else None) for t, v in new_psf.items()})
flags |= np.where(H.LOCM != "bldg", F_PIN, 0)
# big houses assessed far below similar-size homes per sq ft (condition, unusual use, or data issue)
for town in TOWNS:
    m = (H.TOWN == town) & (H.TYPE != "Condo") & (H.RES_AREA.fillna(0) >= 4000)
    med = (H.loc[m, "BLDG_VAL"] / H.loc[m, "RES_AREA"]).median()
    low = m & (H.RES_AREA >= 5000) & ((H.BLDG_VAL / H.RES_AREA) < 0.4 * med)
    flags |= np.where(low, F_LOWHOUSE, 0)
H["FLAGS"] = flags
F_GROSS = 131072                        # living area estimated from a gross-area-only record (Salem)
H["FLAGS"] = H.FLAGS | np.where(H.GROSS.notna(), F_GROSS, 0)
print("\nflag counts:", {n: int(((H.FLAGS & b) > 0).sum()) for n, b in
                         [("land", 1), ("estate", 2), ("multi", 4), ("condo", 8), ("newbuild", 16), ("pin-on-parcel", 32), ("low-house", 64)]})
print("\nprobably unfinished when assessed:")
print(H[(H.FLAGS & F_NEW) > 0][["TOWN", "ADDRESS", "YEAR_BUILT", "TOTAL_VAL", "RES_AREA"]].sort_values("TOTAL_VAL", ascending=False).head(10).to_string(index=False))
print("\nbig houses assessed low for their size:")
print(H[(H.FLAGS & F_LOWHOUSE) > 0][["TOWN", "ADDRESS", "RES_AREA", "BLDG_VAL", "TOTAL_VAL", "YEAR_BUILT", "STYLE"]].to_string(index=False))

# ---------------- unassessed: vacant-land parcels with a sizable building on 2025 imagery ----------------
bbp = S["bld_by_parcel"]
unassessed = []
for town in TOWNS:
    df = A[town]
    vac = df[df.USE_CODE.fillna("").map(code3).isin(["130", "131", "132"])].drop_duplicates("LOC_ID")
    for r in vac.itertuples():
        pi = ploc.get((town, r.LOC_ID))
        if pi is None:
            continue
        bl = [B[b] for b in bbp.get(pi, []) if B[b]["sqft"] >= 2000]
        if bl:
            big = max(bl, key=lambda b: b["sqft"])
            c = big["geom"].representative_point()
            unassessed.append({"town": town, "addr": r.SITE_ADDR, "use": r.USE_CODE, "total": r.TOTAL_VAL,
                               "acres": r.LOT_SIZE, "fp": round(big["sqft"]), "x": c.x, "y": c.y, "pidx": pi})
U = pd.DataFrame(unassessed)
print("\nvacant-land parcels with a 2,000+ sq ft building on 2025 imagery:", len(U))
if len(U):
    print(U.sort_values("fp", ascending=False)[["town", "addr", "use", "total", "acres", "fp"]].head(20).to_string(index=False))

# ---------------- house-footprint sanity for the top homes ----------------
H["REGRANK"] = H.ADJ.rank(ascending=False, method="first").astype(int)
top = H.sort_values("ADJ", ascending=False).head(12)
print("\nregional top 12 (adjusted), chosen footprint vs living area:")
print(top[["REGRANK", "TOWN", "ADDRESS", "ADJ", "TOTAL_VAL", "RES_AREA", "STORIES", "FP", "LOCM", "FLAGS"]].to_string(index=False))

pickle.dump({**S, "H": H, "E": E, "U": U, "new_psf": new_psf}, open(D + "final.pkl", "wb"))
print("\nsaved final.pkl")
