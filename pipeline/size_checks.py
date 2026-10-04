"""Step 8: size sanity checks: recorded living area vs. the building outlines on aerial photos, and house value per sq ft vs. similar homes.

Flags (added to FLAGS):
  F_LOW      64    house value under 40% of similar-size homes per sq ft (old rule kept too)
  F_SIZEBIG  4096  recorded size more than the building can hold, and the value also fits a smaller house
  F_SIZESML  8192  building much bigger than the recorded size implies, and the value also fits a bigger house
  F_HIGHV    16384 house value at least 2.5x similar-size homes per sq ft (no outline explanation)
"""
import json, math, pickle, re, sys, collections
import numpy as np, pandas as pd
from paths import D
from towns import TOWNS, slug
S = pickle.load(open(D + "final4.pkl", "rb"))
H, P, B, bbp = S["H"].copy(), S["P"], S["B"], S["bld_by_parcel"]
F_NEW, F_LOW, F_SIZEBIG, F_SIZESML, F_HIGHV = 16, 64, 4096, 8192, 16384
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 120); pd.set_option("display.max_colwidth", 24)

srcyr = {}
for t in TOWNS:
    for b in json.load(open(D + f"{slug(t)}_bldgs.json")):
        a = b["attributes"]; srcyr[a["STRUCT_ID"]] = int((a.get("SOURCEDATE") or 0) // 10000)

def floors(v, area):
    m = re.match(r"^(\d+(?:\.\d+)?)([AHT]?)$", str(v).strip().upper())
    if not m or float(m.group(1)) <= 0 or float(m.group(1)) > 4:        # "12", "75": typos, not story counts
        return 2.0 if (area or 0) > 2500 else 1.5
    return min(4.0, float(m.group(1)) + {"A": 0.5, "H": 0.5, "T": 0.75, "": 0}[m.group(2)])

# the building each home is pinned to, how many homes share it, and all outlines assigned to the lot
pin_b, roof_lot = [], []
for h in H.itertuples():
    bl = bbp.get(int(h.PIDX), []) if h.PIDX >= 0 else []
    c = [b for b in bl if abs(B[b]["sqft"] - h.FP) < 1] if h.LOCM == "bldg" else []
    pin_b.append(c[0] if c else -1)
    roof_lot.append(sum(B[b]["sqft"] for b in bl if B[b]["sqft"] >= 150))
H["PINB"], H["ROOF_LOT"] = pin_b, roof_lot
share = collections.Counter(b for b in H.PINB if b >= 0)
H["SHARED"] = H.PINB.map(lambda b: share[b] > 1 if b >= 0 else False)
H["OUTYR"] = H.PINB.map(lambda b: srcyr.get(B[b]["id"], 0) if b >= 0 else 0)
H["FLOORS"] = [floors(s, a) for s, a in zip(H.STORIES, H.RES_AREA)]
# share of the pinned outline inside the home's own lot: an outline that runs well into the next lot is probably joined
# to a neighbor's building (or the lot lines are off), so it says little about this house's size
def inside(h):
    if h.PINB < 0 or h.PIDX < 0: return np.nan
    b, p = B[int(h.PINB)]["geom"], P[int(h.PIDX)]["geom"]
    try: return b.intersection(p).area / b.area
    except Exception: return np.nan
H["INSIDE"] = [inside(h) for h in H.itertuples()]

# expected house value per sq ft among similar homes: same town, kind, size band, age band
ok = (H.TYPE != "Condo") & (H.RES_AREA.fillna(0) >= 400) & (H.BLDG_VAL.fillna(0) > 20000)
X = H[ok].copy()
X["LPSF"] = np.log(X.BLDG_VAL / X.RES_AREA)
X["SZB"] = pd.cut(X.RES_AREA, [0, 1200, 1800, 2500, 3500, 5000, 7000, 1e9], labels=False)
yb = X.YEAR_BUILT.fillna(0)
X["AGEB"] = np.select([yb <= 0, yb < 1900, yb < 1946, yb < 1980, yb < 2000, yb < 2015], [0, 1, 2, 3, 4, 5], 6)
X["GRP"] = np.where(X.TYPE.isin(["Two-family", "Three-family", "Multiple houses"]), "multi", "single")
def med(keys, minn):
    g = X.groupby(keys).LPSF
    return g.transform("median").where(g.transform("size") >= minn)
X["ELPSF"] = med(["TOWN", "GRP", "SZB", "AGEB"], 15).fillna(med(["TOWN", "GRP", "SZB"], 15)).fillna(med(["TOWN", "SZB"], 10))
H["EXP_PSF"] = np.exp(X.ELPSF).reindex(H.index)
H["RESID"] = (X.LPSF - X.ELPSF).reindex(H.index)
H.loc[H.TYPE != "Condo", "RESID"] = H.loc[H.TYPE != "Condo", "RESID"]    # condos stay NaN

base = (H.TYPE != "Condo") & H.RESID.notna() & ((H.FLAGS & F_NEW) == 0)
multi = H.TYPE.isin(["Two-family", "Three-family", "Multiple houses", "House + business"])
cap = H.ROOF_LOT * (H.FLOORS + 1)                                  # generous: a finished basement and attic on top of the stated floors
outline_ok = (H.LOCM == "bldg") & (H.OUTYR >= H.YEAR_BUILT.fillna(0)) & (H.INSIDE >= 0.7)  # traced after the house was built, mostly on its own lot
sizebig_dual = base & outline_ok & (H.RESID < math.log(0.5)) & (H.RES_AREA > 1.4 * cap) & (H.RES_AREA - cap >= 1500)
cap_house = H.FP * (H.FLOORS + 1)                                  # the house outline alone, same allowance
sizebig_geo = (H.TYPE != "Condo") & outline_ok & (H.FP >= 1000) & ~multi & ~H.SHARED & (H.RES_AREA > 1.6 * cap_house) & (H.RES_AREA - cap_house >= 3000)
sizebig = sizebig_dual | sizebig_geo
H["SIZEBIG_VALUE_AGREES"] = sizebig_dual
ratio_up = H.FP * H.FLOORS / H.RES_AREA
sizesml = base & outline_ok & ~sizebig & (H.RESID > 0.35) & (ratio_up > 2.5) & (H.FP >= 3000) & ~H.SHARED & ~multi & \
          (H.YEAR_BUILT.fillna(0) >= 1950) & ~H.STYLE.fillna("").str.upper().str.contains("ROW|TOWN|DUPLEX")
low = base & ~sizebig & (H.RESID < math.log(0.4)) & (H.RES_AREA >= 2000)
# a high value per sq ft is normal for luxury houses; flag it only when the outlines don't back up the recorded size
unconfirmed = (H.LOCM != "bldg") | (ratio_up > 1.6) | (H.ROOF_LOT > 2 * H.FP)
highv = base & ~sizesml & (H.RESID > math.log(2.5)) & (H.BLDG_VAL >= 500000) & unconfirmed
H["FLAGS"] = H.FLAGS | np.where(sizebig, F_SIZEBIG, 0) | np.where(sizesml, F_SIZESML, 0) | np.where(low, F_LOW, 0) | np.where(highv, F_HIGHV, 0)
H["FLAGS"] = np.where(sizebig, H.FLAGS & ~F_LOW, H.FLAGS)        # an overstated size explains the low value per sq ft
H["UPRATIO"] = ratio_up
cols = ["TOWN", "ADDRESS", "TYPE", "STORIES", "RES_AREA", "FP", "ROOF_LOT", "OUTYR", "YEAR_BUILT", "BLDG_VAL", "EXP_PSF", "RESID"]
for name, m in [("SIZE OVERSTATED", sizebig), ("SIZE UNDERSTATED", sizesml), ("VALUE LOW FOR SIZE", (H.FLAGS & F_LOW) > 0), ("VALUE HIGH FOR SIZE", highv)]:
    print(f"\n{name}: {int(m.sum())}", H[m].groupby("TOWN").size().to_dict())
    print(H[m].sort_values("BEST", ascending=False)[cols].head(40).round({"EXP_PSF": 0, "RESID": 2}).to_string(index=False))
pickle.dump({**S, "H": H}, open(D + "final5.pkl", "wb"))
print("\nsaved final5.pkl")
