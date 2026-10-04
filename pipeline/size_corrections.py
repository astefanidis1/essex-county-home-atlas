"""Step 9: corrections for the size and value flags, where the evidence supports a number.

1. Unfinished new builds: only houses valued under half the town's recent-build $/sq ft are flagged (F_NEW),
   and each gets an estimated finished value (F_VALEST): land + other + living area x that $/sq ft.
2. Sizes: three clues to the real living area, each independent of the recorded figure in a different way:
     C_BLD  the assessor's own gross building area / the typical gross-to-living ratio for that kind of house in the town
     C_OUT  the building outline on aerial photos x stories x the town's typical living area per sq ft of roof for that kind of house
     C_VAL  the house value / the typical $/sq ft of similar homes (looked up at the clue's own size)
   A size is corrected (F_SIZEFIX, CSF) when every available clue (at least two, one of them the outline or the value)
   sits 30%+ away from the recorded size on the same side, and their middle value is 75%+ away (R1); or when the
   outline and value both say 2x+ and agree (R2, a record that predates a rebuild or addition).
3. Value flags are re-checked against corrected sizes; a house valued far below its size gets its own sale's
   market-adjusted price when that's higher (same rule as other distorted assessments).
"""
import math, pickle, re, sys, itertools
import numpy as np, pandas as pd
from paths import D
from towns import TOWNS
S = pickle.load(open(D + "final5.pkl", "rb"))
H, A, ratio = S["H"].copy(), S["raw_attrs"], S["ratio"]
pd.set_option("display.width", 330); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 20)
F_NEW, F_LOW, F_SALE, F_SIZEBIG, F_SIZESML, F_HIGHV, F_SIZEFIX, F_VALEST = 16, 64, 256, 4096, 8192, 16384, 32768, 65536

# ---------- 1. unfinished new builds ----------
psf = H.BLDG_VAL / H.RES_AREA.where(H.RES_AREA > 0)
norm = H.TOWN.map(S["new_psf"])
recent = H.YEAR_BUILT.fillna(0) >= (H.FYV - 2)
unfinished = recent & (H.TYPE != "Condo") & (psf < 0.5 * norm)
print("unfinished flag: was", int(((H.FLAGS & F_NEW) > 0).sum()), "now", int(unfinished.sum()))
H["FLAGS"] = np.where(unfinished, (H.FLAGS | F_NEW) & ~F_LOW, H.FLAGS & ~F_NEW)     # unfinished explains the low value
H["VALEST"] = (H.LAND_VAL.fillna(0) + H.OTHER_VAL.fillna(0) + H.RES_AREA * norm).where(unfinished)
use = unfinished & (H.VALEST > H.BEST * 1.02)
H.loc[use, "BEST"] = H.loc[use, "VALEST"]
H["FLAGS"] = np.where(use, H.FLAGS | F_VALEST, H.FLAGS)
print(H[unfinished][["TOWN", "ADDRESS", "YEAR_BUILT", "RES_AREA", "BLDG_VAL", "LAND_VAL", "TOTAL_VAL", "VALEST", "BEST"]].sort_values("BEST", ascending=False).round(0).to_string(index=False))

# ---------- 2. size clues ----------
def sclass(s):
    s = (s if isinstance(s, str) else "").upper()
    if re.search(r"SPLIT|SPLT|RAISED|MULTI-LEVEL|BI-LEVEL", s): return "split"
    if "CAPE" in s: return "cape"
    if re.search(r"RANCH|CAMP|DECK HOUSE", s): return "ranch"
    # colonials are two-story houses even when the record says one story; calibrating them apart absorbs that miscoding
    if re.search(r"COLONIAL|COLNL|CLNL|GARRISON|FEDERAL|GEORGIAN|VICTORIAN|4SQ|SALT ?BOX|GAMBREL|MANSION", s): return "colonial"
    return "other"
H["SCLS"] = H.STYLE.map(sclass)
H["GRP"] = np.where(H.TYPE.isin(["Two-family", "Three-family"]), "multi", "single")
H["FLB"] = np.select([H.FLOORS <= 1.25, H.FLOORS <= 1.75, H.FLOORS <= 2.25], [1.0, 1.5, 2.0], 3.0)
# row houses, townhouses and duplex halves share walls and outlines, and price differently from detached houses: no corrections
rowlike = H.STYLE.fillna("").str.upper().str.contains(r"ROW|TOWN|DUPLEX|TNHS")
elig = H.TYPE.isin(["Single family", "Two-family", "Three-family", "House + open land"]) & (H.RES_AREA.fillna(0) >= 500) & ~rowlike
# the outline counts when it was traced after the house was built, belongs to this home alone, sits mostly on its own
# lot, and the story count is believable (4+ stories on a house is almost always a coding slip)
outline_ok = (H.LOCM == "bldg") & (H.OUTYR >= H.YEAR_BUILT.fillna(0)) & ~H.SHARED & (H.FP > 0) & (H.INSIDE >= 0.7) & (H.FLOORS < 4)
extra = (H.LOCM == "bldg") & (H.ROOF_LOT > 1.5 * H.FP)             # barns, arenas, guest houses: their value is in BLDG_VAL too
# calibration homes for each clue (North Andover records have no gross area, so each clue gets its own set)
base_ok = elig & ~extra & ((H.FLAGS & F_NEW) == 0) & (H.BLDG_VAL > 20000)
# some towns' "gross area" is just a copy of the living area (Beverly, Marblehead, Salem): no clue there
copy_town = H.groupby("TOWN").apply(lambda d: float(((d.BLD_AREA == d.RES_AREA) & (d.RES_AREA > 0)).sum()) / max(int((d.RES_AREA > 0).sum()), 1) >= 0.9)
print("towns whose gross area copies the living area:", [t for t, v in copy_town.items() if v])
copy_rec = H.TOWN.map(copy_town).astype(bool) | H.GROSS.notna()      # Salem: living area was estimated from that gross area
CLEAN = {"C_BLD": base_ok & (H.BLD_AREA > 0) & ~copy_rec, "C_OUT": base_ok & outline_ok, "C_VAL": base_ok}
H["BLDB"] = pd.cut(H.BLD_AREA.where(H.BLD_AREA > 0), [0, 2500, 3500, 5000, 7000, 10000, 1e9], labels=False)
print("\ncalibration homes:", {k: int(v.sum()) for k, v in CLEAN.items()})

def calib(vals, keysets, clean, minn=25):
    """median of vals over the first key set with enough homes, per home"""
    C = H[clean].assign(V=vals[clean])
    out = pd.Series(np.nan, index=H.index)
    for ks in keysets:
        g = C.groupby(ks).V.agg(["median", "size"])
        med = g["median"].where(g["size"] >= minn)
        look = pd.Series([med.get(tuple(r) if len(ks) > 1 else r[0], np.nan) for r in H[ks].itertuples(index=False)], index=H.index)
        out = out.fillna(look)
    return out
H["R_EXP"] = calib(H.BLD_AREA / H.RES_AREA, [["TOWN", "GRP", "SCLS", "FLB", "BLDB"], ["TOWN", "GRP", "SCLS", "FLB"], ["GRP", "SCLS", "FLB"], ["TOWN", "GRP"]], CLEAN["C_BLD"])
H["K_EXP"] = calib(H.RES_AREA / (H.FP * H.FLOORS), [["TOWN", "GRP", "SCLS", "FLB"], ["GRP", "SCLS", "FLB"], ["TOWN", "GRP", "FLB"], ["TOWN", "GRP"]], CLEAN["C_OUT"])
H["C_BLD"] = (H.BLD_AREA / H.R_EXP).where(elig & (H.BLD_AREA > 0) & ~copy_rec)
H["C_OUT"] = (H.FP * H.FLOORS * H.K_EXP).where(elig & outline_ok)
# value clue: typical house $/sq ft for the cohort, looked up again at the clue's own size until it settles
SZ = [0, 1200, 1800, 2500, 3500, 5000, 7000, 1e9]
yb = H.YEAR_BUILT.fillna(0)
H["AGEB"] = np.select([yb <= 0, yb < 1900, yb < 1946, yb < 1980, yb < 2000, yb < 2015], [0, 1, 2, 3, 4, 5], 6)
C = H[CLEAN["C_VAL"]].assign(L=np.log(H.BLDG_VAL / H.RES_AREA), SZB=pd.cut(H.RES_AREA, SZ, labels=False))
tabs = []
for ks, mn in ((["TOWN", "GRP", "SZB", "AGEB"], 15), (["TOWN", "GRP", "SZB"], 15), (["TOWN", "SZB"], 10)):
    g = C.groupby(ks).L.agg(["median", "size"]); tabs.append((ks, g["median"].where(g["size"] >= mn)))
def lpsf(h, s):
    b = int(np.digitize(s, SZ[1:-1]))
    vals = {"TOWN": h.TOWN, "GRP": h.GRP, "SZB": b, "AGEB": h.AGEB}
    for ks, t in tabs:
        v = t.get(tuple(vals[k] for k in ks))
        if v is not None and not pd.isna(v): return v
    return np.nan
def exp_psf_at(h, s):
    v = lpsf(h, s)
    return math.exp(v) if not pd.isna(v) else np.nan
cv = []
for h in H.itertuples():
    if not elig[h.Index] or not (h.BLDG_VAL > 20000) or (h.FLAGS & F_NEW) or extra[h.Index]:
        cv.append(np.nan); continue
    s, hist = h.RES_AREA, []
    for _ in range(6):
        e = lpsf(h, s)
        if pd.isna(e): s = np.nan; break
        s = h.BLDG_VAL / math.exp(e); hist.append(s)
    if len(hist) >= 2 and abs(math.log(hist[-1] / hist[-2])) > 0.01:
        s = math.sqrt(hist[-1] * hist[-2])                           # settles between two size bands
    cv.append(s)
H["C_VAL"] = cv
SIG = {}
for c in ["C_BLD", "C_OUT", "C_VAL"]:
    m = CLEAN[c] & outline_ok & (H.BLD_AREA > 0)                      # the same homes for all three, so the spreads compare
    e = np.log(H.loc[m, c] / H.loc[m, "RES_AREA"]).replace([np.inf, -np.inf], np.nan).dropna()
    SIG[c] = (e.quantile(.9) - e.quantile(.1)) / 2.56
    print(f"{c}: n {len(e)}, p10/p50/p90 {np.round(e.quantile([.1, .5, .9]).values, 3)}, sigma ~{SIG[c]:.3f}")
W = {k: 1 / v**2 for k, v in SIG.items()}
wmean = lambda d: math.exp(sum(W[k] * math.log(v) for k, v in d.items()) / sum(W[k] for k in d))

# ---------- 3. corrections ----------
# stronger prior evidence lowers the bar: living area recorded above the building's own gross area (impossible),
# or an earlier size flag (outline and value both disagree with the record)
prior = ((H.BLD_AREA > 0) & (H.RES_AREA > H.BLD_AREA)) | ((H.FLAGS & (F_SIZEBIG | F_SIZESML)) > 0)
print("homes with prior evidence:", int(prior.sum()), "| living area above gross area:", int(((H.BLD_AREA > 0) & (H.RES_AREA > H.BLD_AREA)).sum()))
kind, csf = [], []
for h in H.itertuples():
    res, k, c = h.RES_AREA, None, np.nan
    EACH, CONS = (math.log(1.2), math.log(1.3)) if prior[h.Index] else (math.log(1.3), math.log(1.75))
    cl = {n: getattr(h, n) for n in W if not pd.isna(getattr(h, n)) and getattr(h, n) > 0}
    if elig[h.Index] and len(cl) >= 2 and ("C_OUT" in cl or "C_VAL" in cl):
        s = {n: math.log(v / res) for n, v in cl.items()}
        up, down = all(v >= EACH for v in s.values()), all(v <= -EACH for v in s.values())
        if up or down:
            if len(cl) == 3: m = sorted(cl.values())[1]
            else:
                a, b = cl.values()
                m = wmean(cl) if abs(math.log(a / b)) <= (0.5 if prior[h.Index] else 0.4) else np.nan
            ok = not pd.isna(m) and abs(math.log(m / res)) >= CONS and abs(m - res) >= 600
            # outlines and gross areas both take in attached barns and garages, so a bigger size needs the house value behind it too
            if ok and up and ("C_VAL" not in cl or cl["C_VAL"] < 0.75 * m):
                ok = False
            if ok: k, c = "R1", m
        if k is None and "C_OUT" in cl and "C_VAL" in cl:
            so, sv = s["C_OUT"], s["C_VAL"]
            m = wmean({"C_OUT": cl["C_OUT"], "C_VAL": cl["C_VAL"]})
            if so * sv > 0 and min(abs(so), abs(sv)) >= math.log(2.0) and abs(so - sv) <= 0.4 and abs(m - res) >= 600:
                k, c = "R2", m
    kind.append(k); csf.append(c)
H["FIXKIND"], H["CSF"] = kind, csf
H["PRIOR"] = prior
fix = H.FIXKIND.notna()
H.loc[fix, "CSF"] = (H.loc[fix, "CSF"] / 10).round() * 10
H["FLAGS"] = np.where(fix, (H.FLAGS | F_SIZEFIX) & ~(F_SIZEBIG | F_SIZESML), H.FLAGS)
# value flags re-checked at the corrected size
H["EXP_PSF_C"] = [exp_psf_at(h, h.CSF) if f else np.nan for h, f in zip(H.itertuples(), fix)]
resid_c = np.log(H.BLDG_VAL / H.CSF) - np.log(H.EXP_PSF_C)
low_c = fix & (resid_c < math.log(0.4)) & (H.CSF >= 2000) & ((H.FLAGS & F_NEW) == 0)
high_c = fix & (resid_c > math.log(2.5)) & (H.BLDG_VAL >= 500000)
H["FLAGS"] = np.where(fix, (H.FLAGS & ~(F_LOW | F_HIGHV)) | np.where(low_c, F_LOW, 0) | np.where(high_c, F_HIGHV, 0), H.FLAGS)
H.loc[fix, "EXP_PSF"] = H.loc[fix, "EXP_PSF_C"]
# value high for its size: only when something says the recorded size could be the problem (no outline, other big
# buildings on the lot, or the gross area / outline clues put it 30%+ bigger). Lots with several houses are left out
# of both value checks: some towns record only the main house's living area for them, so $/sq ft means little.
multi_lot = H.TYPE == "Multiple houses"
bigger = (H.C_OUT > 1.3 * H.RES_AREA) | (H.C_BLD > 1.3 * H.RES_AREA)
unconfirmed = (H.LOCM != "bldg") | (H.ROOF_LOT > 2 * H.FP) | bigger
H["HIGHV_WHY"] = np.select([H.LOCM != "bldg", H.ROOF_LOT > 2 * H.FP, bigger], [3, 1, 2], 0)
base_v = (H.TYPE != "Condo") & H.RESID.notna() & ((H.FLAGS & F_NEW) == 0) & ~multi_lot & ~fix
highv = base_v & ((H.FLAGS & F_SIZESML) == 0) & (H.RESID > math.log(2.5)) & (H.BLDG_VAL >= 500000) & unconfirmed
was = int(((H.FLAGS & F_HIGHV) > 0).sum())
H["FLAGS"] = np.where(fix, H.FLAGS, np.where(highv, H.FLAGS | F_HIGHV, H.FLAGS & ~F_HIGHV))
H["FLAGS"] = np.where(multi_lot, H.FLAGS & ~(F_LOW | F_HIGHV), H.FLAGS)
print("value-high flags:", was, "->", int(((H.FLAGS & F_HIGHV) > 0).sum()), "| value-low:", int(((H.FLAGS & F_LOW) > 0).sum()))
cols = ["TOWN", "ADDRESS", "STYLE", "STORIES", "YEAR_BUILT", "RES_AREA", "BLD_AREA", "FP", "OUTYR", "C_BLD", "C_OUT", "C_VAL", "CSF", "FIXKIND", "BLDG_VAL", "EXP_PSF"]
for d, m in (("DOWN", fix & (H.CSF < H.RES_AREA)), ("UP", fix & (H.CSF > H.RES_AREA))):
    print(f"\nSIZE CORRECTED {d}: {int(m.sum())}", H[m].groupby("TOWN").size().to_dict())
    print(H[m].sort_values("RES_AREA", ascending=False)[cols].round(0).to_string(index=False))
print("\nstill flagged (no correction): overstated", int(((H.FLAGS & F_SIZEBIG) > 0).sum()), "understated", int(((H.FLAGS & F_SIZESML) > 0).sum()),
      "| value low", int(((H.FLAGS & F_LOW) > 0).sum()), "value high", int(((H.FLAGS & F_HIGHV) > 0).sum()))
print(H[(H.FLAGS & (F_SIZEBIG | F_SIZESML)) > 0][["TOWN", "ADDRESS", "RES_AREA", "BLD_AREA", "FP", "C_BLD", "C_OUT", "C_VAL", "FLAGS"]].round(0).to_string(index=False))

# ---------- 4. houses valued far below their size: use their own sale when it says more ----------
lowsale = ((H.FLAGS & F_LOW) > 0) & ((H.FLAGS & F_SALE) == 0) & H.SALE_IMPL.notna() & (H.SALE_IMPL > H.BEST * 1.05) & \
          (H.LS_DATE.fillna("0").str[:4].astype(int) >= H.FYV - 11)              # sold within about ten years
print("\nvalue-low homes re-valued by their own sale:", int(lowsale.sum()))
print(H[lowsale][["TOWN", "ADDRESS", "RES_AREA", "BLDG_VAL", "TOTAL_VAL", "BEST", "LS_DATE", "LS_PRICE", "SALE_IMPL"]].round(0).to_string(index=False))
H.loc[lowsale, "BEST"] = H.loc[lowsale, "SALE_IMPL"]
H["FLAGS"] = np.where(lowsale, H.FLAGS | F_SALE, H.FLAGS)
pickle.dump({**S, "H": H}, open(D + "final6.pkl", "wb"))
print("\nsaved final6.pkl")
