"""Step 7: distance from each house footprint to the nearest road centerline (MassDOT roads)."""
import json, pickle, sys
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from paths import D
S = pickle.load(open(D + "final3.pkl", "rb"))
H, P, B = S["H"].copy(), S["P"], S["B"]
LON0, LAT0, MX, MY = S["LON0"], S["LAT0"], S["MX"], S["MY"]
to_m = lambda lon, lat: ((lon - LON0) * MX, (lat - LAT0) * MY)
roads = json.load(open(D + "roads.json"))
lines = []
for f in roads:
    g = f.get("geometry")
    if not g:
        continue
    for path in g.get("paths", []):
        if len(path) >= 2:
            lines.append(LineString([to_m(x, y) for x, y in path]))
tree = STRtree(lines)
print("road lines:", len(lines))
bbp = S["bld_by_parcel"]
geoms = []
for h in H.itertuples():
    g = None
    if h.LOCM == "bldg" and h.PIDX >= 0:
        c = [B[b]["geom"] for b in bbp.get(int(h.PIDX), []) if abs(B[b]["sqft"] - h.FP) < 1]
        g = c[0] if c else None
    geoms.append(g if g is not None else Point(h.HX, h.HY))
idx, dist = tree.query_nearest(geoms, return_distance=True, all_matches=False)
sb = np.full(len(H), np.nan)
sb[idx[0]] = dist
H["SETBACK_M"] = sb
print("setback m by town (median, p95):", H.groupby("TOWN").SETBACK_M.agg(["median", lambda s: s.quantile(.95)]).round(0).to_dict("index"))
print("missing:", int(H.SETBACK_M.isna().sum()))
pickle.dump({**S, "H": H}, open(D + "final4.pkl", "wb"))
print("saved final4.pkl")
