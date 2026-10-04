"""Cross-check the site data against the live MassGIS server: each town's top assessed homes, and that pins land inside their own parcel.

Usage: python checks/verify_against_server.py [Town ...]   (default: every town)
"""
import json, gzip, base64, pickle, time, sys, concurrent.futures as cf
import os
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import D, SITE
from towns import TOWNS, TID
CHECK = sys.argv[1:] or TOWNS
PAR = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/Massachusetts_Property_Tax_Parcels/FeatureServer/0/query"
d = json.loads(gzip.decompress(base64.b64decode(json.load(open(SITE + "homes.json"))["z"])))
c, P = d["cols"], d["proj"]
S = pickle.load(open(D + "final7.pkl", "rb"))
H = S["H"].sort_values(["BEST", "TOTAL_VAL"], ascending=False).reset_index(drop=True)
assert len(H) == d["n"]
assert all(int(round(a)) == b for a, b in zip(H.TOTAL_VAL.head(2000), c["as"][:2000])), "order mismatch"
lon = P["lon0"] + (np.array(c["x"]) + P["minx"]) / P["mx"]
lat = P["lat0"] + (P["maxy"] - np.array(c["y"])) / P["my"]
def get(params):
    for k in range(5):
        try:
            j = requests.get(PAR, params=params, timeout=60).json()
            if "error" not in j: return j
        except Exception: pass
        time.sleep(2 + k * 2)
    return {"features": []}
# 1. top assessed values per town, straight from the server
for t in CHECK:
    j = get(dict(where=f"TOWN_ID={TID[t]} AND (USE_CODE LIKE '10%' OR USE_CODE LIKE '01%' OR USE_CODE LIKE '0101%')", outFields="PROP_ID,SITE_ADDR,TOTAL_VAL,USE_CODE",
                 orderByFields="TOTAL_VAL DESC", resultRecordCount=12, returnGeometry="false", f="json"))
    ours = H[H.TOWN == t].set_index("PROP_ID")
    rows = []
    for f in j["features"]:
        a = f["attributes"]; pid = a["PROP_ID"]
        st = "in atlas" if pid in ours.index else "excluded"
        if pid in ours.index:
            o = ours.loc[pid]; o = o.iloc[0] if isinstance(o, pd.DataFrame) else o
            st += " (assessed matches)" if abs(o.TOTAL_VAL - a["TOTAL_VAL"]) < 1 else f" (MISMATCH ours {o.TOTAL_VAL})"
        rows.append(f"   {a['SITE_ADDR']:<28} {a['USE_CODE']:<5} ${a['TOTAL_VAL']:>12,.0f}  {st}")
    print(f"{t}: server top home-coded records"); print("\n".join(rows))
# 2. pins inside their own parcel: the top 60 overall, plus 50 random homes and the top 40 in each town checked
idx = list(range(60)); import random; random.seed(7); idx += [k for t in CHECK for k in random.sample(list(H.index[H.TOWN == t]), 50)]
for t in CHECK:
    idx += list(H.index[H.TOWN == t][:40])
idx = sorted(set(idx))
def check(i):
    j = get(dict(geometry=f"{lon[i]:.7f},{lat[i]:.7f}", geometryType="esriGeometryPoint", inSR=4326, spatialRel="esriSpatialRelIntersects",
                 outFields="LOC_ID,PROP_ID", returnGeometry="false", f="json"))
    locs = {f["attributes"]["LOC_ID"] for f in j.get("features", [])}
    return i, H.at[i, "LOC_ID"] in locs, locs
bad = []
with cf.ThreadPoolExecutor(6) as ex:
    for i, ok, locs in ex.map(check, idx):
        if not ok: bad.append((i + 1, H.at[i, "TOWN"], H.at[i, "ADDRESS"], H.at[i, "LOCM"], sorted(locs)[:2]))
print(f"\npins checked: {len(idx)}, inside own parcel: {len(idx) - len(bad)}")
for b in bad: print("   not inside:", b)
