"""Step 2: protected open space (conservation restrictions) and MassDOT road centerlines for every town."""
import json, sys, time
import requests
from towns import TOWNS, TID, slug, D

def get(url, params):
    for attempt in range(5):
        try:
            j = requests.get(url, params=params, timeout=180).json()
            if "error" not in j:
                return j
            print("  service error:", j["error"].get("message"), "retrying")
        except Exception as e:
            print("  request failed:", e, "retrying")
        time.sleep(3 + attempt * 3)
    raise SystemExit("gave up on " + url)

def paged(url, base):
    feats, off = [], 0
    while True:
        j = get(url, dict(base, resultOffset=off, resultRecordCount=2000))
        feats += j.get("features", [])
        if not j.get("exceededTransferLimit"):
            return feats
        off += len(j.get("features", []))

OS = "https://gis.eea.mass.gov/server/rest/services/Protected_and_Recreational_OpenSpace_Polygons/FeatureServer/0/query"
import os
out = json.load(open(D + "openspace.json")) if os.path.exists(D + "openspace.json") else {}
for t in TOWNS:
    if str(TID[t]) in out:
        continue                                   # already pulled
    feats = paged(OS, dict(where=f"TOWN_ID={TID[t]}", outFields="TOWN_ID,SITE_NAME,OWNER_TYPE,LEV_PROT,OLI_1_INT,OLI_2_INT,OLI_3_INT,PRIM_PURP",
                           returnGeometry="true", outSR=4326, orderByFields="OBJECTID", f="json"))
    out[str(TID[t])] = feats
    print(t, "open space features:", len(feats))
json.dump(out, open(D + "openspace.json", "w"))

# roads: one envelope per town (the towns aren't contiguous), merged with any earlier pull
RD = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/MassDOTRoads_gdb/FeatureServer/0/query"
have = json.load(open(D + "roads.json")) if os.path.exists(D + "roads.json") else []
done = set(json.load(open(D + "roads_towns.json"))) if os.path.exists(D + "roads_towns.json") else ({"Boxford", "Middleton", "Topsfield", "Georgetown", "North Andover", "Andover", "Lynn", "Lynnfield"} if have else set())
seen = {f["attributes"]["OBJECTID"] for f in have}
for t in TOWNS:
    if t in done:
        continue
    xs, ys = [], []
    for f in json.load(open(D + f"{slug(t)}_parcels.json")):
        g = f.get("geometry")
        if g:
            for r in g["rings"]:
                for x, y in r[::4]:
                    xs.append(x); ys.append(y)
    env = (min(xs) - 0.004, min(ys) - 0.004, max(xs) + 0.004, max(ys) + 0.004)
    feats = paged(RD, dict(where="1=1", geometry=",".join(map(str, env)), geometryType="esriGeometryEnvelope", inSR=4326, outSR=4326,
                           spatialRel="esriSpatialRelIntersects", outFields="OBJECTID,CLASS,RDTYPE,STREET_NAME", returnGeometry="true",
                           orderByFields="OBJECTID", f="json"))
    new = [f for f in feats if f["attributes"]["OBJECTID"] not in seen]
    seen.update(f["attributes"]["OBJECTID"] for f in new)
    have += new; done.add(t)
    print(t, "envelope", [round(v, 4) for v in env], "road segments:", len(feats), "new:", len(new))
print("road segments total:", len(have))
json.dump(have, open(D + "roads.json", "w"))
json.dump(sorted(done), open(D + "roads_towns.json", "w"))
