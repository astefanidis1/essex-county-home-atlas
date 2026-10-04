"""Step 3: open water near the atlas towns: ocean, bays, harbors, tidal flats, lakes, ponds, reservoirs (MassDEP 1:25,000 hydrography
water bodies, not marsh or wetland) plus named rivers from the linear layer, clipped to each town's envelope."""
import json, sys, time, requests
from shapely.geometry import shape, box, mapping
from towns import TOWNS, slug, D
B = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/1_25_000_USGS_MassDEP_Hydrography/FeatureServer/"
def get(url, params):
    for a in range(5):
        try:
            j = requests.get(url, params=params, timeout=240).json()
            if "error" not in j: return j
            print("  error", j["error"].get("message"))
        except Exception as e:
            print("  failed", e)
        time.sleep(3 + 3 * a)
    raise SystemExit("gave up")
def paged(url, base):
    out, off = [], 0
    while True:
        j = get(url, dict(base, resultOffset=off, resultRecordCount=500))
        out += j.get("features", [])
        if not j.get("exceededTransferLimit"): return out
        off += len(j.get("features", []))
def esri_to_geojson(g, kind):
    if kind == "poly": return {"type": "Polygon", "coordinates": g["rings"]}
    return {"type": "MultiLineString", "coordinates": g["paths"]}
water = {}
for t in TOWNS:
    xs, ys = [], []
    for f in json.load(open(D + f"{slug(t)}_parcels.json")):
        g = f.get("geometry")
        if g:
            for r in g["rings"]:
                for x, y in r[::6]:
                    xs.append(x); ys.append(y)
    env = (min(xs) - 0.003, min(ys) - 0.003, max(xs) + 0.003, max(ys) + 0.003)
    clip = box(*env)
    common = dict(geometry=",".join(map(str, env)), geometryType="esriGeometryEnvelope", inSR=4326, outSR=4326,
                  spatialRel="esriSpatialRelIntersects", returnGeometry="true", maxAllowableOffset=0.00002, geometryPrecision=6, f="json")
    polys = paged(B + "1/query", dict(common, where="POLY_CODE IN (1,6,7,8)", outFields="OBJECTID,POLY_CODE,NAME,Shape__Area", orderByFields="OBJECTID"))
    lines = paged(B + "0/query", dict(common, where="NAME LIKE '%River%'", outFields="OBJECTID,NAME,ARC_CODE", orderByFields="OBJECTID"))
    feats = []
    for f in polys:
        a = f["attributes"]
        if a["POLY_CODE"] != 8 and (a["Shape__Area"] or 0) < 20000: continue      # ponds under ~5 acres don't make a waterfront
        g = shape(esri_to_geojson(f["geometry"], "poly")).buffer(0).intersection(clip)
        if not g.is_empty: feats.append({"code": a["POLY_CODE"], "name": a["NAME"], "geom": mapping(g)})
    for f in lines:
        g = shape(esri_to_geojson(f["geometry"], "line")).intersection(clip)
        if not g.is_empty: feats.append({"code": "river", "name": f["attributes"]["NAME"], "geom": mapping(g)})
    water[t] = feats
    print(t, "water polygons:", sum(1 for x in feats if x["code"] != "river"), "river lines:", sum(1 for x in feats if x["code"] == "river"),
          "names:", sorted({x["name"] for x in feats if x["name"]})[:8])
json.dump(water, open(D + "water.json", "w"))
