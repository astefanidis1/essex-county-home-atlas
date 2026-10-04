"""Step 1: every parcel (with assessor attributes) and every building outline in each town, from MassGIS.

Run with town names to pull only those: python fetch_parcels.py Salem Beverly
"""
import json, time, requests, sys
from towns import TID as ALL, slug
from paths import D
PARCELS = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/Massachusetts_Property_Tax_Parcels/FeatureServer/0/query"
BLDGS = "https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/Building_Structures/FeatureServer/0/query"
TOWNS = {k: v for k, v in ALL.items() if k in sys.argv[1:]} if len(sys.argv) > 1 else ALL
PF = ",".join(["OBJECTID","MAP_PAR_ID","LOC_ID","POLY_TYPE","NO_MATCH","TOWN_ID","PROP_ID","BLDG_VAL","LAND_VAL","OTHER_VAL",
               "TOTAL_VAL","FY","LOT_SIZE","LOT_UNITS","LS_DATE","LS_PRICE","LS_BOOK","LS_PAGE","USE_CODE","USE_DESC","SITE_ADDR",
               "ADDR_NUM","FULL_STR","LOCATION","CITY","ZIP","ZONING","YEAR_BUILT","BLD_AREA","UNITS","RES_AREA","STYLE",
               "STORIES","NUM_ROOMS"])
def pull(url, where, fields, offset_tol, page=1000):
    feats, off = [], 0
    while True:
        p = dict(where=where, outFields=fields, returnGeometry="true", outSR=4326, geometryPrecision=6,
                 maxAllowableOffset=offset_tol, orderByFields="OBJECTID", resultOffset=off, resultRecordCount=page, f="json")
        for attempt in range(5):
            try:
                j = requests.get(url, params=p, timeout=180).json()
                if "error" not in j: break
            except Exception as e:
                j = {"error": str(e)}
            time.sleep(3)
        else:
            raise SystemExit(f"failed: {j}")
        feats += j["features"]
        if not j.get("exceededTransferLimit"): return feats
        off += len(j["features"])
for town, tid in TOWNS.items():
    par = pull(PARCELS, f"TOWN_ID={tid}", PF, 0.00003)
    bl = pull(BLDGS, f"TOWN_ID={tid} OR TOWN_ID2={tid} OR TOWN_ID3={tid}", "STRUCT_ID,AREA_SQ_FT,ARCHIVED,SOURCEDATE,TOWN_ID", 0.000004, 2000)
    json.dump(par, open(D + f"{slug(town)}_parcels.json", "w"))
    json.dump(bl, open(D + f"{slug(town)}_bldgs.json", "w"))
    arch = {}
    for b in bl: arch[b["attributes"].get("ARCHIVED")] = arch.get(b["attributes"].get("ARCHIVED"), 0) + 1
    print(f"{town}: parcels {len(par)}, buildings {len(bl)}, ARCHIVED values {arch}")
