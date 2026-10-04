#!/usr/bin/env python3
"""
Rank every home in a Massachusetts town by assessed value, with Zillow links.

Data: MassGIS statewide property-tax parcels (the town assessor's own values),
free and public. No scraping.

Usage:
    python town_home_values.py Boxford            # writes boxford_home_values.csv, prints top 15
    python town_home_values.py "North Andover" --top 25

Needs: pip install requests pandas
"""
import argparse
import re
import sys
import time
import urllib.parse

import pandas as pd
import requests

SERVICE = ("https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/"
           "Massachusetts_Property_Tax_Parcels/FeatureServer/0/query")

FIELDS = ["LOC_ID", "PROP_ID", "SITE_ADDR", "ADDR_NUM", "FULL_STR", "LOCATION", "CITY", "ZIP",
          "USE_CODE", "USE_DESC", "TOTAL_VAL", "BLDG_VAL", "LAND_VAL", "OTHER_VAL", "FY",
          "LOT_SIZE", "LOT_UNITS", "YEAR_BUILT", "RES_AREA", "STYLE", "NUM_ROOMS", "LS_DATE", "LS_PRICE"]

# MA Dept. of Revenue use codes that are homes (first 3 digits; some towns add a 4th-digit suffix)
HOME_CODES = {"101": "Single family", "102": "Condo", "103": "Mobile home",
              "104": "Two-family", "105": "Three-family", "109": "Multiple houses"}
# Mixed use, primarily residential (01x)
MIXED_RES = {
    "012": "House + open land", "016": "House + open land", "017": "House + open land",
    "018": "House + open land",       # land in the farm/forest/recreation programs (Ch. 61, 61A, 61B)
    "013": "House + business", "014": "House + business",
    "019": "House + other use",
}
# "House + open land": that land is taxed at a reduced rate, so the assessed value
# understates what the property would sell for. Check these on Zillow even if ranked lower.

# Assessor street names that differ from the postal name Zillow uses, per town (verified)
POSTAL_FIXES = {
    "BOXFORD": {
        "LAKE SHORE RD": "Lakeshore Rd",
        "WOOD HILL LN": "Woodhill Ln",
        "DANA RD EXT": "Dana Rd",
        "JOSEPH SMITH LN": "Joseph Smith Way",
        "JOSEPH SMITH LN EXT": "Joseph Smith Way",
        "STILES POND RD MIDDLE": "Stiles Pond Rd",
        "STILES POND RD SOUTH": "Stiles Pond Rd",
        "STILES POND SOUTH": "Stiles Pond Rd",
        "CAMP GROUND RD": "Campground Rd",
        "ROUND TOP RD": "Roundtop Rd",
    },
    "SALEM": {
        "WASHINGTON SQUARE NO": "Washington Sq N",
        "WASHINGTON SQUARE SO": "Washington Sq S",
        "WASHINGTON SQUARE EA": "Washington Sq E",
        "WASHINGTON SQUARE WE": "Washington Sq W",
    },
}
SUFFIXES = {"LANE": "Ln", "CIRCLE": "Cir", "ROAD": "Rd", "DRIVE": "Dr", "STREET": "St",
            "AVENUE": "Ave", "COURT": "Ct", "TERRACE": "Ter", "PLACE": "Pl",
            "WY": "Way", "AV": "Ave", "CR": "Cir", "CI": "Cir", "CG": "Crossing", "PH": "Path", "RW": "Row",
            "LA": "Ln", "LNDG": "Landing", "CRES": "Crescent", "EXT.": "Ext",
            "TR": "Ter", "TERR": "Ter", "PK": "Park", "HGTS": "Heights", "BV": "Blvd", "XING": "Crossing"}   # assessor "TR" is Terrace


def home_type(code: str):
    """Return a label if the use code is a home, else None."""
    code = code.strip().upper() if isinstance(code, str) else ""
    if not code:
        return None
    if code[:3] in HOME_CODES:
        return HOME_CODES[code[:3]]
    if len(code) == 4 and code[0] == "0" and code[1:4] in HOME_CODES:   # zero-padded, e.g. 0101
        return HOME_CODES[code[1:4]]
    if code[:3] in MIXED_RES:
        return MIXED_RES[code[:3]]
    return None


def street_name(full_str: str, town: str = "") -> str:
    s = re.sub(r"\s+", " ", full_str.strip().upper()).replace("`", "'") if isinstance(full_str, str) else ""
    fixes = POSTAL_FIXES.get(town.strip().upper(), {})
    if s in fixes:
        return fixes[s]
    words = []
    ws = s.split(" ")
    last = len(ws) - 1                     # the suffix word: last, or before a trailing NORTH/EXT/... modifier
    while last > 0 and ws[last] in {"NORTH", "SOUTH", "EAST", "WEST", "EXT", "EXT.", "EXTENSION", "REAR"}:
        last -= 1
    for k, w in enumerate(ws):
        if w in SUFFIXES and k >= last:
            words.append(SUFFIXES[w])
        elif w == "MT":
            words.append("Mount")
        elif re.match(r"^O'[A-Z]", w):
            words.append("O'" + w[2:].capitalize())
        elif "-" in w:                       # LABOR-IN-VAIN -> Labor-in-Vain
            words.append("-".join(q.lower() if 0 < i < len(w.split("-")) - 1 and q in {"IN", "OF", "THE", "BY", "ON", "AT", "AND"} else q.capitalize()
                                  for i, q in enumerate(w.split("-"))))
        else:
            words.append(w.capitalize())
    return " ".join(words)


def _clean(x) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip()).upper() if isinstance(x, str) else ""


def normalize_num(num: str) -> str:
    """Assessor house-number spellings -> postal style: '491--507' -> '491-507', '53-A' -> '53A', '20--1/2' -> '20½'."""
    n = re.sub(r"\s*-+\s*", "-", num)
    n = re.sub(r"\s*([+&])\s*", r"\1", n)
    n = re.sub(r"-?\s?1/2\b", "½", n)
    n = re.sub(r"(\d)\.5\b", r"\1½", n)                           # '2.5' -> '2½
    n = re.sub(r"(\d)-([A-Z])(?=$|-)", r"\1\2", n)
    return n.strip("-")


def address_parts(row, town: str = ""):
    """(house number, postal street name, unit) for one assessor record."""
    num, street, site = _clean(row.ADDR_NUM), _clean(row.FULL_STR), _clean(row.SITE_ADDR)
    num = re.sub(r"^REAR\b[\s-]*", "", num)                       # "REAR-16": a house behind no. 16
    if "USES" in num or "(" in num:                   # e.g. ADDR_NUM "159-(USES", SITE_ADDR "159 (USES 161 RIVER RD"
        m = re.search(r"USES\s+(\d+[A-Z]?)", site)
        num = m.group(1) if m else re.sub(r"[^0-9A-Z]", "", num.split("(")[0])
    unit = ""
    tail = site.split(street, 1)[1] if street and street in site else ""
    m = re.match(r"^\s*(?:#|UNIT\b|APT\b|U(?=[A-Z0-9]))\s*([A-Z0-9-]+)", tail)
    if m:
        unit = m.group(1)
    elif _clean(row.LOCATION):
        unit = _clean(row.LOCATION)
    if re.fullmatch(r"(?:BLDG|BLD|BUILDING)\s*\d*[A-Z]?", unit):     # "151 LABOR-IN-VAIN RD BLDG2": a building label, not a unit
        unit = ""
    # assessor notes like "COLRAIN RD (USES 19)" or "GAIL ST (USES 1 GAIL PL)": use the address in use
    u = re.search(r"\s*\(?\bUSES\s+([^)]+?)\)?\s*$", street)
    if u:
        used, street = u.group(1).strip(), street[:u.start()].strip()
        if re.fullmatch(r"\d+[A-Z]?", used):
            num = used
        elif re.match(r"^\d+[A-Z]?\s+\S", used):
            num, street = used.split(" ", 1)
    return normalize_num(num), street_name(street, town), unit


def is_rear(row) -> bool:
    return bool(re.match(r"^REAR\b", _clean(row.ADDR_NUM)) or re.match(r"^REAR\b", _clean(row.SITE_ADDR)))


def display_address(row, town: str = "") -> str:
    num, street, unit = address_parts(row, town)
    return f"{num} {street}".strip() + (f" #{unit}" if unit else "") + (" (rear)" if is_rear(row) else "")


def zillow_url(row, town: str) -> str:
    num, street, unit = address_parts(row, town)
    first = re.split(r"[&+,\-\s]+", num)[0] if num else ""          # "8A + 8B" -> "8A"
    addr = f"{first} {street}".strip() + (f" Unit {unit}" if unit else "")
    zip5 = str(row.ZIP or "")[:5] if isinstance(row.ZIP, str) else ""
    full = f"{addr}, {town.title()}, MA {zip5}".strip()
    slug = re.sub(r"\s+", "-", full)
    return "https://www.zillow.com/homes/" + urllib.parse.quote(slug, safe="-,") + "_rb/"


def town_id(town: str) -> int:
    r = requests.get(SERVICE, params=dict(
        where=f"UPPER(CITY)='{town.upper()}'", outFields="TOWN_ID", returnGeometry="false",
        returnDistinctValues="true", f="json"), timeout=60).json()
    ids = [f["attributes"]["TOWN_ID"] for f in r.get("features", []) if f["attributes"]["TOWN_ID"]]
    if not ids:
        sys.exit(f"Couldn't find a town called '{town}'. Check the spelling.")
    return ids[0]


def fetch(tid: int, fields=FIELDS, geometry=False, **extra) -> list:
    feats, offset = [], 0
    while True:
        params = dict(where=f"TOWN_ID={tid}", outFields=",".join(fields),
                      returnGeometry=str(geometry).lower(), orderByFields="OBJECTID",
                      resultOffset=offset, resultRecordCount=1000, f="json", **extra)
        for attempt in range(4):                       # the service throws the odd transient error
            j = requests.get(SERVICE, params=params, timeout=180).json()
            if "error" not in j:
                break
            time.sleep(3)
        else:
            sys.exit(f"MassGIS error: {j['error']}")
        feats += j["features"]
        if not j.get("exceededTransferLimit"):
            return feats
        offset += len(j["features"])


def rank_homes(town: str, feats: list) -> pd.DataFrame:
    df = pd.DataFrame([f["attributes"] for f in feats])
    df["TYPE"] = df.USE_CODE.map(home_type)
    homes = df[df.TYPE.notna() & (df.TOTAL_VAL.fillna(0) > 0)].copy()
    zips = homes.ZIP.dropna().astype(str).str.slice(0, 5)
    zips = zips[zips.str.fullmatch(r"\d{5}")]
    if zips.nunique() == 1:                              # one-ZIP town: fill the blanks
        homes["ZIP"] = zips.iloc[0]
    homes["ADDRESS"] = homes.apply(display_address, axis=1, town=town)
    homes["ZILLOW"] = homes.apply(zillow_url, axis=1, town=town)
    homes = homes.sort_values("TOTAL_VAL", ascending=False).reset_index(drop=True)
    homes.insert(0, "RANK", homes.index + 1)
    return homes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("town")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    tid = town_id(a.town)
    homes = rank_homes(a.town, fetch(tid))
    out = f"{a.town.lower().replace(' ', '_')}_home_values.csv"
    cols = ["RANK", "ADDRESS", "TYPE", "TOTAL_VAL", "BLDG_VAL", "LAND_VAL", "RES_AREA", "LOT_SIZE",
            "LOT_UNITS", "YEAR_BUILT", "LS_DATE", "LS_PRICE", "FY", "ZILLOW"]
    homes[cols].to_csv(out, index=False)

    fy = int(homes.FY.mode()[0]) if homes.FY.notna().any() else "?"
    w = homes.iloc[0]
    print(f"\n{a.town.title()}: {len(homes):,} homes ranked by FY{fy} assessed value -> {out}")
    print(f"\nWinner: {w.ADDRESS}  ${w.TOTAL_VAL:,.0f}\n        {w.ZILLOW}\n")
    for _, h in homes.head(a.top).iterrows():
        flag = "  *open land" if h.TYPE == "House + open land" else ""
        print(f"{h.RANK:>4}. {h.ADDRESS:<32} ${h.TOTAL_VAL:>12,.0f}{flag}   {h.ZILLOW}")
    if (homes.head(a.top).TYPE == "House + open land").any():
        print("\n*open land = farm/forest/recreation land taxed at a reduced rate, so the assessed value "
              "understates market value. Worth checking on Zillow even if it ranks lower.")


if __name__ == "__main__":
    main()
