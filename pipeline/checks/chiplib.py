"""Aerial chips of flagged homes for eyeballing: house_chip("Salem", "29 Chestnut St") draws the lot and building outlines on 2025 imagery."""
import pickle, sys, math
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import D
from imgchip import chip, grid
from shapely.strtree import STRtree
from shapely.geometry import Point
PKL = D + "final3.pkl"
S = pickle.load(open(PKL, "rb"))
H, P, B = S["H"], S["P"], S["B"]
LON0, LAT0, MX, MY = S["LON0"], S["LAT0"], S["MX"], S["MY"]
ll = lambda x, y: (LON0 + x / MX, LAT0 + y / MY)
tree = STRtree([b["geom"] for b in B])
def rings_ll(g):
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    return [[ll(x, y) for x, y in p.exterior.coords] for p in polys if p.geom_type == "Polygon"]
def house_chip(town, addr, extra="", zoom=1.5):
    h = H[(H.TOWN == town) & (H.ADDRESS == addr)].iloc[0]
    rings = []
    if h.PIDX >= 0:
        for r in rings_ll(P[int(h.PIDX)]["geom"]): rings.append((r, (255, 230, 0), 2))
    for j in tree.query(Point(h.HX, h.HY).buffer(120)):
        for r in rings_ll(B[j]["geom"]): rings.append((r, (255, 40, 40), 2))
    half = max(28, math.sqrt(max(h.FP, 400) / 10.764) * zoom)
    lon, lat = ll(h.HX, h.HY)
    lab = f"{addr}, {town}\n{int(h.RES_AREA or 0):,} sq ft rec | {h.STORIES} st | outline {int(h.FP):,} | bldg ${h.BLDG_VAL/1e3:,.0f}K {extra}"
    return chip(lat, lon, half, rings=rings, label=lab)
