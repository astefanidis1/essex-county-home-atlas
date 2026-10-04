"""Where the pipeline keeps things: downloads and intermediate files in data/ (not committed), the site in docs/."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data") + os.sep        # MassGIS downloads and the pickles passed between steps
SITE = os.path.join(ROOT, "docs") + os.sep     # the published site (GitHub Pages serves this folder)
WEB = os.path.join(ROOT, "web") + os.sep       # page source: head, body, script
os.makedirs(D, exist_ok=True)
