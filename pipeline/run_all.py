"""Run the whole pipeline: download from MassGIS, process, and rebuild the site in docs/.

    python pipeline/run_all.py            # everything (the downloads take a while)
    python pipeline/run_all.py --skip-fetch   # reuse the downloads already in data/
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
FETCH = ["fetch_parcels.py", "fetch_openspace_roads.py", "fetch_water.py"]
PROCESS = ["homes.py", "estates_flags.py", "sales_restrictions.py", "setbacks.py", "size_checks.py", "size_corrections.py",
           "why_tags.py", "build_site_data.py", "assemble_page.py"]
steps = PROCESS if "--skip-fetch" in sys.argv else FETCH + PROCESS
for s in steps:
    t0 = time.time()
    print(f"== {s}", flush=True)
    subprocess.run([sys.executable, s], cwd=HERE, check=True)
    print(f"   done in {time.time() - t0:.0f}s", flush=True)
