"""Step 12: assemble docs/index.html from the page source in web/ (head, body, script) as one standalone document."""
from paths import SITE, WEB

URL = "https://astefanidis1.github.io/essex-county-home-atlas/"
TITLE = "Essex County Home Atlas"
DESC = ("Every home in fifteen Essex County, MA towns ranked by value, with the assessor's blind spots corrected: "
        "farm-rate land, stale sizes, apartment blocks coded as houses, and more.")
ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%231c5cab'/%3E"
        "%3Cpath d='M7 16.5 16 9l9 7.5V25a1 1 0 0 1-1 1h-5.5v-6h-5v6H8a1 1 0 0 1-1-1z' fill='%23fff'/%3E%3C/svg%3E")
meta = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="icon" href="{ICON}">
<meta property="og:type" content="website">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="{URL}assets/preview.png">
<meta name="twitter:card" content="summary_large_image">
"""
head, body, script = (open(WEB + f, encoding="utf-8").read() for f in ("head.html", "body.html", "script.html"))
page = ("<!doctype html>\n<html lang=\"en\">\n<head>\n" + meta + head.strip() + "\n</head>\n<body>\n" + body.strip() + "\n" + script.strip() +
        "\n</body>\n</html>\n")
open(SITE + "index.html", "w", encoding="utf-8").write(page)
print("docs/index.html", round(len(page.encode()) / 1024), "KB")
