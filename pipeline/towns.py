"""Town list shared by the pipeline. Order matters: new towns go at the end, since indices are stored in viewers' browsers (My Home)."""
TOWNS = ["Boxford", "Middleton", "Topsfield", "Georgetown", "North Andover", "Andover", "Lynn", "Lynnfield", "Ipswich", "Newburyport", "Gloucester", "Marblehead", "Swampscott", "Salem", "Beverly"]
TID = {"Boxford": 38, "Middleton": 184, "Topsfield": 298, "Georgetown": 105, "North Andover": 210, "Andover": 9, "Lynn": 163, "Lynnfield": 164,
       "Ipswich": 144, "Newburyport": 206, "Gloucester": 107,
       "Marblehead": 168, "Swampscott": 291, "Salem": 258, "Beverly": 30}
slug = lambda t: t.lower().replace(" ", "_")
from paths import D
