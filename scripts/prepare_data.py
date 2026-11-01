"""Build the cache from the Soli zip (or an extracted folder) and write the data report."""
import argparse, json
from soli_attr.data import build_cache

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="path to SoliData.zip or a folder of .h5 files")
ap.add_argument("--out", default="cache")
a = ap.parse_args()
rep = build_cache(a.src, a.out)
print(json.dumps(rep, indent=2))
