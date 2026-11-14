#!/usr/bin/env python3
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
run = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else sorted(p for p in here.iterdir() if p.is_dir())[-1]


def conv(v):
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


records = [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(open(run / "results.csv"))]

AGG = {
    "smi_util_pct": "util", "smi_mem_util_pct": "mem_util", "smi_power_w": "power",
    "smi_sm_clock_mhz": "sm_clock", "smi_temp_c": "temp", "prof_kernel_busy_pct": "busy",
    "prof_matmul_pct": "matmul", "prof_attention_pct": "attention", "prof_comm_pct": "comm",
    "prof_memcpy_pct": "memcpy", "prof_other_pct": "other", "param_gb": "param_gb",
    "optstate_gb": "optstate_gb", "static_gb": "static_gb",
}
groups = defaultdict(list)
for row in csv.DictReader(open(run / "per_gpu.csv")):
    groups[(row["shape"], row["mb"], row["mode"])].append(row)
for r in records:
    rows = groups.get((r["shape"], str(int(r["mb"])), r["mode"]), [])
    for src, dst in AGG.items():
        vals = [float(x[src]) for x in rows if x.get(src) not in (None, "")]
        r[dst] = mean(vals) if vals else None

raw = json.load(open(run / "results.json"))
meta = raw["meta"]
meta["topology"] = re.sub(r"\x1b\[[0-9;]*m", "", meta.get("topology", ""))
first_ok = next(x for x in raw["results"] if x.get("status") == "ok")
for k in ("torch", "cuda", "amp_dtype"):
    meta[k] = first_ok.get(k)

payload = json.dumps({"meta": meta, "records": records}, separators=(",", ":")).replace("</", "<\\/")
html = (here / "dashboard_template.html").read_text().replace("/*__DATA__*/null", payload)
out = run / "dashboard.html"
out.write_text(html)
print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
