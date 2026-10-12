"""Write docs/site/data.json: morphology, synapse sites and chart data for the project page.

Usage (from bench/): uv run python site_data.py
"""

import json
import re

import numpy as np

rows = [r.split() for r in open("../data/neuromorpho/n419.CNG.swc") if r.strip() and r[0] != "#"]
ids = {int(r[0]): i for i, r in enumerate(rows)}
xyz = np.array([[float(v) for v in r[2:5]] for r in rows])
xyz -= xyz[0]
parent = [ids.get(int(r[6]), -1) for r in rows]
xor = json.load(open("xor_tree_seed2.json"))
bench = json.load(open("results.json"))
capacity = [dict(re.findall(r"(\w+)=(\S+)", line)) for line in open("xor_capacity.txt")]
traces = np.array(xor["traces"])[:, ::5]
data = {
    "xyz": np.round(xyz, 1).ravel().tolist(),
    "parent": parent,
    "sites": xor["sites"],
    "dt_frame": xor["dt"] * 5,
    "traces": {k: np.round(traces[i], 2).tolist() for i, k in enumerate(["00", "10", "01", "11"])},
    "capacity": [
        {"mode": c["mode"], "seed": int(c["seed"]), "gap": float(c["best_gap"])} for c in capacity
    ],
    "bench": [
        {
            "name": r["simulator"].split(" (")[0],
            "forward": r["forward_ms"],
            "gradient": r["gradient_ms"],
            "err": r["max_err_mV"] * 1e3,
        }
        for r in bench["rows"]
    ],
}
json.dump(data, open("../docs/site/data.json", "w"), separators=(",", ":"))
