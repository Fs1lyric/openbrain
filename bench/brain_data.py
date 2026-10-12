"""Write docs/site/brain.bin: a point cloud of the fsaverage6 pial surface (FreeSurfer, fetched by
nilearn), sampled densely on sulcal walls and gyral crowns so the folds read.

Layout: uint32 count, then count x (int16 x, y, z in 0.1 mm, uint8 colour index, uint8 pad).
Usage (from bench/): uv run python brain_data.py
"""

import struct
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import datasets

fs = datasets.fetch_surf_fsaverage("fsaverage6")
rng = np.random.default_rng(7)
points, curvs = [], []
for hemi in ["left", "right"]:
    xyz = nib.load(fs[f"pial_{hemi}"]).darrays[0].data
    curv = nib.load(fs[f"curv_{hemi}"]).darrays[0].data
    points.append(xyz)
    curvs.append(curv)
xyz, curv = np.concatenate(points), np.concatenate(curvs)
xyz -= (xyz.max(0) + xyz.min(0)) / 2
# keep strongly curved vertices (fold edges) more often than flat ones
keep = rng.random(len(xyz)) < 0.04 + 0.5 * np.clip(np.abs(curv) / 0.3, 0, 1) ** 2
xyz = xyz[keep]
# jitter within ~0.5 mm so the cloud doesn't show the mesh lattice
xyz = xyz + rng.normal(0, 0.4, xyz.shape)
colour = rng.choice(3, size=len(xyz), p=[0.45, 0.33, 0.22])
out = bytearray(struct.pack("<I", len(xyz)))
for (x, y, z), c in zip(np.round(xyz * 10).astype(int), colour, strict=True):
    out += struct.pack("<hhhBB", x, y, z, c, 0)
Path("../docs/site/brain.bin").write_bytes(bytes(out))
print(len(xyz), "points", len(out), "bytes")
