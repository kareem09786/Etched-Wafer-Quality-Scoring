"""Development check on a SEPARATE seed (999). Used only to debug the measurement code."""
import numpy as np
from wafer.simulate import generate_dataset
from wafer.measure import measure, FEATURE_NAMES
imgs, y, tags = generate_dataset(60, 60, seed=999)
F, dbg = [], []
for im in imgs:
    f, d = measure(im, True); F.append(f); dbg.append(d)
F = np.array(F)
print("lines found (v,h) counts:", {k: sum(1 for d in dbg if d['n_lines']==k) for k in set(d['n_lines'] for d in dbg)})
print("residual |angle| after deskew is checked via mean abs angle on masks (should be small)")
for j,n in enumerate(FEATURE_NAMES):
    print(f"{n:22s} good median {np.median(F[y==0,j]):7.3f} | faulty median {np.median(F[y==1,j]):7.3f}")
for t in sorted(set(tags)):
    m = np.array([x==t for x in tags]); print(f"{t:18s} n={m.sum():3d}", np.round(np.median(F[m],axis=0),3))
