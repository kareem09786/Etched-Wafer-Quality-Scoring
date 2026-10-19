"""Simulated etched wafer images with controlled, known faults.

Each wafer is a rectangular ring-shaped etched channel (dark) on a lighter
surface. Three fault types can be switched on independently:

  bend   - the channel sides bow away from a straight line (amplitude in px)
  width  - the channel width varies along its length (fraction of nominal width)
  corner - the channel corners are rounded instead of square (radius in px)

Every image also gets a random position, a small rotation, an uneven
illumination gradient, optical blur and sensor noise. "Good" wafers still carry
tiny nominal imperfections, so the two classes overlap at the low-severity end.
"""
import numpy as np
import cv2

IMG_SIZE = 256
FAULT_TYPES = ("bend", "width", "corner")

# Acceptable imperfection on a good wafer, and severity range of a real fault.
NOMINAL = {"bend": (0.0, 1.0), "width": (0.0, 0.05), "corner": (0.0, 1.5)}
FAULT = {"bend": (1.5, 8.0), "width": (0.08, 0.30), "corner": (2.5, 10.0)}


def _sd_round_box(x, y, hx, hy, r):
    """Signed distance to a box with half sizes (hx, hy) and corner radius r."""
    qx = np.abs(x) - (hx - r)
    qy = np.abs(y) - (hy - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


def _profile(t, rng):
    """Smooth random profile over t in [0, 1] with peak magnitude 1."""
    p = np.zeros_like(t)
    for k in rng.integers(1, 4, size=2):
        p += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * (k * t + rng.random()))
    return p / np.abs(p).max()


def render_wafer(rng, bend=0.0, width=0.0, corner=0.0, size=IMG_SIZE):
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.uniform(-6, 6), size / 2 + rng.uniform(-6, 6)
    ang = np.deg2rad(rng.uniform(-6, 6))
    Ha, Hb, w0 = rng.uniform(78, 92), rng.uniform(58, 72), rng.uniform(13, 19)

    dx, dy = xs - cx, ys - cy
    u = np.cos(ang) * dx + np.sin(ang) * dy
    v = -np.sin(ang) * dx + np.cos(ang) * dy
    su = np.clip((u + Ha) / (2 * Ha), 0, 1)
    sv = np.clip((v + Hb) / (2 * Hb), 0, 1)

    # Bend: each of the four sides bows independently (zero displacement at corners).
    amp = bend * rng.uniform(0.6, 1.0, size=4) * rng.choice([-1, 1], size=4)
    ax = amp[0] * (1 - su) + amp[1] * su
    ay = amp[2] * (1 - sv) + amp[3] * sv
    uw = u - ax * np.sin(np.pi * sv)
    vw = v - ay * np.sin(np.pi * su)

    # Width: the inner boundary moves in and out along each side.
    wv = w0 * (1 + width * _profile(sv, rng))
    wh = w0 * (1 + width * _profile(su, rng))
    sd_out = _sd_round_box(uw, vw, Ha, Hb, corner)
    sd_in = _sd_round_box(uw * (Ha - w0) / (Ha - wv), vw * (Hb - w0) / (Hb - wh),
                          Ha - w0, Hb - w0, corner)
    coverage = np.minimum(np.clip(0.5 - sd_out, 0, 1), np.clip(0.5 + sd_in, 0, 1))

    surface, trench = rng.uniform(165, 190), rng.uniform(55, 85)
    img = surface + (trench - surface) * coverage
    img += rng.uniform(-18, 18) * (xs - size / 2) / size + rng.uniform(-18, 18) * (ys - size / 2) / size
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), rng.uniform(1.0, 2.0))
    img += rng.normal(0, rng.uniform(5, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def sample_params(rng, faulty):
    """Pick severities for one wafer. Returns (params, fault_tag)."""
    params = {k: rng.uniform(*NOMINAL[k]) for k in FAULT_TYPES}
    if not faulty:
        return params, "good"
    n = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
    chosen = sorted(rng.choice(FAULT_TYPES, size=n, replace=False))
    for k in chosen:
        params[k] = rng.uniform(*FAULT[k])
    return params, "+".join(chosen)


def generate_dataset(n_good, n_faulty, seed):
    """Return images (N, H, W) uint8, labels (1 = faulty) and fault tags."""
    rng = np.random.default_rng(seed)
    order = rng.permutation([0] * n_good + [1] * n_faulty)
    imgs, tags = [], []
    for label in order:
        params, tag = sample_params(rng, bool(label))
        imgs.append(render_wafer(rng, **params))
        tags.append(tag)
    return np.stack(imgs), order.astype(int), tags
