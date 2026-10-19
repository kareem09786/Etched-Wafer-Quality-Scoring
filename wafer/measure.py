"""Measure channel straightness, corner squareness and width variation.

Pipeline per image:
  1. Gaussian blur + Otsu threshold  -> binary channel mask
  2. Canny edge detection            -> channel boundaries
  3. Hough transform                 -> straight-line model of the boundaries
     (the dominant angle is used to straighten the image first)
  4. Measures
       straightness_rms_px : RMS distance of boundary pixels from the fitted
                             straight lines (corner zones excluded)
       corner_offset_px    : mean distance from each ideal corner (where two
                             fitted lines meet) to the nearest boundary pixel;
                             a square corner gives ~0, a rounded corner does not
       width_variation     : coefficient of variation of the channel width
                             along each of the four sides, averaged
Larger values mean a worse wafer. If the line fit fails, a penalty value is used.
"""
import cv2
import numpy as np

FEATURE_NAMES = ["straightness_rms_px", "corner_offset_px", "width_variation"]
CORNER_EXCLUDE_PX = 14
PENALTY = {"straightness_rms_px": 10.0, "corner_offset_px": 20.0, "width_variation": 0.5}


def segment(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1.5)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def dominant_angle(edges):
    """Rotation (degrees) of the rectangle, from length-weighted Hough segments."""
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=25, maxLineGap=4)
    if segs is None:
        return 0.0
    s = segs[:, 0, :].astype(float)
    dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
    length, theta = np.hypot(dx, dy), np.arctan2(dy, dx)
    # multiplying the angle by 4 folds 0/90/180/270 degrees onto the same value
    return float(np.degrees(np.arctan2((length * np.sin(4 * theta)).sum(),
                                       (length * np.cos(4 * theta)).sum()) / 4))


def find_lines(edges, shape, min_sep=8):
    """Up to 4 near-vertical and 4 near-horizontal Hough lines, sorted by position."""
    h, w = shape
    groups = {"v": [], "h": []}
    found = cv2.HoughLines(edges, 1, np.pi / 180, 25)
    if found is None:
        return groups
    for rho, theta in found[:, 0, :]:
        if abs(theta - np.pi / 2) < np.radians(10):
            key, pos = "h", (rho - (w / 2) * np.cos(theta)) / np.sin(theta)
        elif theta < np.radians(10) or theta > np.radians(170):
            key, pos = "v", (rho - (h / 2) * np.sin(theta)) / np.cos(theta)
        else:
            continue
        if len(groups[key]) < 4 and all(abs(pos - p) >= min_sep for _, _, p in groups[key]):
            groups[key].append((float(rho), float(theta), float(pos)))
    for key in groups:
        groups[key].sort(key=lambda t: t[2])
    return groups


def _intersect(a, b):
    A = np.array([[np.cos(a[1]), np.sin(a[1])], [np.cos(b[1]), np.sin(b[1])]])
    return np.linalg.solve(A, [a[0], b[0]])


def _central_gap(profile):
    """Start/end of the hollow centre: the longest run of short projection values."""
    tall = np.nonzero(profile > 0.6 * profile.max())[0]
    if len(tall) < 2:
        return None
    i = int(np.argmax(np.diff(tall)))
    return tall[i] + 1, tall[i + 1] - 1


def width_variation(mask):
    m = mask > 0
    gx, gy = _central_gap(m.sum(0).astype(float)), _central_gap(m.sum(1).astype(float))
    if gx is None or gy is None:
        return PENALTY["width_variation"]
    margin = 8
    x0, x1, y0, y1 = gx[0] + margin, gx[1] - margin, gy[0] + margin, gy[1] - margin
    if x1 - x0 < 10 or y1 - y0 < 10:
        return PENALTY["width_variation"]
    cx, cy = (gx[0] + gx[1]) // 2, (gy[0] + gy[1]) // 2
    sides = [m[y0:y1, :cx].sum(1), m[y0:y1, cx:].sum(1),   # left, right
             m[:cy, x0:x1].sum(0), m[cy:, x0:x1].sum(0)]   # top, bottom
    cvs = [s.std() / s.mean() for s in sides if s.mean() > 0]
    return float(np.mean(cvs)) if cvs else PENALTY["width_variation"]


def measure(img, return_debug=False):
    h, w = img.shape
    mask = segment(img)
    angle = dominant_angle(cv2.Canny(mask, 100, 200))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    mask = (cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR) > 127).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 100, 200)

    groups = find_lines(edges, mask.shape)
    V, H = groups["v"], groups["h"]
    ys, xs = np.nonzero(edges)
    E = np.column_stack([xs, ys]).astype(float)

    corners = []
    if len(V) >= 2 and len(H) >= 2:
        corners += [_intersect(a, b) for a in (V[0], V[-1]) for b in (H[0], H[-1])]
    if len(V) == 4 and len(H) == 4:
        corners += [_intersect(a, b) for a in (V[1], V[2]) for b in (H[1], H[2])]

    if corners and len(E):
        corner_offset = float(np.mean([np.hypot(*(E - c).T).min() for c in corners]))
    else:
        corner_offset = PENALTY["corner_offset_px"]

    lines = V + H
    straight = PENALTY["straightness_rms_px"]
    if lines and len(E):
        keep = np.ones(len(E), bool)
        for c in corners:
            keep &= np.hypot(*(E - c).T) > CORNER_EXCLUDE_PX
        if keep.sum() >= 20:
            P = E[keep]
            res = np.min([np.abs(P[:, 0] * np.cos(t) + P[:, 1] * np.sin(t) - r) for r, t, _ in lines], axis=0)
            straight = float(np.sqrt(np.mean(res ** 2)))

    feats = np.array([straight, corner_offset, width_variation(mask)])
    if return_debug:
        return feats, {"mask": mask, "edges": edges, "lines": lines, "corners": corners,
                       "angle": angle, "n_lines": (len(V), len(H))}
    return feats
