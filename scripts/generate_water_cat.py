"""
Chordal Axis Transform (CAT) for water canal / river centerline extraction.

Core algorithm (following Shewchuk + Prasad's CAT):
  1. Binary mask → boundary contour polygon
  2. Constrained Delaunay Triangulation (CDT) — enforces boundary edges
  3. Triangle classification by number of constrained edges:
        TERMINAL  (2 boundary edges): branch tips
        SLEEVE    (1 boundary edge ): canal shaft
        JUNCTION  (0 boundary edges): branching nodes
  4. Chordal axis segments: midpoints of interior edges connected per triangle type
  5. Adaptive graph pruning via EDT (Chamfer distance):
        threshold = PRUNE_K × local_radius_at_junction
        Wide rivers → large local_radius → tolerant of longer spurs
        Narrow canals → small local_radius → tight threshold
  6. Taubin λ/μ smoothing (non-shrinking) of axis polylines

Scale-adaptiveness: Delaunay circumradius scales with local canal width.
  - Wide river  → large JUNCTION triangle clusters  → smooth branch node
  - Narrow canal → thin SLEEVE chain               → precise 1-px centerline
  - Width change → smooth transition, no staircase artifact (unlike Zhang-Suen)

Output per VI per dataset in docs/images/water_analysis/{run_id}/:
  {VI}_cat_mesh.png  — colored CDT triangulation overlay (T=green S=blue J=red)
  {VI}_cat.png       — dark RGB + fill + glowing CAT centerlines
  {VI}_compare.png   — side-by-side: Zhang-Suen skeleton vs CAT

Authors: Vietnam Hyperspectral EDA project
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay as ScipyDelaunay
from scipy.ndimage import (binary_fill_holes, distance_transform_edt,
                           gaussian_filter, gaussian_filter1d)
from skimage.draw import line as draw_line
from skimage.filters import frangi
from skimage.measure import find_contours, label as sklabel
from skimage.morphology import (binary_closing, binary_dilation, disk,
                                remove_small_objects, skeletonize)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.ingest import load_cube
from pipeline.utils import nearest_band_idx

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "20250301_143913_32_4001": ROOT / "Planet_Data/20250301_143913_32_4001_ortho_sr_hdf5.h5",
    "20250407_035527_47_4001": ROOT / "Planet_Data/20250407_035527_47_4001_ortho_sr_hdf5.h5",
    "20250407_035451_00_4001": ROOT / "Planet_Data/20250407_035451_00_4001_ortho_sr_hdf5.h5",
}

WATER_VIS = {
    "WBI":  {"water_high": True,  "prior_pct": 30},
    "WI":   {"water_high": True,  "prior_pct": 30},
    "MSI":  {"water_high": False, "prior_pct": 30},
    "HDWI": {"water_high": True,  "prior_pct": 30},
}

FRANGI_SIGMAS   = [2, 3, 4, 6, 8, 11, 16]
BLOB_SIGMAS     = [4, 6, 8, 11, 15, 20]
RGB_NM          = (665, 560, 470)
DPI             = 130

N_CONTOUR_PTS   = 1500  # max contour sample points per component (higher = finer triangles = better curves)
MIN_COMP_PX     = 400   # skip components smaller than this
PRUNE_K         = 1.5   # prune leaf if length < PRUNE_K × local_radius
SMOOTH_LAMBDA   = 0.50  # Taubin positive step
SMOOTH_MU       = -0.53 # Taubin negative step (|mu| > |lambda| to prevent shrink)
SMOOTH_ITER     = 3     # Taubin iterations (reduced: more iterations straighten curves)
RESAMPLE_SPACING = 1.5  # resample axis paths at this pixel interval before Taubin
CURVE_WEIGHT    = 3.5   # adaptive sampling: extra density at high-curvature (bend) regions

# Triangle type colors (for CDT mesh visualization)
TRI_COLORS = {"T": "#22cc55", "S": "#4499ff", "J": "#ff3333"}
TRI_ALPHAS  = {"T": 0.60,     "S": 0.28,     "J": 0.70}


# ── Image helpers ──────────────────────────────────────────────────────────────

def load_raster(path):
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def clip_norm(arr, lo=0.5, hi=99.5):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vlo, vhi = np.percentile(v, [lo, hi])
    out = np.where(np.isfinite(arr), (arr - vlo) / (vhi - vlo + 1e-8), 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_rgb(cube, wavelengths):
    rgb = np.stack(
        [cube[nearest_band_idx(wavelengths, nm)] for nm in RGB_NM], axis=-1
    ).astype(np.float32)
    for c in range(3):
        lo, hi = np.nanpercentile(rgb[..., c], [2, 98])
        rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return rgb


def geometric_mask(vi_map, water_high, prior_pct=30):
    """Frangi + LoG blob → cleaned binary mask (identical to export_water_web.py)."""
    from scipy.ndimage import gaussian_laplace
    img = clip_norm(vi_map)
    if not water_high:
        img = 1.0 - img
    tube = frangi(img, sigmas=FRANGI_SIGMAS, alpha=0.5, beta=0.5,
                  gamma=None, black_ridges=False)
    mx = tube.max()
    tube = (tube / mx if mx > 0 else tube).astype(np.float32)
    blobs = []
    for s in BLOB_SIGMAS:
        r = -gaussian_laplace(img.astype(np.float64), sigma=s) * s ** 2
        blobs.append(np.clip(r, 0, None))
    blob = np.stack(blobs).max(axis=0).astype(np.float32)
    mx = blob.max()
    blob = blob / mx if mx > 0 else blob
    combined = 0.60 * tube + 0.40 * blob
    thr = np.percentile(combined, 85)
    geo = combined > thr
    raw = vi_map[np.isfinite(vi_map)]
    p_thr = np.percentile(raw, (100 - prior_pct) if water_high else prior_pct)
    prior = (vi_map > p_thr) if water_high else (vi_map < p_thr)
    geo = geo & prior & np.isfinite(vi_map)
    geo = binary_closing(geo, disk(3))
    geo = remove_small_objects(geo, min_size=150)
    geo = binary_fill_holes(geo)
    return geo.astype(bool)


# ── CAT core ───────────────────────────────────────────────────────────────────

def curvature_adaptive_sample(contour, max_n, curve_weight=CURVE_WEIGHT):
    """
    Sample contour in arc-length space with curvature-adaptive density.

    Key design decisions:
    - Uses arc-length parameterization (not raw index steps) so it works
      correctly regardless of whether find_contours returns 0.7-1px points
      or 2-3px points.
    - Enforces a hard minimum spacing of MIN_SPACING=2.5px between samples
      to avoid degenerate Delaunay triangles.
    - At high-curvature (bend) regions: spacing reduced by up to 1/(1+curve_weight)
      → more triangles → shorter axis chords → centerline follows the bend.
    - At straight regions: uniform spacing = total_arc / max_n (or MIN_SPACING floor).

    curve_weight=3.5 means: at the sharpest bend, spacing is 1/4.5 of the base.
    But always ≥ MIN_SPACING=2.5px.
    """
    n = len(contour)
    if n < 6:
        return contour

    # Arc-length parameterization
    segs = np.diff(contour, axis=0)
    seg_lens = np.linalg.norm(segs, axis=1)
    total_arc = seg_lens.sum()
    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    # Base spacing: uniform distribution gives max_n points
    # Never go below MIN_SPACING to ensure triangle quality
    MIN_SPACING = 2.5
    base_spacing = max(MIN_SPACING, total_arc / max_n)

    # Curvature proxy: angle change between consecutive tangent vectors
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        v1 = contour[i] - contour[i - 1]
        v2 = contour[i + 1] - contour[i]
        l1 = np.linalg.norm(v1) + 1e-8
        l2 = np.linalg.norm(v2) + 1e-8
        kappa[i] = 1.0 - np.clip(np.dot(v1 / l1, v2 / l2), -1.0, 1.0)

    # Smooth kappa so we don't over-concentrate on noise spikes
    kappa = gaussian_filter1d(kappa, sigma=max(2, n // 80))
    kappa_norm = kappa / (kappa.max() + 1e-8)

    # Walk along arc length, placing samples with adaptive spacing
    indices = [0]
    next_target = cum_arc[0] + _adaptive_spacing(kappa_norm[0], base_spacing, curve_weight, MIN_SPACING)

    for i in range(1, n):
        if cum_arc[i] >= next_target:
            indices.append(i)
            sp = _adaptive_spacing(kappa_norm[min(i, n - 1)], base_spacing, curve_weight, MIN_SPACING)
            next_target = cum_arc[i] + sp

    if not indices or indices[-1] != n - 1:
        indices.append(n - 1)

    return contour[np.array(indices)]


def _adaptive_spacing(kappa_norm_i, base_spacing, curve_weight, min_sp):
    """Compute adaptive spacing at a single point. Floored at min_sp."""
    return max(min_sp, base_spacing / (1.0 + curve_weight * float(kappa_norm_i)))


def resample_path_uniform(path, spacing=RESAMPLE_SPACING):
    """
    Resample a path (list of (row,col) tuples) at uniform arc-length intervals.

    Purpose: before Taubin smoothing, give the smoother many control points
    so it can preserve local curvature instead of pulling sharp bends straight.
    Without this, sparse graph nodes become the only control points → Taubin
    averages over wide spans → corners get cut.
    """
    if len(path) < 2:
        return path
    pts = np.array([[p[0], p[1]] for p in path], dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len < spacing:
        return path

    n_new = max(3, int(total_len / spacing) + 1)
    new_ts = np.linspace(0.0, total_len, n_new)
    new_pts = []
    for t in new_ts:
        idx = int(np.searchsorted(cum_len, t, side="right")) - 1
        idx = np.clip(idx, 0, len(seg_lens) - 1)
        frac = np.clip((t - cum_len[idx]) / (seg_lens[idx] + 1e-10), 0.0, 1.0)
        p = pts[idx] + frac * diffs[idx]
        new_pts.append(tuple(p))
    return new_pts


def build_cdt(comp_mask, n_pts=N_CONTOUR_PTS):
    """
    Build approximate Constrained Delaunay Triangulation via scipy.spatial.Delaunay.

    True CDT enforces boundary edges as mesh edges.  Here we approximate:
      1. Sample N points uniformly around the boundary contour.
      2. Run standard Delaunay triangulation on those boundary points.
      3. Filter triangles whose centroid falls outside the mask.
      4. Detect "boundary edges": edge (i,j) is constrained if
           (a) i and j are consecutive boundary samples, OR
           (b) the edge midpoint has EDT < BND_THRESH (close to mask edge).

    This avoids the C-extension `triangle` library which can segfault on
    complex contours.  Results are nearly identical to true CDT for the
    elongated canal shapes we process.

    Returns: (verts_xy, triangles, bnd_frozensets) or None
    Coordinate convention: verts in (x=col, y=row); image ops need (row,col).
    """
    H, W = comp_mask.shape

    contours = find_contours(comp_mask.astype(float), 0.5)
    if not contours:
        return None

    contour = max(contours, key=len)   # (N_raw, 2) as (row, col)

    # Curvature-adaptive arc-length sampling (enforces ≥2.5px spacing internally)
    # More points at bends → finer triangles → shorter axis chords → better curves
    pts_rc = curvature_adaptive_sample(contour, n_pts)
    N = len(pts_rc)
    if N < 6:
        return None

    pts_xy = pts_rc[:, ::-1].copy()   # (N, 2) as (x=col, y=row)

    # EDT inside component — used for boundary edge detection
    dt_comp = distance_transform_edt(comp_mask)

    # Delaunay on boundary sample points only
    try:
        tri = ScipyDelaunay(pts_xy)
    except Exception:
        return None

    verts = pts_xy
    tris  = tri.simplices

    # Filter triangles: centroid must be inside comp_mask
    keep = []
    for simplex in tris:
        cx = int(np.clip(verts[simplex, 0].mean(), 0, W - 1))
        cy = int(np.clip(verts[simplex, 1].mean(), 0, H - 1))
        if comp_mask[cy, cx]:
            keep.append(simplex)
    if not keep:
        return None
    tris = np.array(keep)

    # Primary boundary edges: consecutive contour sample pairs
    bnd_set = set()
    for i in range(N):
        bnd_set.add(frozenset({i, (i + 1) % N}))

    # Secondary boundary edges: midpoint close to mask boundary (EDT < BND_THRESH)
    # This compensates for non-constrained Delaunay missing some boundary edges.
    BND_THRESH = 3.0
    seen_edges = set()
    for simplex in tris:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        for a, b in ((i, j), (j, k), (i, k)):
            e = frozenset({a, b})
            if e in seen_edges or e in bnd_set:
                continue
            seen_edges.add(e)
            mx = int(np.clip((verts[a, 0] + verts[b, 0]) / 2, 0, W - 1))
            my = int(np.clip((verts[a, 1] + verts[b, 1]) / 2, 0, H - 1))
            if dt_comp[my, mx] < BND_THRESH:
                bnd_set.add(e)

    return verts, tris, bnd_set


def classify_triangles(tris, bnd_set):
    """
    TERMINAL (T): 2 constrained boundary edges  → branch tip
    SLEEVE   (S): 1 constrained boundary edge   → canal shaft
    JUNCTION (J): 0 constrained boundary edges  → branching point

    This classification is the core of CAT and directly determines how wide/narrow
    each section of the canal is (JUNCTION = wide / no visible boundary = interior).
    """
    labels = []
    for tri in tris:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        n_bnd = sum(1 for e in (
            frozenset({i, j}), frozenset({j, k}), frozenset({i, k})
        ) if e in bnd_set)

        if n_bnd >= 2:
            labels.append("T")
        elif n_bnd == 1:
            labels.append("S")
        else:
            labels.append("J")
    return labels


def extract_chordal_axis(verts, tris, labels, bnd_set):
    """
    Build chordal axis as a list of ((row1,col1), (row2,col2)) segment pairs.

    Rule per triangle type:
      SLEEVE (S)   → connect midpoints of its 2 interior edges
      JUNCTION (J) → connect centroid to midpoints of all 3 interior edges  (Y/T-node)
      TERMINAL (T) → connect interior edge midpoint to the shared "tip" vertex

    All coordinates returned in (row, col) = image convention.
    """
    segments = []

    for tri, lab in zip(tris, labels):
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])

        # Boundary status of each edge
        b01 = frozenset({i, j}) in bnd_set
        b12 = frozenset({j, k}) in bnd_set
        b02 = frozenset({i, k}) in bnd_set

        def mid_rc(va, vb):
            """Midpoint of edge (va, vb) in (row, col) image coords."""
            mx = (verts[va, 0] + verts[vb, 0]) * 0.5  # x = col
            my = (verts[va, 1] + verts[vb, 1]) * 0.5  # y = row
            return (my, mx)

        interior_mids = []
        if not b01:
            interior_mids.append(mid_rc(i, j))
        if not b12:
            interior_mids.append(mid_rc(j, k))
        if not b02:
            interior_mids.append(mid_rc(i, k))

        if lab == "S" and len(interior_mids) == 2:
            # Sleeve: single chord connecting 2 adjacent triangles
            segments.append((interior_mids[0], interior_mids[1]))

        elif lab == "J" and len(interior_mids) == 3:
            # Junction: centroid connects to all 3 interior edge midpoints
            cen_x = (verts[i, 0] + verts[j, 0] + verts[k, 0]) / 3.0
            cen_y = (verts[i, 1] + verts[j, 1] + verts[k, 1]) / 3.0
            cen_rc = (cen_y, cen_x)
            for m in interior_mids:
                segments.append((cen_rc, m))

        elif lab == "T" and len(interior_mids) == 1:
            # Terminal: find the "tip" vertex (shared by the 2 boundary edges)
            bnd_vs = []
            if b01:
                bnd_vs += [i, j]
            if b12:
                bnd_vs += [j, k]
            if b02:
                bnd_vs += [i, k]
            cnt = Counter(bnd_vs)
            if cnt:
                tip_v = max(cnt, key=lambda v: cnt[v])
                tip_rc = (float(verts[tip_v, 1]), float(verts[tip_v, 0]))
                segments.append((interior_mids[0], tip_rc))

    return segments


def segs_to_mask(segs, H, W):
    """Rasterize (row,col) segment list to binary image."""
    mask = np.zeros((H, W), dtype=bool)
    for (r1, c1), (r2, c2) in segs:
        r1, c1 = int(round(r1)), int(round(c1))
        r2, c2 = int(round(r2)), int(round(c2))
        r1, c1 = np.clip(r1, 0, H - 1), np.clip(c1, 0, W - 1)
        r2, c2 = np.clip(r2, 0, H - 1), np.clip(c2, 0, W - 1)
        if r1 == r2 and c1 == c2:
            mask[r1, c1] = True
        else:
            rr, cc = draw_line(r1, c1, r2, c2)
            mask[np.clip(rr, 0, H - 1), np.clip(cc, 0, W - 1)] = True
    return mask


# ── Graph pruning (adaptive via EDT) ──────────────────────────────────────────

def build_graph(segs, H, W):
    """Build networkx graph from segment list; nodes = (row, col) tuples."""
    G = nx.Graph()
    for (r1, c1), (r2, c2) in segs:
        r1, c1 = np.clip(int(round(r1)), 0, H - 1), np.clip(int(round(c1)), 0, W - 1)
        r2, c2 = np.clip(int(round(r2)), 0, H - 1), np.clip(int(round(c2)), 0, W - 1)
        n1, n2 = (r1, c1), (r2, c2)
        if n1 == n2:
            continue
        d = float(np.hypot(r2 - r1, c2 - c1))
        if G.has_edge(n1, n2):
            G[n1][n2]["w"] = min(G[n1][n2]["w"], d)
        else:
            G.add_edge(n1, n2, w=d)
    return G


def prune_graph(G, dt_field, prune_k=PRUNE_K, max_iter=25):
    """
    Iterative adaptive pruning.

    For each leaf branch:
      length_of_branch < PRUNE_K × radius_at_junction  →  prune

    EDT distance field gives local canal radius at each pixel.
    Wide canal junction → large radius → long spurs allowed.
    Narrow passage → small radius → short spurs pruned aggressively.
    This is scale-adaptiveness: no fixed pixel threshold.
    """
    if G.number_of_nodes() == 0:
        return G

    H, W = dt_field.shape

    for _ in range(max_iter):
        leaves = [n for n in list(G.nodes()) if G.degree(n) == 1]
        pruned_any = False

        for leaf in leaves:
            if leaf not in G or G.degree(leaf) != 1:
                continue

            # Walk branch from leaf to first junction/dead-end
            path = [leaf]
            branch_len = 0.0
            prev = None

            for _step in range(2000):
                curr = path[-1]
                nbrs = [n for n in G.neighbors(curr) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                branch_len += G[curr][nxt]["w"]
                path.append(nxt)
                prev = curr
                if G.degree(nxt) != 2:
                    break  # junction or another leaf

            junction = path[-1]
            jr, jc = np.clip(junction[0], 0, H - 1), np.clip(junction[1], 0, W - 1)
            local_r = float(dt_field[jr, jc])
            threshold = prune_k * max(local_r, 2.0)

            if branch_len < threshold:
                for node in path[:-1]:  # keep junction node
                    if node in G:
                        G.remove_node(node)
                pruned_any = True

        if not pruned_any:
            break

    return G


# ── Taubin λ/μ smoothing (non-shrinking) ──────────────────────────────────────

def taubin_smooth_path(pts, lam=SMOOTH_LAMBDA, mu=SMOOTH_MU, n_iter=SMOOTH_ITER):
    """
    Taubin smoothing for an ordered list of (row,col) points.
    Alternates positive (λ) and negative (μ) Laplacian steps.
    Unlike simple averaging (box filter), Taubin does not shrink the curve.
    """
    if len(pts) <= 2:
        return pts
    pts = np.array(pts, dtype=float)
    n = len(pts)
    ends = pts[[0, -1]].copy()

    for _ in range(n_iter):
        # Positive step
        lap = np.zeros_like(pts)
        lap[1:-1] = (pts[:-2] + pts[2:]) / 2.0 - pts[1:-1]
        pts = pts + lam * lap
        pts[[0, -1]] = ends  # fix endpoints

        # Negative step
        lap = np.zeros_like(pts)
        lap[1:-1] = (pts[:-2] + pts[2:]) / 2.0 - pts[1:-1]
        pts = pts + mu * lap
        pts[[0, -1]] = ends

    return [tuple(p) for p in pts]


def smooth_graph_paths(G):
    """
    Decompose graph into simple paths (between degree-1 or degree≥3 nodes)
    and apply Taubin smoothing to each path.
    Returns: new list of (node, node) segments approximating the smoothed paths.
    """
    if G.number_of_edges() == 0:
        return []

    # Find path endpoints: degree ≠ 2 (leaves or junctions)
    endpoints = {n for n in G.nodes() if G.degree(n) != 2}
    if not endpoints:
        # All degree-2 → closed loop; pick one arbitrarily
        endpoints = {next(iter(G.nodes()))}

    visited_edges = set()
    smoothed_segs = []

    for start in endpoints:
        for nbr in list(G.neighbors(start)):
            e_key = frozenset({start, nbr})
            if e_key in visited_edges:
                continue
            visited_edges.add(e_key)

            # Trace path
            path = [start, nbr]
            prev = start
            curr = nbr
            for _ in range(10000):
                if G.degree(curr) != 2:
                    break  # reached junction or another endpoint
                next_ns = [n for n in G.neighbors(curr) if n != prev]
                if not next_ns:
                    break
                nxt = next_ns[0]
                e_key2 = frozenset({curr, nxt})
                if e_key2 in visited_edges:
                    break
                visited_edges.add(e_key2)
                path.append(nxt)
                prev, curr = curr, nxt

            # Resample path at fine intervals before Taubin so the smoother
            # has enough control points to preserve sharp bends.
            # Without this, sparse graph nodes cause Taubin to cut corners.
            if len(path) >= 2:
                resampled = resample_path_uniform(path)
            else:
                resampled = path

            if len(resampled) >= 3:
                smooth = taubin_smooth_path(resampled)
                for a, b in zip(smooth[:-1], smooth[1:]):
                    smoothed_segs.append((a, b))
            else:
                for a, b in zip(resampled[:-1], resampled[1:]):
                    smoothed_segs.append((a, b))

    return smoothed_segs


# ── Rendering ─────────────────────────────────────────────────────────────────

def make_glow_overlay(rgb, geo_mask, axis_mask):
    """Dark RGB + cyan fill + glowing yellow-white axis centerlines."""
    out = rgb * 0.60
    fill_c = np.array([0.0, 0.80, 1.0])
    out[geo_mask] = 0.70 * out[geo_mask] + 0.30 * fill_c

    if axis_mask.any():
        gf = gaussian_filter(axis_mask.astype(np.float32), sigma=2.5)
        mx = gf.max()
        if mx > 0:
            gf /= mx
        glow_c = np.array([0.0, 0.95, 1.0])
        for ci, v in enumerate(glow_c):
            out[..., ci] = out[..., ci] * (1 - 0.60 * gf) + v * 0.60 * gf
        core = binary_dilation(axis_mask, disk(1))
        out[core] = np.array([1.0, 1.0, 0.70])  # near-white yellow core

    return np.clip(out, 0, 1)


def save_img(arr, path, dpi=DPI, facecolor="black"):
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(np.clip(arr, 0, 1), interpolation="nearest")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor=facecolor)
    plt.close(fig)


def save_cdt_mesh(all_verts, all_tris, all_labels, all_segs_rc, H, W, rgb, path, dpi=DPI):
    """
    Render colored CDT triangulation + chordal axis on dimmed RGB.

    Color code:
      Green  (Terminal): branch tips — triangle has 2 constrained boundary edges
      Blue   (Sleeve):   canal shaft — 1 constrained edge
      Red    (Junction): branching node — 0 constrained edges (fully interior)
    """
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Dimmed RGB background
    ax.imshow(rgb * 0.45, origin="upper", interpolation="nearest")

    # Colored triangles — iterate per-component vertex/triangle sets
    v_offset = 0
    lab_idx  = 0
    for verts, tris in zip(all_verts, all_tris):
        n_tris = len(tris)
        comp_labels = all_labels[lab_idx: lab_idx + n_tris]
        lab_idx += n_tris
        for tri, lab in zip(tris, comp_labels):
            # verts in (x=col, y=row); matplotlib axes match imshow (x=col, y=row)
            xs = verts[tri, 0]
            ys = verts[tri, 1]
            poly = plt.Polygon(
                list(zip(xs, ys)),
                facecolor=TRI_COLORS[lab],
                edgecolor="white",
                linewidth=0.20,
                alpha=TRI_ALPHAS[lab],
            )
            ax.add_patch(poly)
        v_offset += len(verts)

    # Chordal axis (in (row,col) = (y,x) for matplotlib after imshow)
    for (r1, c1), (r2, c2) in all_segs_rc:
        ax.plot([c1, c2], [r1, r2], color="yellow", linewidth=1.0, alpha=0.88)

    # Legend
    patches = [
        mpatches.Patch(color=TRI_COLORS["T"], alpha=0.8, label="Terminal (tip)"),
        mpatches.Patch(color=TRI_COLORS["S"], alpha=0.55, label="Sleeve (shaft)"),
        mpatches.Patch(color=TRI_COLORS["J"], alpha=0.85, label="Junction (branch node)"),
        mpatches.Patch(color="yellow", label="Chordal Axis"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=5.5,
              facecolor="#111", labelcolor="white", edgecolor="gray",
              markerscale=0.7, handlelength=1.4, framealpha=0.88)

    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_cat(geo_mask, dt_field, H, W):
    """
    Run full CAT pipeline on a binary water mask.

    Returns:
        cat_axis_mask: (H,W) bool  — rasterized pruned+smoothed axis
        all_verts, all_tris, all_labels: lists per component (for mesh viz)
        smoothed_segs: list of ((r,c),(r,c)) after Taubin smoothing
    """
    labeled = sklabel(geo_mask)
    n_comps = labeled.max()

    all_verts, all_tris, all_labels = [], [], []
    raw_segs = []

    for cid in range(1, n_comps + 1):
        comp = labeled == cid
        if comp.sum() < MIN_COMP_PX:
            continue

        result = build_cdt(comp)
        if result is None:
            continue

        verts, tris, bnd_set = result
        labels = classify_triangles(tris, bnd_set)
        segs = extract_chordal_axis(verts, tris, labels, bnd_set)

        # Graph pruning (adaptive, uses EDT)
        G = build_graph(segs, H, W)
        G = prune_graph(G, dt_field)

        # Taubin smoothing of remaining paths
        smooth = smooth_graph_paths(G)

        raw_segs.extend(smooth)
        all_verts.append(verts)
        all_tris.append(tris)
        all_labels.extend(labels)

    axis_mask = segs_to_mask(raw_segs, H, W)

    return axis_mask, all_verts, all_tris, all_labels, raw_segs


def process(run_id, h5_path):
    web_dir    = ROOT / "docs" / "images" / "water_analysis" / run_id
    raster_dir = ROOT / "outputs" / run_id / "rasters"
    web_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {run_id}")
    print(f"{'='*60}")

    print("  Loading HDF5...")
    cube, wl, _, _ = load_cube(h5_path)
    rgb = make_rgb(cube, wl)
    H, W = rgb.shape[:2]
    del cube

    for vi, cfg in WATER_VIS.items():
        rp = raster_dir / f"{vi}.tif"
        if not rp.exists():
            print(f"  [skip] {vi}.tif missing")
            continue

        vi_map = load_raster(rp)
        if vi_map.shape != (H, W):
            from PIL import Image as PILImage
            vi_map = np.array(
                PILImage.fromarray(vi_map).resize((W, H), PILImage.BILINEAR))

        print(f"  {vi}: geo...", end=" ", flush=True)
        geo = geometric_mask(vi_map, cfg["water_high"], cfg["prior_pct"])
        dt  = distance_transform_edt(geo)   # Chamfer-like distance field

        print("CDT+classify...", end=" ", flush=True)
        cat_axis, all_v, all_t, all_lbl, smooth_segs = run_cat(geo, dt, H, W)

        n_T = all_lbl.count("T")
        n_S = all_lbl.count("S")
        n_J = all_lbl.count("J")
        print(f"T={n_T} S={n_S} J={n_J}", end="  ", flush=True)

        # ── Image 1: glowing CAT overlay (same style as skeleton) ──────────────
        print("render-cat...", end=" ", flush=True)
        save_img(make_glow_overlay(rgb, geo, cat_axis), web_dir / f"{vi}_cat.png")

        # ── Image 2: colored CDT mesh ──────────────────────────────────────────
        print("render-mesh...", end=" ", flush=True)
        if all_v:
            save_cdt_mesh(all_v, all_t, all_lbl, smooth_segs, H, W, rgb,
                          web_dir / f"{vi}_cat_mesh.png")

        # ── Image 3: ZS vs CAT comparison (2-panel) ────────────────────────────
        print("compare...", end=" ", flush=True)
        zs_axis = skeletonize(geo)
        zs_img  = make_glow_overlay(rgb, geo, zs_axis)
        cat_img = make_glow_overlay(rgb, geo, cat_axis)

        fig, axes = plt.subplots(1, 2, figsize=(2 * W / DPI, H / DPI), dpi=DPI)
        fig.patch.set_facecolor("#080810")
        axes[0].imshow(zs_img,  interpolation="nearest")
        axes[0].set_title("Zhang-Suen Skeleton",
                           color="white", fontsize=7, pad=2)
        axes[0].axis("off")
        axes[1].imshow(cat_img, interpolation="nearest")
        axes[1].set_title("Chordal Axis Transform (CAT) — smooth + scale-adaptive",
                           color="yellow", fontsize=7, pad=2)
        axes[1].axis("off")
        plt.tight_layout(pad=0.15)
        fig.savefig(web_dir / f"{vi}_compare.png", dpi=DPI,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"done  (axis_px={cat_axis.sum()})")


def main():
    for run_id, h5 in DATASETS.items():
        if not h5.exists():
            print(f"[WARN] {h5} not found")
            continue
        process(run_id, h5)
    print("\nAll done.")


if __name__ == "__main__":
    main()
