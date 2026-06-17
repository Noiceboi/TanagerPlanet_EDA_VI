"""
Export individual clean images for the Water Analysis GitHub Pages section.

Per dataset × VI generates 4 images in docs/images/water_analysis/{run_id}/:
  {VI}_mask.png        — pure B&W binary mask (threshold P88)
  {VI}_threshold.png   — RGB + cyan overlay  (threshold)
  {VI}_geometric.png   — RGB + cyan overlay  (Frangi + LoG geometric)
  {VI}_skeleton.png    — dark RGB + fill + glowing skeleton (ZS centerlines)

Plus two overview images copied from existing outputs:
  overview_geo.png     <- summary_geo.png
  overview_skeleton.png <- skeleton_summary.png
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.filters import frangi
from skimage.morphology import (
    binary_closing, binary_dilation, disk,
    remove_small_objects, skeletonize,
)

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

FRANGI_SIGMAS = [2, 3, 4, 6, 8, 11, 16]
BLOB_SIGMAS   = [4, 6, 8, 11, 15, 20]
RGB_NM        = (665, 560, 470)
DPI           = 130


# ── Helpers ────────────────────────────────────────────────────────────────────

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


def threshold_mask(vi_map, water_high, pct=88):
    """Simple percentile threshold → binary mask."""
    valid = vi_map[np.isfinite(vi_map)]
    thr = np.percentile(valid, pct if water_high else (100 - pct))
    mask = (vi_map > thr) if water_high else (vi_map < thr)
    return (mask & np.isfinite(vi_map)).astype(bool)


def geometric_mask(vi_map, water_high, prior_pct=30):
    """Frangi + LoG blob → cleaned binary mask."""
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
        r = -gaussian_laplace(img.astype(np.float64), sigma=s) * s**2
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


def make_overlay(rgb, mask, color=(0.0, 0.85, 1.0), alpha=0.60):
    """Clean RGB + solid-alpha color overlay + white border."""
    out = rgb.copy()
    m = mask.astype(bool)
    c = np.array(color)
    out[m] = (1 - alpha) * out[m] + alpha * c
    border = binary_dilation(m, disk(1)) & ~m
    out[border] = [1.0, 1.0, 1.0]
    return np.clip(out, 0, 1)


def make_skeleton_overlay(rgb, geo, skeleton):
    """Dark RGB + cyan fill + yellow-white glowing skeleton lines."""
    out = rgb * 0.60
    fill_c = np.array([0.0, 0.80, 1.0])
    m = geo.astype(bool)
    out[m] = 0.70 * out[m] + 0.30 * fill_c

    # outer glow (cyan)
    gf = gaussian_filter(skeleton.astype(np.float32), sigma=2.5)
    gmax = gf.max()
    if gmax > 0:
        gf /= gmax
    for c, v in enumerate([0.0, 0.90, 1.0]):
        out[..., c] = out[..., c] * (1 - 0.55 * gf) + v * 0.55 * gf

    # hard core (near-white yellow)
    core = binary_dilation(skeleton, disk(1))
    core_c = np.array([1.0, 1.0, 0.80])
    out[core] = core_c
    return np.clip(out, 0, 1)


def save_img(arr, path, dpi=DPI):
    """Save a 2-D (grayscale) or 3-D (H,W,3) array as a clean image file."""
    h, w = arr.shape[:2]
    fig_w = w / dpi
    fig_h = h / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if arr.ndim == 2:
        ax.imshow(arr, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    else:
        ax.imshow(np.clip(arr, 0, 1), interpolation="nearest")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0,
                facecolor="black")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def process(run_id, h5_path):
    web_dir = ROOT / "docs" / "images" / "water_analysis" / run_id
    web_dir.mkdir(parents=True, exist_ok=True)
    raster_dir = ROOT / "outputs" / run_id / "rasters"
    geo_src_dir = ROOT / "outputs" / run_id / "water_geometric"

    print(f"\n{'='*56}")
    print(f"  {run_id}")
    print(f"{'='*56}")

    # RGB
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

        print(f"  {vi}: mask...", end=" ", flush=True)
        t_mask = threshold_mask(vi_map, cfg["water_high"])
        save_img(t_mask.astype(np.float32), web_dir / f"{vi}_mask.png")

        print("threshold...", end=" ", flush=True)
        save_img(make_overlay(rgb, t_mask), web_dir / f"{vi}_threshold.png")

        print("geometric...", end=" ", flush=True)
        geo = geometric_mask(vi_map, cfg["water_high"], cfg["prior_pct"])
        save_img(make_overlay(rgb, geo), web_dir / f"{vi}_geometric.png")

        print("skeleton...", end=" ", flush=True)
        skel = skeletonize(geo)
        save_img(make_skeleton_overlay(rgb, geo, skel), web_dir / f"{vi}_skeleton.png")

        wp = 100 * geo.sum() / geo.size
        sp = geo.sum()
        print(f"done  (water={wp:.1f}%  skel_px={skel.sum()})")

    # Copy overview multi-panel images
    for src_name, dst_name in [
        ("summary_geo.png",      "overview_geo.png"),
        ("skeleton_summary.png", "overview_skeleton.png"),
    ]:
        src = geo_src_dir / src_name
        dst = web_dir / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {dst_name}")

    print(f"  -> {web_dir.relative_to(ROOT)}")


def main():
    for run_id, h5 in DATASETS.items():
        if not h5.exists():
            print(f"[WARN] {h5} not found")
            continue
        process(run_id, h5)
    print("\nAll done.")


if __name__ == "__main__":
    main()
