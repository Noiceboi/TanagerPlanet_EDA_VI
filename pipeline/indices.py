"""Spectral index computation — driven by pipeline/indices.yaml."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from .utils import safe_div, safe_log_inv, nearest_band_idx


# ──────────────────────────────────────────────────────────────────────────────
# Private compute functions
# Each receives:
#   R   — dict {band_key: 2D array}  built from YAML bands_nm
#   wavelengths — 1D array (unused by most, needed by REP)
#   mean_spec_sg — 1D SG-smoothed mean spectrum (used by REP scalar)
# Returns: 2D ndarray
# ──────────────────────────────────────────────────────────────────────────────

def _compute_ndvi(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_NIR"] - R["R_Red"], R["R_NIR"] + R["R_Red"])


def _compute_psri(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_680"] - R["R_490"], R["R_800"])


def _compute_rep_pixel(R, wavelengths, mean_spec_sg):
    return 700.0 + 40.0 * safe_div(
        ((R["R_670"] + R["R_780"]) / 2.0) - R["R_700"],
        R["R_740"] - R["R_700"],
    )


def _compute_pri(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_531"] - R["R_570"], R["R_531"] + R["R_570"])


def _compute_ari(R, wavelengths, mean_spec_sg):
    return safe_div(1.0, R["R_550"]) - safe_div(1.0, R["R_700"])


def _compute_cri550(R, wavelengths, mean_spec_sg):
    return safe_div(1.0, R["R_510"]) - safe_div(1.0, R["R_550"])


def _compute_cri700(R, wavelengths, mean_spec_sg):
    return safe_div(1.0, R["R_510"]) - safe_div(1.0, R["R_700"])


def _compute_mcari(R, wavelengths, mean_spec_sg):
    return (
        (R["R_700"] - R["R_670"]) - 0.2 * (R["R_700"] - R["R_550"])
    ) * safe_div(R["R_700"], R["R_670"])


def _compute_mcari_osavi(R, wavelengths, mean_spec_sg):
    mcari = (
        (R["R_700"] - R["R_670"]) - 0.2 * (R["R_700"] - R["R_550"])
    ) * safe_div(R["R_700"], R["R_670"])
    osavi = 1.16 * safe_div(R["R_800"] - R["R_670"], R["R_800"] + R["R_670"] + 0.16)
    return safe_div(mcari, osavi)


def _compute_wbi(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_900"], R["R_970"])


def _compute_wi(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_900"], R["R_970"])


def _compute_msi(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_1599"], R["R_819"])


def _compute_ndli(R, wavelengths, mean_spec_sg):
    return safe_div(
        safe_log_inv(R["R_1754"]) - safe_log_inv(R["R_1680"]),
        safe_log_inv(R["R_1754"]) + safe_log_inv(R["R_1680"]),
    )


def _compute_ndni(R, wavelengths, mean_spec_sg):
    return safe_div(
        safe_log_inv(R["R_1510"]) - safe_log_inv(R["R_1680"]),
        safe_log_inv(R["R_1510"]) + safe_log_inv(R["R_1680"]),
    )


def _compute_protein_proxy(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_2180"], R["R_2100"])


def _compute_cai(R, wavelengths, mean_spec_sg):
    return 0.5 * (R["R_2000"] + R["R_2200"]) - R["R_2100"]


def _compute_lcai(R, wavelengths, mean_spec_sg):
    part1 = safe_div(
        0.5 * (R["R_2000"] + R["R_2200"]) - R["R_2100"],
        0.5 * (R["R_2000"] + R["R_2200"]),
    )
    part2 = safe_div(
        0.5 * (R["R_2200"] + R["R_2400"]) - R["R_2300"],
        0.5 * (R["R_2200"] + R["R_2400"]),
    )
    return part1 + part2


def _compute_hdwi(R, wavelengths, mean_spec_sg):
    return safe_div(R["R_572"] - R["R_420"], R["R_572"] + R["R_420"])


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch registry — key must match indices.yaml `compute:` field
# ──────────────────────────────────────────────────────────────────────────────
_COMPUTE_REGISTRY = {
    "ndvi":          _compute_ndvi,
    "psri":          _compute_psri,
    "rep_pixel":     _compute_rep_pixel,
    "pri":           _compute_pri,
    "ari":           _compute_ari,
    "cri550":        _compute_cri550,
    "cri700":        _compute_cri700,
    "mcari":         _compute_mcari,
    "mcari_osavi":   _compute_mcari_osavi,
    "wbi":           _compute_wbi,
    "wi":            _compute_wi,
    "msi":           _compute_msi,
    "hdwi":          _compute_hdwi,
    "ndli":          _compute_ndli,
    "ndni":          _compute_ndni,
    "protein_proxy": _compute_protein_proxy,
    "cai":           _compute_cai,
    "lcai":          _compute_lcai,
}


def compute_all_indices(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    mean_spec_sg: np.ndarray,
    indices_cfg: dict,
):
    """
    Compute all indices defined in indices.yaml.

    Parameters
    ----------
    cube : [B, H, W] float32
    wavelengths : [B,] float32 (nm)
    mean_spec_sg : [B,] float32 — Savitzky-Golay smoothed mean spectrum
    indices_cfg : loaded indices.yaml dict

    Returns
    -------
    index_maps : dict {index_name: 2D ndarray}
    metadata : dict — {"rep_nm": float, ...}
    """
    index_maps = {}
    metadata = {}

    for index_name, cfg in indices_cfg["indices"].items():
        compute_key = cfg.get("compute")
        fn = _COMPUTE_REGISTRY.get(compute_key)
        if fn is None:
            print(f"  [WARN] No compute function for '{compute_key}' (index={index_name}), skipping")
            continue

        # Build band slice dict from YAML bands_nm
        bands_nm = cfg.get("bands_nm", {})
        R = {}
        for var_name, target_nm in bands_nm.items():
            idx = nearest_band_idx(wavelengths, target_nm)
            R[var_name] = cube[idx]

        try:
            result = fn(R, wavelengths, mean_spec_sg)
            index_maps[index_name] = result.astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Failed computing {index_name}: {e}")
            continue

    # REP scalar from mean spectrum (derivative method)
    wl_window = min(11, (len(mean_spec_sg) // 2) * 2 - 1)
    wl_window = max(wl_window, 5)
    deriv = np.gradient(mean_spec_sg, wavelengths)
    red_edge_mask = (wavelengths >= 680) & (wavelengths <= 780)
    if np.any(red_edge_mask):
        rep_nm = float(wavelengths[red_edge_mask][np.argmax(deriv[red_edge_mask])])
    else:
        rep_nm = float(wavelengths[np.argmax(deriv)])
    metadata["rep_nm"] = rep_nm

    print(f"  Indices computed: {list(index_maps.keys())}")
    print(f"  REP (mean spectrum): {rep_nm:.2f} nm")

    return index_maps, metadata


def run_index_summary(
    index_maps: dict,
    wavelengths: np.ndarray,
    mean_spec: np.ndarray,
    mean_spec_sg: np.ndarray,
    rep_nm: float,
    out_dir: Path,
    indices_cfg: dict,
):
    """
    Save index statistics tables and overview PNGs.

    Saves
    -----
    out_dir/tables/index_summary.csv + .parquet
    out_dir/png/02_index_maps.png
    out_dir/png/03_spectrum_rep.png
    """
    out_dir = Path(out_dir)
    tab_dir = out_dir / "tables"
    png_dir = out_dir / "png"
    tab_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    # Statistics table
    rows = []
    for name, arr in index_maps.items():
        rows.append({
            "index": name,
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "p05": float(np.nanpercentile(arr, 5)),
            "p50": float(np.nanpercentile(arr, 50)),
            "p95": float(np.nanpercentile(arr, 95)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / "index_summary.csv", index=False)
    df.to_parquet(tab_dir / "index_summary.parquet", index=False)

    # Read colormaps from config
    idx_cfg = indices_cfg.get("indices", {})
    idx_cmaps = {k: v.get("colormap", "viridis") for k, v in idx_cfg.items()}

    # 02 — Overview grid
    n = len(index_maps)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (name, arr) in zip(axes, index_maps.items()):
        cmap = idx_cmaps.get(name, "viridis")
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_dir / "02_index_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 03 — Spectrum + REP
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wavelengths, mean_spec, alpha=0.5, linewidth=1, label="Mean (raw)")
    ax.plot(wavelengths, mean_spec_sg, linewidth=1.8, label="Savitzky-Golay")
    ax.axvline(rep_nm, color="r", linestyle="--", linewidth=1.5, label=f"REP = {rep_nm:.1f} nm")
    ax.set_title("Spectral Profile and Red Edge Position")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_dir / "03_spectrum_rep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Index summary saved ({len(rows)} indices)")
