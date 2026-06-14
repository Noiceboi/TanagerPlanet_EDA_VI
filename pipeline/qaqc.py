"""QA/QC statistics and quicklook generation."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def run_qaqc(cube: np.ndarray, wavelengths: np.ndarray, out_dir: Path, run_id: str) -> dict:
    """
    Compute band statistics, generate quicklook PNG, and compute mean spectrum.

    Saves
    -----
    out_dir/tables/qa_band_stats.csv + .parquet
    out_dir/png/01_qaqc_rgb_and_spectrum.png

    Returns
    -------
    dict with keys: mean_spec, mean_spec_sg
    """
    out_dir = Path(out_dir)
    tab_dir = out_dir / "tables"
    png_dir = out_dir / "png"
    tab_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    band_mean = np.nanmean(cube, axis=(1, 2))
    band_std = np.nanstd(cube, axis=(1, 2))
    valid_ratio = np.isfinite(cube).mean(axis=(1, 2))

    qa_df = pd.DataFrame({
        "band_index": np.arange(cube.shape[0]),
        "wavelength_nm": wavelengths,
        "mean": band_mean,
        "std": band_std,
        "valid_ratio": valid_ratio,
    })
    qa_df.to_csv(tab_dir / "qa_band_stats.csv", index=False)
    qa_df.to_parquet(tab_dir / "qa_band_stats.parquet", index=False)

    # RGB quicklook (nearest to 665, 560, 490 nm)
    r_idx = int(np.argmin(np.abs(wavelengths - 665)))
    g_idx = int(np.argmin(np.abs(wavelengths - 560)))
    b_idx = int(np.argmin(np.abs(wavelengths - 490)))
    rgb = np.stack([cube[r_idx], cube[g_idx], cube[b_idx]], axis=-1)
    p2 = np.nanpercentile(rgb, 2)
    p98 = np.nanpercentile(rgb, 98)
    rgb_vis = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    mean_spec = band_mean.copy()
    wl_window = min(11, (len(mean_spec) // 2) * 2 - 1)
    wl_window = max(wl_window, 5)
    mean_spec_sg = savgol_filter(mean_spec, window_length=wl_window, polyorder=2, mode="interp")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(rgb_vis)
    ax[0].set_title(f"RGB Quicklook — {run_id}")
    ax[0].axis("off")
    ax[1].plot(wavelengths, mean_spec, alpha=0.5, label="Mean")
    ax[1].fill_between(wavelengths, mean_spec - band_std, mean_spec + band_std, alpha=0.15)
    ax[1].plot(wavelengths, mean_spec_sg, linewidth=1.5, label="SG smooth")
    ax[1].set_title("Mean Spectrum ± 1σ")
    ax[1].set_xlabel("Wavelength (nm)")
    ax[1].set_ylabel("Reflectance")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_dir / "01_qaqc_rgb_and_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  QA/QC: shape={cube.shape}, finite_ratio={float(np.isfinite(cube).mean()):.3f}")

    return {"mean_spec": mean_spec, "mean_spec_sg": mean_spec_sg}
