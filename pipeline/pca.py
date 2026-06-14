"""PCA dimensionality reduction and anomaly detection."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


def run_pca(
    cube: np.ndarray,
    n_components: int = 3,
    anomaly_percentile: float = 99.0,
    out_dir: Path = None,
) -> dict:
    """
    Fit PCA on valid pixel spectra, compute anomaly score map.

    Saves (if out_dir provided)
    ---------------------------
    out_dir/png/04_pca_anomaly.png
    out_dir/tables/anomaly_scores_flat.csv

    Returns
    -------
    dict with keys: pc_maps, score_map, anomaly, threshold, variance_ratio
    """
    B, H, W = cube.shape
    X = np.moveaxis(cube, 0, -1).reshape(-1, B)
    valid = np.all(np.isfinite(X), axis=1)
    Xv = X[valid]

    wl_window = min(11, (B // 2) * 2 - 1)
    wl_window = max(wl_window, 5)
    Xv_sg = savgol_filter(Xv, window_length=wl_window, polyorder=2, axis=1, mode="interp")

    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(Xv_sg)

    pc_maps = np.full((X.shape[0], n_components), np.nan, dtype=np.float32)
    pc_maps[valid] = pcs.astype(np.float32)
    pc_maps = pc_maps.reshape(H, W, n_components)

    z = (pcs - pcs.mean(axis=0)) / (pcs.std(axis=0) + 1e-8)
    score = np.sqrt((z ** 2).sum(axis=1))

    score_map = np.full(X.shape[0], np.nan, dtype=np.float32)
    score_map[valid] = score.astype(np.float32)
    score_map = score_map.reshape(H, W)

    thr = float(np.nanpercentile(score_map, anomaly_percentile))
    anomaly = score_map >= thr

    var_ratio = pca.explained_variance_ratio_.tolist()
    print(f"  PCA explained variance: {[f'{v:.3f}' for v in var_ratio]}")
    print(f"  Anomaly threshold (P{anomaly_percentile:.0f}): {thr:.4f}")

    if out_dir is not None:
        out_dir = Path(out_dir)
        png_dir = out_dir / "png"
        tab_dir = out_dir / "tables"
        png_dir.mkdir(parents=True, exist_ok=True)
        tab_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        cmap_names = ["coolwarm", "coolwarm", "coolwarm"]
        for i in range(n_components):
            im = axes.ravel()[i].imshow(pc_maps[:, :, i], cmap=cmap_names[i % 3])
            axes.ravel()[i].set_title(f"PC{i + 1} ({var_ratio[i] * 100:.1f}%)")
            axes.ravel()[i].axis("off")
            fig.colorbar(im, ax=axes.ravel()[i], fraction=0.046, pad=0.04)
        im = axes.ravel()[3].imshow(score_map, cmap="magma")
        axes.ravel()[3].contour(anomaly, levels=[0.5], colors="cyan", linewidths=0.5)
        axes.ravel()[3].set_title(f"Anomaly Score (≥P{anomaly_percentile:.0f} contour)")
        axes.ravel()[3].axis("off")
        fig.colorbar(im, ax=axes.ravel()[3], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_dir / "04_pca_anomaly.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame({"score": score}).to_csv(tab_dir / "anomaly_scores_flat.csv", index=False)

    return {
        "pc_maps": pc_maps,
        "score_map": score_map,
        "anomaly": anomaly,
        "threshold": thr,
        "variance_ratio": var_ratio,
    }
