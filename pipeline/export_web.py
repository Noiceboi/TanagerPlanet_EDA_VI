"""Web asset generation: JSON spectra, index PNGs, datasets registry."""
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import load_datasets_registry, save_datasets_registry

_CARD_COLORS = [
    "bg-primary text-white",
    "bg-success text-white",
    "bg-info text-dark",
    "bg-warning text-dark",
]


def export_spectra_json(
    run_id: str,
    wavelengths: np.ndarray,
    mean_spec: np.ndarray,
    mean_spec_sg: np.ndarray,
    docs_root: Path,
):
    """Write mean_spectra_{run_id}.json to docs/data/."""
    docs_root = Path(docs_root)
    data_dir = docs_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": run_id,
        "wavelengths": wavelengths.tolist(),
        "mean_reflectance": mean_spec_sg.tolist(),
        "mean_reflectance_raw": mean_spec.tolist(),
    }
    out = data_dir / f"mean_spectra_{run_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Spectra JSON: {out.name}")


def export_index_pngs(
    run_id: str,
    index_maps: dict,
    indices_cfg: dict,
    docs_root: Path,
):
    """Export per-index PNG maps to docs/images/maps/{run_id}/."""
    docs_root = Path(docs_root)
    web_maps_dir = docs_root / "images" / "maps" / run_id
    web_maps_dir.mkdir(parents=True, exist_ok=True)

    idx_cfg = indices_cfg.get("indices", {})
    idx_cmaps = {k: v.get("colormap", "viridis") for k, v in idx_cfg.items()}

    saved = []
    for key, arr2d in index_maps.items():
        if arr2d is None:
            continue
        valid_px = arr2d[np.isfinite(arr2d)]
        if valid_px.size == 0:
            continue

        cmap = idx_cmaps.get(key, "viridis")
        vmin, vmax = np.percentile(valid_px, [2, 98])

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im = ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{key}  —  {run_id}", fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        out_path = web_maps_dir / f"{key}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        saved.append(key)

    print(f"  Index PNGs ({len(saved)}): {web_maps_dir}")


def copy_overview_pngs(run_id: str, out_dir: Path, docs_root: Path):
    """Copy the 4 overview PNGs (01-04) from outputs/png/ to docs/images/maps/{run_id}/."""
    out_dir = Path(out_dir)
    docs_root = Path(docs_root)
    src_dir = out_dir / "png"
    dst_dir = docs_root / "images" / "maps" / run_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in ["01_qaqc_rgb_and_spectrum.png", "02_index_maps.png",
                 "03_spectrum_rep.png", "04_pca_anomaly.png"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    print(f"  Overview PNGs copied to {dst_dir}")


def update_datasets_registry(
    run_id: str,
    cube_shape: tuple,
    rep_nm: float,
    pca_variance_ratio: list,
    anomaly_threshold: float,
    docs_root: Path,
):
    """Upsert run_id entry in docs/data/datasets.json."""
    docs_root = Path(docs_root)
    registry = load_datasets_registry(docs_root)
    datasets = registry.setdefault("datasets", {})

    # Determine card color by position
    existing_keys = list(datasets.keys())
    if run_id not in existing_keys:
        color_idx = len(existing_keys)
    else:
        color_idx = list(datasets.keys()).index(run_id)
    card_class = _CARD_COLORS[color_idx % len(_CARD_COLORS)]

    # Parse date from run_id (YYYYMMDD_...)
    date_str = ""
    label = run_id
    if len(run_id) >= 8 and run_id[:8].isdigit():
        y, m, d = run_id[:4], run_id[4:6], run_id[6:8]
        date_str = f"{y}-{m}-{d}"
        from datetime import datetime
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            label = dt.strftime("%B %d %Y")
        except ValueError:
            label = date_str

    n = len(datasets) + (0 if run_id in datasets else 1)
    card_label = f"Dataset {color_idx + 1}"

    var = pca_variance_ratio
    pca_str = " / ".join(f"{v * 100:.1f}%" for v in var[:3])

    datasets[run_id] = {
        "label": label,
        "jsonFile": f"mean_spectra_{run_id}.json",
        "cubeShape": f"({cube_shape[0]}, {cube_shape[1]}, {cube_shape[2]})",
        "repMean": f"{rep_nm:.2f} nm",
        "pcaVar": pca_str,
        "anomalyThr": f"{anomaly_threshold:.2f}",
        "date": date_str,
        "cardHeaderClass": card_class,
        "cardLabel": card_label,
    }

    registry["_meta"] = {"generated_by": "run_pipeline.py", "schema_version": 1}
    save_datasets_registry(registry, docs_root)
    print(f"  datasets.json updated ({len(datasets)} datasets)")
