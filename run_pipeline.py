#!/usr/bin/env python
"""
run_pipeline.py — Unified hyperspectral EDA pipeline

Usage
-----
    python run_pipeline.py Planet_Data/20250407_035451_00_4001_ortho_sr_hdf5.h5
    python run_pipeline.py Planet_Data/xxx.h5 --docs-root docs --outputs-root outputs

After running, update the web pages:
    python docs/generate_pages.py

Then commit and push:
    git add docs/ outputs/
    git commit -m "feat: add dataset <RUN_ID>"
    git push
"""
import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description="Planet Tanager hyperspectral EDA pipeline")
    ap.add_argument("h5_file", type=pathlib.Path, help="Path to .h5 HDF5 file")
    ap.add_argument("--docs-root", type=pathlib.Path, default=ROOT / "docs",
                    help="Path to docs/ directory (default: ./docs)")
    ap.add_argument("--outputs-root", type=pathlib.Path, default=ROOT / "outputs",
                    help="Path to outputs/ directory (default: ./outputs)")
    ap.add_argument("--skip-rasters", action="store_true",
                    help="Skip GeoTIFF export (faster, saves disk)")
    args = ap.parse_args()

    data_path = args.h5_file.resolve()
    if not data_path.exists():
        print(f"ERROR: File not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # RUN_ID: strip sensor suffix from filename
    stem = data_path.stem  # e.g. 20250407_035451_00_4001_ortho_sr_hdf5
    run_id = stem.replace("_ortho_sr_hdf5", "").replace("_ortho_sr", "")
    print(f"\n{'='*60}")
    print(f"  RUN_ID : {run_id}")
    print(f"  Input  : {data_path}")
    print(f"  Outputs: {args.outputs_root / run_id}")
    print(f"  Docs   : {args.docs_root}")
    print(f"{'='*60}\n")

    out_dir = args.outputs_root / run_id
    for sub in ["png", "tables", "rasters"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────────
    from pipeline.config import load_indices_config
    indices_cfg = load_indices_config()

    # ── Step 1: Ingest ─────────────────────────────────────────────────────────
    print("[1/6] Ingest HDF5")
    from pipeline.ingest import load_cube, walk_hdf5, select_schema
    import pandas as pd

    cube, wavelengths, mask, root_attrs = load_cube(data_path)
    print(f"  cube={cube.shape}, wavelengths={wavelengths.size}, "
          f"range={wavelengths.min():.0f}–{wavelengths.max():.0f} nm")

    # Save schema inventory
    entries, _ = walk_hdf5(data_path)
    pd.DataFrame(entries).to_csv(out_dir / "tables" / "schema_inventory.csv", index=False)
    with open(out_dir / "schema_root_attrs.json", "w") as f:
        json.dump(root_attrs, f, indent=2)

    # ── Step 2: QA/QC ──────────────────────────────────────────────────────────
    print("[2/6] QA/QC")
    from pipeline.qaqc import run_qaqc
    qaqc = run_qaqc(cube, wavelengths, out_dir, run_id)

    # ── Step 3: Spectral Indices ───────────────────────────────────────────────
    print("[3/6] Spectral indices")
    from pipeline.indices import compute_all_indices, run_index_summary
    index_maps, idx_meta = compute_all_indices(cube, wavelengths, qaqc["mean_spec_sg"], indices_cfg)
    run_index_summary(
        index_maps, wavelengths,
        qaqc["mean_spec"], qaqc["mean_spec_sg"],
        idx_meta["rep_nm"], out_dir, indices_cfg,
    )

    # ── Step 4: PCA + Anomaly ──────────────────────────────────────────────────
    print("[4/6] PCA + Anomaly")
    from pipeline.pca import run_pca
    pca_result = run_pca(cube, out_dir=out_dir)

    # ── Step 5: Raster Export ──────────────────────────────────────────────────
    if not args.skip_rasters:
        print("[5/6] GeoTIFF export")
        from pipeline.export_raster import export_all_geotiffs
        export_all_geotiffs(index_maps, pca_result, out_dir, root_attrs)
    else:
        print("[5/6] GeoTIFF export — SKIPPED")

    # ── Step 6: Web Export ─────────────────────────────────────────────────────
    print("[6/6] Web export")
    from pipeline.export_web import (
        export_spectra_json, export_index_pngs,
        copy_overview_pngs, update_datasets_registry,
    )
    export_spectra_json(run_id, wavelengths, qaqc["mean_spec"], qaqc["mean_spec_sg"], args.docs_root)
    export_index_pngs(run_id, index_maps, indices_cfg, args.docs_root)
    copy_overview_pngs(run_id, out_dir, args.docs_root)
    update_datasets_registry(
        run_id=run_id,
        cube_shape=cube.shape,
        rep_nm=idx_meta["rep_nm"],
        pca_variance_ratio=pca_result["variance_ratio"],
        anomaly_threshold=pca_result["threshold"],
        docs_root=args.docs_root,
    )

    # ── Summary markdown ───────────────────────────────────────────────────────
    var = pca_result["variance_ratio"]
    summary = "\n".join([
        f"# EDA Summary — {run_id}",
        "",
        f"- Data: {data_path}",
        f"- Cube shape [bands, rows, cols]: {cube.shape}",
        f"- REP mean-spectrum (nm): {idx_meta['rep_nm']:.2f}",
        f"- PCA explained variance: {[round(v, 4) for v in var]}",
        f"- Anomaly threshold (P99): {pca_result['threshold']:.4f}",
        "",
        "## Outputs",
        f"- PNG: {out_dir / 'png'}",
        f"- Tables: {out_dir / 'tables'}",
        f"- Rasters: {out_dir / 'rasters'}",
    ])
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  Done! Run ID: {run_id}")
    print(f"  Next steps:")
    print(f"    python docs/generate_pages.py")
    print(f"    git add docs/ outputs/{run_id}/")
    print(f"    git commit -m 'feat: add dataset {run_id}'")
    print(f"    git push")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
