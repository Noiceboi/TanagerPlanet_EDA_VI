"""GeoTIFF raster export."""
from pathlib import Path

import numpy as np
import pandas as pd


def export_all_geotiffs(
    index_maps: dict,
    pca_result: dict,
    out_dir: Path,
    root_attrs: dict,
):
    """
    Export all index maps, anomaly score, and PCA components as GeoTIFF.

    Saves
    -----
    out_dir/rasters/*.tif
    out_dir/tables/raster_export_log.csv
    """
    out_dir = Path(out_dir)
    raster_dir = out_dir / "rasters"
    tab_dir = out_dir / "tables"
    raster_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    log = []
    for name, arr in index_maps.items():
        log.append({"layer": name, "result": _try_export_one(name, arr, raster_dir, root_attrs)})

    log.append({"layer": "ANOMALY_SCORE", "result": _try_export_one(
        "ANOMALY_SCORE", pca_result["score_map"], raster_dir, root_attrs
    )})

    pc_maps = pca_result.get("pc_maps")
    if pc_maps is not None:
        for i in range(pc_maps.shape[2]):
            log.append({"layer": f"PC{i+1}", "result": _try_export_one(
                f"PC{i+1}", pc_maps[:, :, i], raster_dir, root_attrs
            )})

    pd.DataFrame(log).to_csv(tab_dir / "raster_export_log.csv", index=False)
    ok = sum(1 for r in log if not r["result"].startswith("skip"))
    print(f"  Rasters: {ok}/{len(log)} exported to {raster_dir}")


def _try_export_one(name: str, arr2d: np.ndarray, raster_dir: Path, root_attrs: dict) -> str:
    try:
        import rasterio
        from rasterio.transform import Affine

        profile = {
            "driver": "GTiff",
            "height": arr2d.shape[0],
            "width": arr2d.shape[1],
            "count": 1,
            "dtype": "float32",
            "compress": "deflate",
        }

        transform = None
        crs = None
        for k in ["transform", "geo_transform", "geotransform"]:
            if k in root_attrs:
                vals = str(root_attrs[k]).replace("[", "").replace("]", "").split(",")
                nums = [float(v) for v in vals if v.strip()]
                if len(nums) >= 6:
                    transform = Affine(nums[1], nums[2], nums[0], nums[4], nums[5], nums[3])
                    break
        for k in ["crs", "spatial_ref", "epsg"]:
            if k in root_attrs:
                crs = str(root_attrs[k])
                break

        if transform is not None:
            profile["transform"] = transform
        if crs is not None:
            profile["crs"] = crs

        out = raster_dir / f"{name}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr2d.astype(np.float32), 1)
        return str(out)
    except Exception as e:
        return f"skip: {e}"
