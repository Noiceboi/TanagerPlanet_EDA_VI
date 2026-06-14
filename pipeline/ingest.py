"""HDF5 ingestion — extracted verbatim from planet_eda notebooks."""
from pathlib import Path

import h5py
import numpy as np


def walk_hdf5(path: Path):
    """Recursively catalog all HDF5 datasets and groups."""
    entries = []
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                entries.append({
                    "path": name,
                    "type": "dataset",
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": {k: str(v) for k, v in obj.attrs.items()},
                })
            elif isinstance(obj, h5py.Group):
                entries.append({
                    "path": name,
                    "type": "group",
                    "shape": "",
                    "dtype": "",
                    "attrs": {k: str(v) for k, v in obj.attrs.items()},
                })
        f.visititems(visitor)
        root_attrs = {k: str(v) for k, v in f.attrs.items()}
    return entries, root_attrs


def find_first_dataset(entries, include_keywords, min_ndim=2):
    for e in entries:
        if e["type"] != "dataset":
            continue
        p = e["path"].lower()
        if all(k in p for k in include_keywords):
            shape = e["shape"]
            if isinstance(shape, tuple) and len(shape) >= min_ndim:
                return e["path"]
    return None


def select_schema(entries):
    candidates = [
        ["reflectance"],
        ["surface", "reflectance"],
        ["sr"],
        ["cube"],
        ["data"],
        ["radiance"],
    ]
    refl = None
    for keys in candidates:
        refl = find_first_dataset(entries, keys, min_ndim=3)
        if refl:
            break

    wave = None
    for keys in [["wavelength"], ["bands"], ["center", "wavelength"]]:
        wave = find_first_dataset(entries, keys, min_ndim=1)
        if wave:
            break

    mask = None
    for keys in [["mask"], ["quality"], ["qa"], ["nodata"]]:
        mask = find_first_dataset(entries, keys, min_ndim=2)
        if mask:
            break

    return {"reflectance": refl, "wavelength": wave, "mask": mask}


def read_dataset(file_path: Path, ds_path: str):
    with h5py.File(file_path, "r") as f:
        arr = np.array(f[ds_path][...])
        attrs = {k: f[ds_path].attrs[k] for k in f[ds_path].attrs.keys()}
    return arr, attrs


def parse_attr_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.float32).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=np.float32).reshape(-1)
    text = str(value).replace("\n", " ").replace("[", " ").replace("]", " ").replace(",", " ")
    arr = np.fromstring(text, sep=" ")
    return arr.astype(np.float32)


def load_cube(data_path: Path):
    """
    Load and normalize hyperspectral cube from HDF5.

    Returns
    -------
    cube : np.ndarray [bands, rows, cols] float32
    wavelengths : np.ndarray [bands,] float32  (nm)
    mask : np.ndarray [rows, cols] bool | None
    root_attrs : dict
    """
    data_path = Path(data_path)
    entries, root_attrs = walk_hdf5(data_path)
    schema = select_schema(entries)

    if schema["reflectance"] is None:
        raise RuntimeError(
            "Cannot find 3D reflectance dataset. "
            "Check schema_inventory.csv for the correct path."
        )

    cube, cube_attrs = read_dataset(data_path, schema["reflectance"])

    if cube.ndim != 3:
        raise RuntimeError(f"Reflectance dataset is not 3D: {cube.shape}")

    # Normalize to [bands, rows, cols]
    if cube.shape[0] < 20 and cube.shape[-1] > 20:
        cube = np.moveaxis(cube, -1, 0)
    elif cube.shape[-1] < 20 and cube.shape[0] > 20:
        cube = np.moveaxis(cube, 0, -1)
        cube = np.moveaxis(cube, -1, 0)

    if cube.shape[0] < 20:
        raise RuntimeError(f"Cannot infer valid band dimension: {cube.shape}")

    cube = cube.astype(np.float32, copy=False)

    scale = float(cube_attrs["scale_factor"]) if "scale_factor" in cube_attrs else 1.0
    offset = float(cube_attrs["add_offset"]) if "add_offset" in cube_attrs else 0.0
    cube = cube * scale + offset

    nodata = None
    for k in ["_FillValue", "nodata", "nodata_value", "missing_value"]:
        if k in cube_attrs:
            try:
                nodata = float(cube_attrs[k])
                break
            except Exception:
                pass
    if nodata is not None:
        cube[cube == nodata] = np.nan

    mask = None
    if schema["mask"] is not None:
        mask_arr, _ = read_dataset(data_path, schema["mask"])
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[0]
        mask = mask_arr.astype(bool)

    # Wavelength extraction: dataset → attrs → fallback linspace
    wavelengths = None
    if schema["wavelength"] is not None:
        wl_arr, _ = read_dataset(data_path, schema["wavelength"])
        wl_arr = np.array(wl_arr).reshape(-1).astype(np.float32)
        if wl_arr.size > 10:
            wavelengths = wl_arr

    if wavelengths is None:
        for key in ["wavelengths", "wavelength", "bands", "band_centers", "center_wavelength"]:
            if key in cube_attrs:
                wl_arr = parse_attr_array(cube_attrs[key])
                if wl_arr.size > 10:
                    wavelengths = wl_arr
                    break

    if wavelengths is None:
        wavelengths = np.linspace(380.0, 2500.0, cube.shape[0], dtype=np.float32)

    # Unit normalization: convert μm → nm
    wl_unit = str(cube_attrs.get("wavelengths_units", cube_attrs.get("wavelength_unit", ""))).lower()
    if "um" in wl_unit or "mic" in wl_unit or np.nanmax(wavelengths) < 10:
        wavelengths = wavelengths * 1000.0

    # Size alignment
    if wavelengths.size != cube.shape[0]:
        x_old = np.linspace(0, 1, wavelengths.size, dtype=np.float32)
        x_new = np.linspace(0, 1, cube.shape[0], dtype=np.float32)
        wavelengths = np.interp(x_new, x_old, wavelengths).astype(np.float32)

    return cube, wavelengths, mask, root_attrs
