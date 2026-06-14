import numpy as np


def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def safe_log_inv(x, floor=1e-6):
    return np.log(1.0 / np.clip(x, floor, None))


def nearest_band_idx(wavelengths, target_nm):
    return int(np.argmin(np.abs(wavelengths - float(target_nm))))
