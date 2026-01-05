import numpy as np

def stack_rgb_img(DN_key, band_data):
    R = np.squeeze(band_data['red'][DN_key])
    G = np.squeeze(band_data['green'][DN_key])
    B = np.squeeze(band_data['blue'][DN_key])

    # Stack into (H, W, 3)
    DN_rgb_image = np.stack([R, G, B], axis=-1)

    return DN_rgb_image

def crop_black_border_image(img_array: np.ndarray, threshold: int = 10) -> np.ndarray:
    gray = np.mean(img_array, axis=2)
    mask = gray > threshold

    # Find rows and columns where content exists
    valid_rows = np.where(np.any(mask, axis=1))[0]
    valid_cols = np.where(np.any(mask, axis=0))[0]

    if valid_rows.size == 0 or valid_cols.size == 0:
        return img_array  # image is all black

    y0, y1 = valid_rows[0], valid_rows[-1] + 1
    x0, x1 = valid_cols[0], valid_cols[-1] + 1

    return img_array[y0:y1, x0:x1]

def DN255_to_linear(img_DN):
    img = img_DN / 255.0
    img_linear = np.power(img, 2.2)
    return img_linear

def linear_to_DN255(img_linear):
    img = np.power(img_linear, 1/2.2)
    img_DN = img * 255
    img_DN = np.array(img_DN)
    img_DN[img_DN>=255] = 255
    return img_DN.astype(int)

def _read_spd_two_col(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read 2-col SPD (wavelength,value); returns (wvl_nm, values)."""
    arr = np.genfromtxt(path, comments="#", dtype=float, invalid_raise=False)
    arr = np.atleast_2d(arr)

    if arr.shape[1] < 2:
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                try:
                    w = float(parts[0]); v = float(parts[1])
                except Exception:
                    continue
                if np.isfinite(w) and np.isfinite(v):
                    rows.append((w, v))
        if not rows:
            raise ValueError(f"SPD file has no valid numeric rows: {path}")
        arr = np.array(rows, dtype=float)

    wvl = arr[:, 0].astype(float)
    val = arr[:, 1].astype(float)
    m = np.isfinite(wvl) & np.isfinite(val)
    wvl, val = wvl[m], val[m]

    if wvl.size < 2:
        raise ValueError(f"SPD file has too few valid samples: {path}")

    # auto-convert µm -> nm if needed
    if float(np.max(wvl)) < 20.0:
        wvl = wvl * 1000.0

    order = np.argsort(wvl)
    return wvl[order], val[order]


def _resample_to(wvl_src: np.ndarray, val_src: np.ndarray, wvl_dst: np.ndarray) -> np.ndarray:
    """Resample values onto wavelength grid; returns array."""
    return np.interp(wvl_dst, wvl_src, val_src, left=0.0, right=0.0)


def band_weighted_irradiance_integral(spd_path: str, band_rsp_path: str) -> float:
    """Compute ∫ E(λ)R(λ)dλ over overlap; returns float."""
    w_spd, E = _read_spd_two_col(spd_path)
    w_rsp, R = _read_spd_two_col(band_rsp_path)

    w_min = max(float(w_spd.min()), float(w_rsp.min()))
    w_max = min(float(w_spd.max()), float(w_rsp.max()))
    if w_max <= w_min:
        return 0.0

    m = (w_rsp >= w_min) & (w_rsp <= w_max)
    w = w_rsp[m]
    if w.size < 2:
        w = w_spd[(w_spd >= w_min) & (w_spd <= w_max)]
        if w.size < 2:
            return 0.0
        Rw = _resample_to(w_rsp, R, w)
        Ew = _resample_to(w_spd, E, w)
    else:
        Rw = R[m]
        Ew = _resample_to(w_spd, E, w)

    return float(np.trapz(Ew * Rw, w))


def radiance_to_toa_reflectance(L, E_band, cos_theta_s, d_au=1.0):
    """Convert band-weighted radiance to TOA reflectance; returns array."""
    return (np.pi * L * (d_au ** 2)) / (E_band * cos_theta_s + 1e-12)

