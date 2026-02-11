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

def spd_area_nm(spd_path: str) -> float:
    """spd_area_nm(spd_path) -> float: Integral ∫R(λ)dλ in nm for a band SPD (wl,value)."""
    arr = np.loadtxt(spd_path, dtype=np.float64)
    wl = arr[:, 0]
    y = arr[:, 1]
    ok = np.isfinite(wl) & np.isfinite(y)
    wl = wl[ok]
    y = y[ok]
    if wl.size < 2:
        return 0.0
    return float(np.trapezoid(y, wl))


def radiance_to_toa_reflectance(L, E_band, cos_theta_s, d_au=1.0):
    """Convert band-weighted radiance to TOA reflectance; returns array."""
    return (np.pi * L * (d_au ** 2)) / (E_band * cos_theta_s + 1e-12)

def radiance_rgb_to_toa_reflectance(
    radiance_rgb,
    band_data,
    sun_spd_path,
    cos_theta_s,
    d_au=1.0,
    eps=1e-12,
):
    """
    radiance_rgb_to_toa_reflectance(radiance_rgb, band_data, sun_spd_path, cos_theta_s, d_au=1.0)
    -> np.ndarray

    Convert band-integrated RGB radiance to TOA reflectance using band SPDs.

    Inputs
    - radiance_rgb : (H,W,3) array, band-integrated radiance [W m^-2 sr^-1]
    - band_data    : dict with keys 'red','green','blue', each containing 'spd'
    - sun_spd_path : path to solar spectral irradiance SPD [W m^-2 nm^-1]
    - cos_theta_s  : cos(solar zenith angle) at target
    - d_au         : Earth–Sun distance in AU (default 1.0)

    Output
    - rho_rgb : (H,W,3) TOA reflectance
    """

    # --- band SPDs ---
    band_R = band_data["red"]["spd"]
    band_G = band_data["green"]["spd"]
    band_B = band_data["blue"]["spd"]

    # --- in-band solar irradiance  (∫E(λ)R(λ)dλ) ---
    E_R = band_weighted_irradiance_integral(sun_spd_path, band_R)
    E_G = band_weighted_irradiance_integral(sun_spd_path, band_G)
    E_B = band_weighted_irradiance_integral(sun_spd_path, band_B)

    # --- band widths (∫R(λ)dλ) ---
    A_R = spd_area_nm(band_R)
    A_G = spd_area_nm(band_G)
    A_B = spd_area_nm(band_B)

    # --- band-averaged irradiance [W m^-2 nm^-1] ---
    Ebar_R = E_R / (A_R + eps)
    Ebar_G = E_G / (A_G + eps)
    Ebar_B = E_B / (A_B + eps)

    # --- convert to spectral radiance [W m^-2 sr^-1 nm^-1] ---
    Lnm = np.empty_like(radiance_rgb, dtype=np.float32)
    Lnm[..., 0] = radiance_rgb[..., 0] / (A_R + eps)
    Lnm[..., 1] = radiance_rgb[..., 1] / (A_G + eps)
    Lnm[..., 2] = radiance_rgb[..., 2] / (A_B + eps)

    # --- TOA reflectance ---
    rho_R = radiance_to_toa_reflectance(Lnm[..., 0], Ebar_R, cos_theta_s, d_au)
    rho_G = radiance_to_toa_reflectance(Lnm[..., 1], Ebar_G, cos_theta_s, d_au)
    rho_B = radiance_to_toa_reflectance(Lnm[..., 2], Ebar_B, cos_theta_s, d_au)

    return np.stack([rho_R, rho_G, rho_B], axis=-1).astype(np.float32)

def reflectance_stats_rgb(arr: np.ndarray, mask: np.ndarray | None = None, name: str = "") -> dict:
    """reflectance_stats_rgb(arr,mask,name) -> dict: Per-channel stats (min/max/mean/p1/p50/p99)."""
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 array, got {arr.shape}")
    m = mask.astype(bool) if mask is not None else None
    out = {"name": name, "channels": {}}
    for ci, ch in enumerate(["R", "G", "B"]):
        x = arr[..., ci].astype(np.float64)
        x = x[m] if m is not None else x.reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            out["channels"][ch] = {"n": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "p1": np.nan, "p50": np.nan, "p99": np.nan}
            continue
        p1, p50, p99 = np.percentile(x, [1, 50, 99])
        out["channels"][ch] = {
            "n": int(x.size),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "p1": float(p1),
            "p50": float(p50),
            "p99": float(p99),
        }
    return out