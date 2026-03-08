import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from pySMARTS.main import SMARTSTimeLocation, IOUT_to_code
from matplotlib import pyplot as plt


def delete_if_exists(*paths) -> None:
    """Delete each path if it exists; ignore missing paths."""
    for p in paths:
        if p is None:
            continue
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass


def _as_utc_datetime(dt: datetime) -> datetime:
    """Return timezone-aware datetime in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def generate_sun_and_sky_spds(datetime_utc: datetime,
                              target_lat: float,
                              target_lon: float,
                              target_alt_m: float,
                              out_sun_spd: str | Path,
                              out_sky_spd: str | Path,
                              wavelength_range: tuple[float, float] = (350.0, 2500.0),
                              timezone_offset: int = 0,
                              material: str = "Water",
                              plot_spd: bool = False) -> tuple[str, str]:
    """Generate SMARTS sun (direct normal irradiance) and sky (diffuse radiance=E_dif/pi) SPDs and write to disk."""
    out_sun_spd = Path(out_sun_spd)
    out_sky_spd = Path(out_sky_spd)

    # Always remove stale files first (prevents Mitsuba using old spectra when SMARTS fails)
    delete_if_exists(out_sun_spd, out_sky_spd)

    min_wvl, max_wvl = float(wavelength_range[0]), float(wavelength_range[1])
    if max_wvl <= min_wvl:
        raise ValueError(f"Invalid wavelength_range: {wavelength_range}")

    iout_dir = IOUT_to_code("Direct normal irradiance W m-2")
    iout_dif = IOUT_to_code("Diffuse horizontal irradiance W m-2")
    if iout_dir is None or iout_dif is None:
        raise RuntimeError("SMARTS IOUT codes not available for required outputs.")

    dt_utc = _as_utc_datetime(datetime_utc)
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    out_sun_spd.parent.mkdir(parents=True, exist_ok=True)
    out_sky_spd.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_dir = SMARTSTimeLocation(
            IOUT=iout_dir,
            YEAR=str(dt_utc.year),
            MONTH=str(dt_utc.month),
            DAY=str(dt_utc.day),
            HOUR=f"{hour_utc:.4f}",
            ZONE=str(timezone_offset),
            LATIT=str(target_lat),
            LONGIT=str(target_lon),
            ALTIT=str(float(target_alt_m) / 1000.0),
            min_wvl=str(min_wvl),
            max_wvl=str(max_wvl),
            material=material
        )

        df_dif = SMARTSTimeLocation(
            IOUT=iout_dif,
            YEAR=str(dt_utc.year),
            MONTH=str(dt_utc.month),
            DAY=str(dt_utc.day),
            HOUR=f"{hour_utc:.4f}",
            ZONE=str(timezone_offset),
            LATIT=str(target_lat),
            LONGIT=str(target_lon),
            ALTIT=str(float(target_alt_m) / 1000.0),
            min_wvl=str(min_wvl),
            max_wvl=str(max_wvl),
            material=material
        )

        wl = df_dir.iloc[:, 0].values.astype(float)         # [nm]
        e_dir = df_dir.iloc[:, 1].values.astype(float)      # [W/m^2/nm] (direct normal irradiance)
        e_dif_h = df_dif.iloc[:, 1].values.astype(float)    # [W/m^2/nm] (diffuse horizontal irradiance)

        if wl.size == 0 or e_dir.size == 0 or e_dif_h.size == 0:
            raise RuntimeError("SMARTS returned empty spectra.")
        if wl.size != e_dir.size or wl.size != e_dif_h.size:
            raise RuntimeError("SMARTS wavelength grid mismatch between outputs.")

        l_sky = e_dif_h / np.pi  # [W/m^2/sr/nm] isotropic radiance assumption

        with out_sun_spd.open("w", encoding="utf-8") as f:
            for w, v in zip(wl, e_dir):
                f.write(f"{w:.2f} {v:.8e}\n")

        with out_sky_spd.open("w", encoding="utf-8") as f:
            for w, v in zip(wl, l_sky):
                f.write(f"{w:.2f} {v:.8e}\n")

    except Exception:
        # If anything fails, delete outputs so the caller never uses stale files
        delete_if_exists(out_sun_spd, out_sky_spd)
        raise

    if plot_spd:
        plt.figure(figsize=(8, 5))
        plt.plot(wl, e_dif_h, label="Diffuse horizontal irradiance (E_dif, sky)")
        plt.plot(wl, e_dir, label="Direct normal irradiance (E_dir,n, sun)")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Spectral Irradiance [W m⁻² nm⁻¹]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return str(out_sun_spd), str(out_sky_spd)


if __name__ == "__main__":
    import os

    dt = datetime(2025, 6, 11, 16, 0, 0, tzinfo=timezone.utc)
    lat, lon, alt_m = 0.0, 0.0, 0.0

    dt= datetime(2025, 3, 11, 14, 0, 0, tzinfo=timezone.utc)

    lat, lon, alt_m = 53.0, 0.0, 0.0



    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    spd_dir = script_dir / "spd_files"
    sun_spd = spd_dir / "solar_direct_normal.spd"
    sky_spd = spd_dir / "sky_diffuse_radiance.spd"

    try:
        out_sun, out_sky = generate_sun_and_sky_spds(
            datetime_utc=dt,
            target_lat=lat,
            target_lon=lon,
            target_alt_m=alt_m,
            out_sun_spd=sun_spd,
            out_sky_spd=sky_spd,
            wavelength_range=(250.0, 2000.0),
            timezone_offset=0,
            material="Water",
            plot_spd=True
        )
        print("Generated:")
        print("  Sun SPD:", out_sun)
        print("  Sky SPD:", out_sky)
    except Exception as e:
        print(f"SMARTS failed: {repr(e)}")
        print("Deleted any pre-existing SPD outputs to avoid stale spectra.")



    # https://www.imgonline.com.ua/eng/cut-photo-into-pieces.php
