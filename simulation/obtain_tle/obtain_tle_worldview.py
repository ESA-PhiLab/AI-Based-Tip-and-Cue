#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from sgp4.api import Satrec


MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137

FUTURE_TOLERANCE_SECONDS = 1.0
SATCHECKER_HISTORY_WINDOW_DAYS = 3
SPACE_TRACK_WINDOW_DAYS = 14
SATCHECKER_BACKOFF_MINUTES = [0, 10, 30, 90, 180, 360, 720, 1440]


@dataclass(frozen=True)
class SatelliteInfo:
    name: str
    norad_id: int
    earliest_valid_utc: datetime
    decay_date_utc: datetime | None = None


@dataclass(frozen=True)
class TLERecord:
    satellite: SatelliteInfo
    source: str
    tle_epoch: str
    tle1: str
    tle2: str


@dataclass(frozen=True)
class OrbitalElements:
    hp_m: float
    ha_m: float
    a_km: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    M_deg: float
    nu_deg: float


def datetime_to_jd(dt_utc: datetime) -> float:
    """Convert timezone-aware UTC datetime to Julian Date."""
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware")
    dt_utc = dt_utc.astimezone(timezone.utc)

    y, m = dt_utc.year, dt_utc.month
    d = dt_utc.day
    hh = dt_utc.hour
    mm = dt_utc.minute
    ss = dt_utc.second + dt_utc.microsecond / 1e6

    if m <= 2:
        y -= 1
        m += 12

    a = y // 100
    b = 2 - a + (a // 4)
    jd0 = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5
    frac = (hh + (mm + ss / 60.0) / 60.0) / 24.0
    return jd0 + frac


def parse_tle_epoch_to_datetime(tle1: str) -> datetime:
    """Parse epoch from fixed-width or space-compressed TLE line 1."""
    tle1 = " ".join(tle1.strip().split())

    if len(tle1) >= 32:
        epoch_field = tle1[18:32].strip()
        if re.fullmatch(r"\d{5}\.\d+", epoch_field):
            yy = int(epoch_field[:2])
            day_frac = float(epoch_field[2:])
            year = 2000 + yy if yy < 57 else 1900 + yy
            day_int = int(day_frac)
            frac = day_frac - day_int
            dt0 = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int - 1)
            return dt0 + timedelta(seconds=frac * 86400.0)

    parts = tle1.split()
    if len(parts) >= 4 and re.fullmatch(r"\d{5}\.\d+", parts[3]):
        epoch_field = parts[3]
        yy = int(epoch_field[:2])
        day_frac = float(epoch_field[2:])
        year = 2000 + yy if yy < 57 else 1900 + yy
        day_int = int(day_frac)
        frac = day_frac - day_int
        dt0 = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_int - 1)
        return dt0 + timedelta(seconds=frac * 86400.0)

    raise ValueError(f"Could not parse TLE epoch from line: {tle1}")


def check_tle_epoch_for_request(
    tle1: str,
    requested_dt_utc: datetime,
    satellite_name: str,
    future_tolerance_seconds: float = FUTURE_TOLERANCE_SECONDS,
) -> tuple[datetime, float]:
    """Return TLE epoch and signed delta seconds versus requested time."""
    tle_epoch_dt = parse_tle_epoch_to_datetime(tle1)
    dt_sec = (tle_epoch_dt - requested_dt_utc).total_seconds()

    if dt_sec > future_tolerance_seconds:
        raise RuntimeError(
            f"{satellite_name}: TLE epoch {tle_epoch_dt.isoformat()} is too far after requested time "
            f"{requested_dt_utc.isoformat()} (delta={dt_sec:.6f} s, tol={future_tolerance_seconds:.6f} s)"
        )

    return tle_epoch_dt, float(dt_sec)


def normalize_tle_line(line: str) -> str:
    """Preserve TLE fixed-width formatting; only strip line endings."""
    return line.rstrip("\r\n")


def extract_tle_pair_from_text(text: str, norad_id: int) -> tuple[str, str] | None:
    """Extract TLE pair from scraped text, including merged blocks."""
    text = text.replace("\r", "\n")

    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    line1 = None
    line2 = None

    for ln in raw_lines:
        ln_norm = normalize_tle_line(ln)
        if ln_norm.startswith(f"1 {norad_id}U "):
            line1 = ln_norm
        elif ln_norm.startswith(f"2 {norad_id} "):
            line2 = ln_norm
        if line1 is not None and line2 is not None:
            return line1, line2

    flat = re.sub(r"\s+", " ", text)
    m = re.search(
        rf"(1\s+{norad_id}U\s+.*?)(?=\s+2\s+{norad_id}\s)(\s+2\s+{norad_id}\s+.*?)(?=\s+Source of the keplerian elements:|$)",
        flat,
    )
    if m:
        line1 = normalize_tle_line(m.group(1))
        line2 = normalize_tle_line(m.group(2))
        return line1, line2

    return None


def fetch_current_tle_from_n2yo(norad_id: int) -> tuple[str, str, str]:
    """Scrape current TLE from N2YO as last-resort fallback."""
    url = f"https://www.n2yo.com/satellite/?s={norad_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    html = r.text

    text = html.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"&nbsp;", " ", text)

    pair = extract_tle_pair_from_text(text, norad_id)
    if pair is None:
        raise RuntimeError(f"Could not scrape current TLE from N2YO for NORAD {norad_id}")

    tle1, tle2 = pair

    print(tle1, tle2)
    try:
        Satrec.twoline2rv(tle1.rstrip("\r\n"), tle2.rstrip("\r\n"))
    except Exception as e:
        raise RuntimeError(f"N2YO returned unparseable TLE for NORAD {norad_id}: {e}") from e

    tle_epoch = parse_tle_epoch_to_datetime(tle1).isoformat()
    return tle1, tle2, tle_epoch


def fetch_nearest_tle_from_satchecker(norad_id: int, jd_epoch: float, data_source: str | None = None) -> tuple[str, str, str]:
    """Fetch nearest TLE lines and epoch from SatChecker."""
    url = "https://satchecker.cps.iau.org/tools/get-nearest-tle/"
    params = {"id": str(norad_id), "id_type": "catalog", "epoch": str(jd_epoch)}
    if data_source:
        params["data_source"] = data_source

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    if not isinstance(payload, list) or len(payload) == 0:
        raise RuntimeError(f"Unexpected SatChecker nearest response: {payload!r}")

    tle_data = payload[0].get("tle_data", [])
    if not tle_data:
        raise RuntimeError(f"No TLE returned for NORAD {norad_id} at JD {jd_epoch}")

    item = tle_data[0]
    return normalize_tle_line(item["tle_line1"]), normalize_tle_line(item["tle_line2"]), item.get("epoch", "unknown")


def fetch_satchecker_tle_not_after(
    sat: SatelliteInfo,
    dt_utc: datetime,
    future_tolerance_seconds: float = FUTURE_TOLERANCE_SECONDS,
) -> tuple[str, str, str]:
    """Query SatChecker nearest with backoff until TLE is not meaningfully after dt_utc."""
    last_error: Exception | None = None

    for backoff_min in SATCHECKER_BACKOFF_MINUTES:
        query_dt = dt_utc - timedelta(minutes=backoff_min)
        jd = datetime_to_jd(query_dt)

        try:
            tle1, tle2, tle_epoch = fetch_nearest_tle_from_satchecker(sat.norad_id, jd, data_source=None)
            _tle_epoch_dt, dt_sec = check_tle_epoch_for_request(
                tle1=tle1,
                requested_dt_utc=dt_utc,
                satellite_name=sat.name,
                future_tolerance_seconds=future_tolerance_seconds,
            )

            label = "SatChecker nearest"
            if backoff_min > 0:
                label = f"SatChecker nearest backoff={backoff_min}min"

            if dt_sec > 0.0:
                print(
                    f"[INFO] {sat.name}: accepted near-future nearest TLE within tolerance "
                    f"(delta={dt_sec:.6f} s, tol={future_tolerance_seconds:.6f} s)"
                )

            return tle1, tle2, f"{tle_epoch} [{label}]"

        except Exception as e:
            last_error = e

    if last_error is None:
        raise RuntimeError(f"{sat.name}: SatChecker nearest failed without specific error")
    raise RuntimeError(str(last_error))


def fetch_tle_history_from_satchecker(norad_id: int, start_jd: float, end_jd: float) -> list[tuple[datetime, str, str]]:
    """Fetch raw TLE history from SatChecker over a time window."""
    url = "https://satchecker.cps.iau.org/tools/get-tle-data/"
    params = {
        "id": str(norad_id),
        "id_type": "catalog",
        "start_date_jd": str(start_jd),
        "end_date_jd": str(end_jd),
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected SatChecker raw-history response type: {type(payload).__name__}")

    tle_data = payload.get("data", [])
    if not isinstance(tle_data, list) or not tle_data:
        raise RuntimeError(f"No SatChecker raw-history TLE data for NORAD {norad_id}: {payload!r}")

    out: list[tuple[datetime, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in tle_data:
        try:
            tle1 = normalize_tle_line(item["tle_line1"])
            tle2 = normalize_tle_line(item["tle_line2"])
            epoch_dt = parse_tle_epoch_to_datetime(tle1)
        except Exception:
            continue

        key = (tle1, tle2)
        if key in seen:
            continue
        seen.add(key)
        out.append((epoch_dt, tle1, tle2))

    if not out:
        raise RuntimeError(f"SatChecker raw-history returned no parseable TLEs for NORAD {norad_id}")

    out.sort(key=lambda x: x[0])
    return out


def select_best_tle_from_history(
    history: list[tuple[datetime, str, str]],
    requested_dt_utc: datetime,
    satellite_name: str,
    future_tolerance_seconds: float = FUTURE_TOLERANCE_SECONDS,
) -> tuple[str, str, str]:
    """Choose latest TLE before request, otherwise earliest after within tolerance."""
    before = [(epoch, l1, l2) for epoch, l1, l2 in history if epoch <= requested_dt_utc]
    if before:
        best_epoch, best_l1, best_l2 = before[-1]
        return best_l1, best_l2, f"{best_epoch.isoformat()} [SatChecker raw history]"

    after = [(epoch, l1, l2) for epoch, l1, l2 in history if epoch > requested_dt_utc]
    if after:
        best_epoch, best_l1, best_l2 = after[0]
        delta_sec = (best_epoch - requested_dt_utc).total_seconds()
        if delta_sec <= future_tolerance_seconds:
            return best_l1, best_l2, f"{best_epoch.isoformat()} [SatChecker raw history + tolerance]"
        raise RuntimeError(
            f"{satellite_name}: nearest raw-history TLE is after requested time by {delta_sec:.6f} s"
        )

    raise RuntimeError(f"{satellite_name}: no suitable TLE found in raw history")


def fetch_tle_before_and_after_for_interpolation_from_history(
    history: list[tuple[datetime, str, str]],
    requested_dt_utc: datetime,
    satellite_name: str,
) -> tuple[tuple[str, str, datetime], tuple[str, str, datetime]]:
    """Choose bracketing TLEs from a history list."""
    before = [(epoch, l1, l2) for epoch, l1, l2 in history if epoch < requested_dt_utc]
    after = [(epoch, l1, l2) for epoch, l1, l2 in history if epoch > requested_dt_utc]

    if not before or not after:
        raise RuntimeError(f"{satellite_name}: could not obtain TLEs bracketing requested time from history")

    epoch_b, tle1_b, tle2_b = before[-1]
    epoch_a, tle1_a, tle2_a = after[0]
    return (tle1_b, tle2_b, epoch_b), (tle1_a, tle2_a, epoch_a)


def fetch_tle_from_spacetrack(
    norad_id: int,
    target_dt_utc: datetime,
    identity: str,
    password: str,
    window_days: int = SPACE_TRACK_WINDOW_DAYS,
) -> tuple[str, str, str]:
    """Fetch closest historical TLE from Space-Track gp_history at or before target time."""
    if target_dt_utc.tzinfo is None:
        raise ValueError("target_dt_utc must be timezone-aware")

    session = requests.Session()

    login_url = "https://www.space-track.org/ajaxauth/login"
    login_resp = session.post(
        login_url,
        data={"identity": identity, "password": password},
        timeout=30,
    )
    login_resp.raise_for_status()

    start_dt = (target_dt_utc - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_dt = target_dt_utc.strftime("%Y-%m-%d")

    query_url = (
        "https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{norad_id}/"
        f"EPOCH/{start_dt}--{end_dt}/"
        "orderby/EPOCH desc/format/tle"
    )

    r = session.get(query_url, timeout=60)
    r.raise_for_status()
    txt = r.text.strip()

    if not txt:
        raise RuntimeError(f"No Space-Track gp_history data for NORAD {norad_id} in {start_dt}..{end_dt}")

    lines = [normalize_tle_line(ln) for ln in txt.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected Space-Track response for NORAD {norad_id}")

    pairs: list[tuple[datetime, str, str]] = []
    for i in range(0, len(lines) - 1, 2):
        l1 = lines[i]
        l2 = lines[i + 1]
        if not l1.startswith("1 ") or not l2.startswith("2 "):
            continue
        try:
            epoch_dt = parse_tle_epoch_to_datetime(l1)
        except Exception:
            continue
        if epoch_dt <= target_dt_utc:
            pairs.append((epoch_dt, l1, l2))

    if not pairs:
        raise RuntimeError(f"No valid historical Space-Track TLE before requested time for NORAD {norad_id}")

    best_epoch, best_l1, best_l2 = pairs[0]
    return best_l1, best_l2, f"{best_epoch.isoformat()} [Space-Track gp_history]"


def propagate_tle_to_teme(tle1: str, tle2: str, dt_utc: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Propagate TLE with SGP4 and return TEME position [km] and velocity [km/s]."""
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware")
    dt_utc = dt_utc.astimezone(timezone.utc)

    sat = Satrec.twoline2rv(tle1.strip(), tle2.strip())
    jd = datetime_to_jd(dt_utc)
    jd_int = float(int(jd))
    fr = jd - jd_int

    err, r_km, v_km_s = sat.sgp4(jd_int, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code: {err}")

    return np.array(r_km, dtype=np.float64), np.array(v_km_s, dtype=np.float64)


def wrap_deg(angle_deg: float) -> float:
    """Wrap angle to [0,360) deg."""
    return float(angle_deg % 360.0)


def rv_to_coe(r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float = MU_EARTH_KM3_S2) -> dict[str, float]:
    """Convert inertial state vectors to classical orbital elements."""
    r = np.array(r_km, dtype=np.float64)
    v = np.array(v_km_s, dtype=np.float64)

    r_norm = float(np.linalg.norm(r))
    v_norm = float(np.linalg.norm(v))

    h_vec = np.cross(r, v)
    h_norm = float(np.linalg.norm(h_vec))

    k_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_vec = np.cross(k_vec, h_vec)
    n_norm = float(np.linalg.norm(n_vec))

    e_vec = (np.cross(v, h_vec) / mu_km3_s2) - (r / r_norm)
    e = float(np.linalg.norm(e_vec))

    energy = 0.5 * v_norm * v_norm - mu_km3_s2 / r_norm
    if abs(energy) < 1e-15:
        raise RuntimeError("Parabolic orbit encountered; semi-major axis undefined.")
    a_km = -mu_km3_s2 / (2.0 * energy)

    i_rad = math.acos(max(-1.0, min(1.0, h_vec[2] / h_norm)))

    if n_norm > 1e-12:
        raan_rad = math.acos(max(-1.0, min(1.0, n_vec[0] / n_norm)))
        if n_vec[1] < 0.0:
            raan_rad = 2.0 * math.pi - raan_rad
    else:
        raan_rad = 0.0

    if n_norm > 1e-12 and e > 1e-12:
        argp_rad = math.acos(max(-1.0, min(1.0, np.dot(n_vec, e_vec) / (n_norm * e))))
        if e_vec[2] < 0.0:
            argp_rad = 2.0 * math.pi - argp_rad
    else:
        argp_rad = 0.0

    if e > 1e-12:
        nu_rad = math.acos(max(-1.0, min(1.0, np.dot(e_vec, r) / (e * r_norm))))
        if np.dot(r, v) < 0.0:
            nu_rad = 2.0 * math.pi - nu_rad
    else:
        nu_rad = 0.0

    if e >= 1.0:
        raise RuntimeError("Non-elliptic orbit encountered.")

    E_rad = 2.0 * math.atan2(
        math.sqrt(1.0 - e) * math.sin(nu_rad / 2.0),
        math.sqrt(1.0 + e) * math.cos(nu_rad / 2.0),
    )
    M_rad = (E_rad - e * math.sin(E_rad)) % (2.0 * math.pi)

    return {
        "a_km": float(a_km),
        "e": float(e),
        "i_deg": wrap_deg(math.degrees(i_rad)),
        "raan_deg": wrap_deg(math.degrees(raan_rad)),
        "argp_deg": wrap_deg(math.degrees(argp_rad)),
        "nu_deg": wrap_deg(math.degrees(nu_rad)),
        "M_deg": wrap_deg(math.degrees(M_rad)),
    }


def interpolate_angles_deg(a1: float, a2: float, w: float) -> float:
    """Interpolate angles correctly over 360 wrap."""
    da = ((a2 - a1 + 180.0) % 360.0) - 180.0
    return float((a1 + w * da) % 360.0)


def interpolate_coe(coe1: dict[str, float], coe2: dict[str, float], t1: datetime, t2: datetime, t: datetime) -> dict[str, float]:
    """Linear interpolation of orbital elements."""
    if not (t1 < t < t2):
        raise ValueError("Interpolation time must lie strictly between t1 and t2")

    w = (t - t1).total_seconds() / (t2 - t1).total_seconds()

    return {
        "a_km": float(coe1["a_km"] + w * (coe2["a_km"] - coe1["a_km"])),
        "e": float(coe1["e"] + w * (coe2["e"] - coe1["e"])),
        "i_deg": interpolate_angles_deg(coe1["i_deg"], coe2["i_deg"], w),
        "raan_deg": interpolate_angles_deg(coe1["raan_deg"], coe2["raan_deg"], w),
        "argp_deg": interpolate_angles_deg(coe1["argp_deg"], coe2["argp_deg"], w),
        "M_deg": interpolate_angles_deg(coe1["M_deg"], coe2["M_deg"], w),
    }


def interpolated_orbital_elements_from_bracketing_tles(
    tle1_before: str,
    tle2_before: str,
    tle1_after: str,
    tle2_after: str,
    dt_utc: datetime,
) -> OrbitalElements:
    """Build orbital elements by interpolation between two bracketing TLE-derived COE sets."""
    t_before = parse_tle_epoch_to_datetime(tle1_before)
    t_after = parse_tle_epoch_to_datetime(tle1_after)

    if not (t_before < dt_utc < t_after):
        raise RuntimeError(
            f"Interpolation bracket invalid: before={t_before.isoformat()} requested={dt_utc.isoformat()} after={t_after.isoformat()}"
        )

    r_before, v_before = propagate_tle_to_teme(tle1_before, tle2_before, t_before)
    r_after, v_after = propagate_tle_to_teme(tle1_after, tle2_after, t_after)

    coe_before = rv_to_coe(r_before, v_before)
    coe_after = rv_to_coe(r_after, v_after)
    coe_interp = interpolate_coe(coe_before, coe_after, t_before, t_after, dt_utc)

    a_km = float(coe_interp["a_km"])
    e = float(coe_interp["e"])

    rp_km = a_km * (1.0 - e)
    ra_km = a_km * (1.0 + e)

    hp_m = (rp_km - R_EARTH_KM) * 1000.0
    ha_m = (ra_km - R_EARTH_KM) * 1000.0

    return OrbitalElements(
        hp_m=float(hp_m),
        ha_m=float(ha_m),
        a_km=a_km,
        e=e,
        i_deg=float(coe_interp["i_deg"]),
        raan_deg=float(coe_interp["raan_deg"]),
        argp_deg=float(coe_interp["argp_deg"]),
        M_deg=float(coe_interp["M_deg"]),
        nu_deg=float("nan"),
    )


def orbital_elements_from_tle_at_time(tle1: str, tle2: str, dt_utc: datetime) -> OrbitalElements:
    """Propagate TLE to dt_utc and compute orbital elements."""
    r_km, v_km_s = propagate_tle_to_teme(tle1, tle2, dt_utc)
    coe = rv_to_coe(r_km, v_km_s)

    a_km = coe["a_km"]
    e = coe["e"]

    rp_km = a_km * (1.0 - e)
    ra_km = a_km * (1.0 + e)

    hp_m = (rp_km - R_EARTH_KM) * 1000.0
    ha_m = (ra_km - R_EARTH_KM) * 1000.0

    return OrbitalElements(
        hp_m=float(hp_m),
        ha_m=float(ha_m),
        a_km=float(a_km),
        e=float(e),
        i_deg=float(coe["i_deg"]),
        raan_deg=float(coe["raan_deg"]),
        argp_deg=float(coe["argp_deg"]),
        M_deg=float(coe["M_deg"]),
        nu_deg=float(coe["nu_deg"]),
    )


def get_worldview_satellites() -> list[SatelliteInfo]:
    """Return WV-2, WV-3, and WV Legion 1-6 with valid time ranges."""
    return [
        SatelliteInfo(name="WorldView-2", norad_id=35946, earliest_valid_utc=datetime(2009, 10, 8, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView-3", norad_id=40115, earliest_valid_utc=datetime(2014, 8, 13, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView Legion 1", norad_id=59625, earliest_valid_utc=datetime(2024, 5, 2, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView Legion 2", norad_id=59626, earliest_valid_utc=datetime(2024, 5, 2, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView Legion 3", norad_id=60452, earliest_valid_utc=datetime(2024, 8, 15, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView Legion 4", norad_id=60453, earliest_valid_utc=datetime(2024, 8, 15, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView Legion 5", norad_id=62900, earliest_valid_utc=datetime(2025, 2, 4, 0, 0, 0, tzinfo=timezone.utc)),
        SatelliteInfo(name="WorldView Legion 6", norad_id=62901, earliest_valid_utc=datetime(2025, 2, 4, 0, 0, 0, tzinfo=timezone.utc)),
    ]


def fetch_best_available_tle(
    sat: SatelliteInfo,
    dt_utc: datetime,
    allow_n2yo_current_fallback: bool = False,
) -> TLERecord:
    """Try SatChecker raw history, then nearest, then Space-Track, then optional N2YO."""
    if dt_utc < sat.earliest_valid_utc:
        raise RuntimeError(
            f"{sat.name} was not yet in orbit at requested time: requested={dt_utc.isoformat()} earliest_valid={sat.earliest_valid_utc.isoformat()}"
        )

    if sat.decay_date_utc is not None and dt_utc > sat.decay_date_utc:
        raise RuntimeError(
            f"{sat.name} decayed before requested date: requested={dt_utc.isoformat()} decay={sat.decay_date_utc.isoformat()}"
        )

    start_jd = datetime_to_jd(dt_utc - timedelta(days=SATCHECKER_HISTORY_WINDOW_DAYS))
    end_jd = datetime_to_jd(dt_utc + timedelta(days=SATCHECKER_HISTORY_WINDOW_DAYS))

    try:
        history = fetch_tle_history_from_satchecker(sat.norad_id, start_jd, end_jd)
        tle1, tle2, tle_epoch = select_best_tle_from_history(history, dt_utc, sat.name)
        return TLERecord(satellite=sat, source="SatChecker raw history", tle_epoch=tle_epoch, tle1=tle1, tle2=tle2)
    except Exception as e_history:
        print(f"[WARN] SatChecker raw-history failed for {sat.name} ({sat.norad_id}): {e_history}")

    try:
        tle1, tle2, tle_epoch = fetch_satchecker_tle_not_after(sat=sat, dt_utc=dt_utc)
        return TLERecord(satellite=sat, source="SatChecker nearest", tle_epoch=tle_epoch, tle1=tle1, tle2=tle2)
    except Exception as e_nearest:
        print(f"[WARN] SatChecker nearest failed for {sat.name} ({sat.norad_id}): {e_nearest}")

    st_user = os.getenv("SPACETRACK_IDENTITY", "").strip()
    st_pass = os.getenv("SPACETRACK_PASSWORD", "").strip()

    if st_user and st_pass:
        try:
            tle1, tle2, tle_epoch = fetch_tle_from_spacetrack(
                norad_id=sat.norad_id,
                target_dt_utc=dt_utc,
                identity=st_user,
                password=st_pass,
                window_days=SPACE_TRACK_WINDOW_DAYS,
            )
            return TLERecord(satellite=sat, source="Space-Track gp_history", tle_epoch=tle_epoch, tle1=tle1, tle2=tle2)
        except Exception as e_spacetrack:
            print(f"[WARN] Space-Track failed for {sat.name} ({sat.norad_id}): {e_spacetrack}")

    if allow_n2yo_current_fallback:
        try:
            tle1, tle2, tle_epoch = fetch_current_tle_from_n2yo(sat.norad_id)
            check_tle_epoch_for_request(
                tle1=tle1,
                requested_dt_utc=dt_utc,
                satellite_name=sat.name,
                future_tolerance_seconds=FUTURE_TOLERANCE_SECONDS,
            )
            return TLERecord(satellite=sat, source="N2YO current TLE fallback", tle_epoch=tle_epoch, tle1=tle1, tle2=tle2)
        except Exception as e_n2yo:
            print(f"[WARN] N2YO fallback failed for {sat.name} ({sat.norad_id}): {e_n2yo}")

    raise RuntimeError(
        f"No valid TLE source succeeded for {sat.name} ({sat.norad_id}). "
        "Best fix: use SatChecker raw-history or Space-Track gp_history."
    )


def fetch_bracketing_tles_for_interpolation(
    sat: SatelliteInfo,
    dt_utc: datetime,
) -> tuple[tuple[str, str, datetime], tuple[str, str, datetime]]:
    """Try SatChecker raw history first for interpolation bracket."""
    start_jd = datetime_to_jd(dt_utc - timedelta(days=SATCHECKER_HISTORY_WINDOW_DAYS))
    end_jd = datetime_to_jd(dt_utc + timedelta(days=SATCHECKER_HISTORY_WINDOW_DAYS))

    history = fetch_tle_history_from_satchecker(sat.norad_id, start_jd, end_jd)
    return fetch_tle_before_and_after_for_interpolation_from_history(history, dt_utc, sat.name)


def print_python_assignment_block(name: str, elems: OrbitalElements) -> None:
    """Print orbital parameters in assignment format."""
    safe_name = name.lower().replace("-", "_").replace(" ", "_")

    print(f"\n# {name}")
    print(f"hp_{safe_name} = {elems.hp_m:.3f}  # Perigee altitude [m]")
    print(f"ha_{safe_name} = {elems.ha_m:.3f}  # Apogee altitude [m]")
    print(f"i_{safe_name}_deg = {elems.i_deg:.8f}  # Inclination [deg]")
    print(f"RAAN_{safe_name}_deg = {elems.raan_deg:.8f}  # RAAN [deg]")
    print(f"argp_{safe_name}_deg = {elems.argp_deg:.8f}  # Argument of periapsis [deg]")
    print(f"M_{safe_name}_deg = {elems.M_deg:.8f}  # Mean anomaly [deg]")


def print_full_summary(record: TLERecord, elems: OrbitalElements, requested_dt_utc: datetime) -> None:
    """Print TLE and derived orbital elements for one satellite."""
    print("\n" + "=" * 100)
    print(f"{record.satellite.name} | NORAD {record.satellite.norad_id}")
    print(f"Requested epoch: {requested_dt_utc.isoformat()}")
    print(f"TLE source     : {record.source}")
    print(f"TLE epoch      : {record.tle_epoch}")
    print(record.tle1)
    print(record.tle2)

    print("\nDerived orbital elements at requested time:")
    print(f"a_km      = {elems.a_km:.6f}")
    print(f"e         = {elems.e:.10f}")
    print(f"hp_m      = {elems.hp_m:.3f}")
    print(f"ha_m      = {elems.ha_m:.3f}")
    print(f"i_deg     = {elems.i_deg:.8f}")
    print(f"raan_deg  = {elems.raan_deg:.8f}")
    print(f"argp_deg  = {elems.argp_deg:.8f}")
    print(f"M_deg     = {elems.M_deg:.8f}")
    print(f"nu_deg    = {elems.nu_deg:.8f}")

    print_python_assignment_block(record.satellite.name, elems)


if __name__ == "__main__":
    t0 = datetime(2025, 9, 20, 2, 33, 39, tzinfo=timezone.utc)

    ALLOW_N2YO_CURRENT_FALLBACK = False

    satellites = get_worldview_satellites()

    print(f"Requested UTC time: {t0.isoformat()}")
    print(f"Number of satellites requested: {len(satellites)}")
    print(f"N2YO current fallback enabled: {ALLOW_N2YO_CURRENT_FALLBACK}")
    print(f"Future tolerance [s]: {FUTURE_TOLERANCE_SECONDS}")
    print(f"SatChecker raw-history window [days]: {SATCHECKER_HISTORY_WINDOW_DAYS}")
    print(f"Space-Track window [days]: {SPACE_TRACK_WINDOW_DAYS}")

    successes: list[str] = []
    failures: list[str] = []

    for sat in satellites:
        try:
            record = fetch_best_available_tle(
                sat=sat,
                dt_utc=t0,
                allow_n2yo_current_fallback=ALLOW_N2YO_CURRENT_FALLBACK,
            )
            elems = orbital_elements_from_tle_at_time(record.tle1, record.tle2, t0)
            print_full_summary(record, elems, t0)
            successes.append(f"{sat.name}: direct TLE")

        except Exception as e_direct:
            print(f"[WARN] Direct TLE route failed for {sat.name}: {e_direct}")

            try:
                before_data, after_data = fetch_bracketing_tles_for_interpolation(sat, t0)
                tle1_b, tle2_b, epoch_b = before_data
                tle1_a, tle2_a, epoch_a = after_data

                elems_interp = interpolated_orbital_elements_from_bracketing_tles(
                    tle1_before=tle1_b,
                    tle2_before=tle2_b,
                    tle1_after=tle1_a,
                    tle2_after=tle2_a,
                    dt_utc=t0,
                )

                fake_record = TLERecord(
                    satellite=sat,
                    source="Interpolated from SatChecker raw-history bracket",
                    tle_epoch=f"{epoch_b.isoformat()} .. {epoch_a.isoformat()}",
                    tle1=tle1_b,
                    tle2=tle2_b,
                )

                print_full_summary(fake_record, elems_interp, t0)
                successes.append(f"{sat.name}: interpolated")

            except Exception as e_interp:
                print("\n" + "=" * 100)
                print(f"{sat.name} | NORAD {sat.norad_id}")
                print("Could not compute orbital parameters.")
                print(f"Direct route reason: {e_direct}")
                print(f"Interpolation reason: {e_interp}")
                failures.append(f"{sat.name}: failed")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("Succeeded:")
    for item in successes:
        print(f"  - {item}")
    print("Failed:")
    for item in failures:
        print(f"  - {item}")
    print("=" * 100)
    print("Done.")