from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ----------------------------
# Project root on sys.path
# ----------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------
# Orekit init BEFORE importing sim_utils
# ----------------------------
import orekit
from orekit.pyhelpers import setup_orekit_curdir

_ = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

from settings import *  # noqa: F401,F403

import pykep as pk
from paseos import ActorBuilder, SpacecraftActor

from paseos.custom_paseos.utils.point_transformation import Point_Geodetic2ECI
from simulation.constants import R_earth
from simulation.sim_utils import init_eo_tools, init_attitude_models, link_eo_attitude
from simulation.plotting.plot_functions import plot_all_fov_footprints_plotly


class DummyTarget:
    def __init__(self, lat: float, lon: float, alt: float = 0.0):
        self.lat = float(lat)
        self.lon = float(lon)
        self.alt = float(alt)


def _close_polygon(p: np.ndarray) -> np.ndarray:
    """_close_polygon(p) -> np.ndarray: Ensure polygon is closed by appending first vertex."""
    p = np.asarray(p, float)
    if p.shape[0] < 3:
        return p
    if np.allclose(p[0], p[-1]):
        return p
    return np.vstack([p, p[0]])


def main() -> None:
    # =========================
    # USER INPUTS
    # =========================
    use_cue = True
    actor_name = "Cue_0"

    # time (UTC)
    t_datetime_utc = datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc)
    t0_pykep = pk.epoch_from_string(t_datetime_utc.strftime("%Y-%m-%d %H:%M:%S"))

    # satellite geodetic
    sat_lat_deg = 20.94716
    sat_lon_deg = -10.60062
    sat_alt_m = 613e3

    # target geodetic
    target_lat_deg = 27.94716
    target_lon_deg = -15.60062
    target_alt_m = 0.0
    target_geo = (target_lat_deg, target_lon_deg, target_alt_m)

    eul_default = [0.0, 0.0, 0.0]
    swath_cue = 45 * 10 ** 3  # m


    fov = math.degrees(2 * math.atan(swath_cue / (2 * (a_tip - R_earth))))  # deg

    # IMPORTANT: what does plot_all_fov_footprints_plotly expect?
    # try "latlon" first; if still no polygon, switch to "lonlat"
    PLOT_COORD_ORDER = "latlon"  # "latlon" or "lonlat"

    extension = "single_pointed_to_target"
    # =========================

    # satellite ECI from your geodetic
    sat_r_eci_m = np.array(
        Point_Geodetic2ECI(sat_lat_deg, sat_lon_deg, sat_alt_m, t_datetime_utc),
        dtype=float
    ).reshape(3)

    # velocity only matters for LVLH frame; use a reasonable circular-orbit magnitude
    # direction doesn’t need to be perfect for footprint plotting, but must be non-zero
    sat_v_eci_mps = np.array([0.0, 7500.0, 0.0], dtype=float)

    # actor scaffold
    actor = ActorBuilder.get_actor_scaffold(name=actor_name, actor_type=SpacecraftActor, epoch=t0_pykep)
    actor.running_ai = False

    ActorBuilder.set_custom_orbit(
        actor,
        lambda _t, r=sat_r_eci_m, v=sat_v_eci_mps: (list(r), list(v)),
        t0_pykep
    )
    ActorBuilder.set_geometric_model(actor, sat_mass)
    ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)
    actor.set_time(t0_pykep)


    eo_tools_dict = init_eo_tools([actor], [], fov, fov, offnadir_limit)
    att_models_dict = init_attitude_models(
        [actor], [],
        eul_default, eul_default,
        omega_max_rad, alpha_max_rad, zeta, wn_rad,
        offnadir_limit, offnadir_margin
    )
    link_eo_attitude(eo_tools_dict, att_models_dict)

    tool = eo_tools_dict[actor.name]
    att = att_models_dict[actor.name]
    tool._actor = actor

    # ------------------------------------------------------------
    # 1) Compute Euler to point to target (UNBOUNDED, NOW)
    # ------------------------------------------------------------
    los_lvlh, offnadir_deg = tool.point_to_target_unbounded(
        sat_r_eci_m, sat_v_eci_mps, target_geo, t_datetime_utc, frame="LVLH"
    )
    target_eul_deg = np.array(att.pointing_attitude_lvlh(los_lvlh), dtype=float)

    # ------------------------------------------------------------
    # 2) Compute footprint using that Euler
    #    (use eul_deg_override to remove any ambiguity)
    # ------------------------------------------------------------
    fov_latlon = np.asarray(
        tool.get_FovPoints(sat_r_eci_m, sat_v_eci_mps, t_datetime_utc, eul_deg_override=target_eul_deg),
        float
    )

    # close polygon for Plotly fill rendering
    fov_latlon = _close_polygon(fov_latlon)

    # If plot function expects lon/lat, swap
    if PLOT_COORD_ORDER.lower() == "lonlat":
        fov_for_plot = fov_latlon[:, [1, 0]]
    else:
        fov_for_plot = fov_latlon

    # ------------------------------------------------------------
    # 3) Print results
    # ------------------------------------------------------------
    print("\n--- Pointing solution ---")
    print(f"UTC time:         {t_datetime_utc.isoformat()}")
    print(f"Off-nadir (deg):  {offnadir_deg:.3f} (unbounded)")
    print(f"Euler (deg):      roll={target_eul_deg[0]:.3f}, pitch={target_eul_deg[1]:.3f}, yaw={target_eul_deg[2]:.3f}")

    print("\n--- Footprint vertices ---")
    for i, (a, b) in enumerate(fov_for_plot.tolist()):
        if PLOT_COORD_ORDER.lower() == "lonlat":
            print(f"P{i+1}: lon={a:.6f}, lat={b:.6f}")
        else:
            print(f"P{i+1}: lat={a:.6f}, lon={b:.6f}")

    # ------------------------------------------------------------
    # 4) Plot footprint + target
    # ------------------------------------------------------------
    all_fov_polygons = [fov_for_plot]

    dummy_target = DummyTarget(target_lat_deg, target_lon_deg, target_alt_m)
    all_targets = {0: dummy_target}
    observed_targets = {0: dummy_target}

    html_path = plot_all_fov_footprints_plotly(
        all_fov_polygons=all_fov_polygons,
        all_targets=all_targets,
        observed_targets=observed_targets,
        nPlanes=1,
        nSats=1,
        extension=extension,
        verbose=True,
        plot_whale_trajectories=False,
        whale_trajectories=None,
    )

    print(f"\nHTML written: {html_path}")
    print(f"Plot coord order used: {PLOT_COORD_ORDER}")


if __name__ == "__main__":
    main()