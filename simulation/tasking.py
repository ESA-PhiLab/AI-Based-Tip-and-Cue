import numpy as np
from datetime import timedelta


def wrap_lon_deg(lon_deg): return ((lon_deg + 180.0) % 360.0) - 180.0


def lon_diff_deg(lon_a_deg, lon_b_deg):
    """Return wrapped absolute longitude difference in degrees."""
    return abs(wrap_lon_deg(float(lon_a_deg) - float(lon_b_deg)))


def central_angle_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Return great-circle central angle in degrees between two lat/lon points."""
    lat1 = np.radians(float(lat1_deg))
    lon1 = np.radians(float(lon1_deg))
    lat2 = np.radians(float(lat2_deg))
    lon2 = np.radians(float(lon2_deg))

    cos_d = (
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    )
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_d)))


def quick_task_geo_filter(sat_lat_deg, sat_lon_deg, tgt_lat_deg, tgt_lon_deg, max_dlat_deg=22.0, max_dlon_deg=40.0, max_central_angle_deg=40.0):
    """Cheap lat/lon prefilter before expensive visibility and slew checks."""
    dlat = abs(float(tgt_lat_deg) - float(sat_lat_deg))
    dlon = lon_diff_deg(float(tgt_lon_deg), float(sat_lon_deg))

    if dlat > max_dlat_deg:
        return False, None

    if dlon > max_dlon_deg:
        return False, None

    central_deg = central_angle_deg(sat_lat_deg, sat_lon_deg, tgt_lat_deg, tgt_lon_deg)
    if central_deg > max_central_angle_deg:
        return False, central_deg

    return True, central_deg


def ensure_cue_task_state(eo_tools_dict, cue_actors):
    """Initialize per-Cue tasking state on EO tools."""
    for actor in cue_actors:
        eo_tools_dict[actor.name].t_task_assigned = None
        eo_tools_dict[actor.name].t_to_obs_expected = None
        eo_tools_dict[actor.name].current_task = None
        eo_tools_dict[actor.name].task_queue = []
        eo_tools_dict[actor.name].pointing_vec_lvlh_target = None
        eo_tools_dict[actor.name].offnadir_unbound_target = None
        eo_tools_dict[actor.name].move_set = False
        eo_tools_dict[actor.name].time_to_obs_target = None
        eo_tools_dict[actor.name].offnadir_at_obs_target = None
        eo_tools_dict[actor.name].slew_stab_time = None


def add_tip_confirmed_task(whale, whale_idx, tasked_targets, t_datetime):
    """Add a Tip-confirmed target to the shared global Cue task pool."""
    whale.assigned_cue = None
    whale.t_tasked_tip = t_datetime
    tasked_targets[whale_idx] = whale
    whale.ai_class_predicted = "whale-tipped"


def release_current_task(actor_name, eo_tools_dict, att_models_dict, all_targets, eul_default):
    """Release a Cue reservation without removing the target from the global pool."""
    current_task = eo_tools_dict[actor_name].current_task

    if current_task is not None:
        task_id = current_task.get("target_id")
        if task_id in all_targets:
            all_targets[task_id].assigned_cue = None

    eo_tools_dict[actor_name].current_task = None
    eo_tools_dict[actor_name].t_task_assigned = None
    eo_tools_dict[actor_name].t_to_obs_expected = None
    eo_tools_dict[actor_name].pointing_vec_lvlh_target = None
    eo_tools_dict[actor_name].offnadir_unbound_target = None
    eo_tools_dict[actor_name].move_set = False
    eo_tools_dict[actor_name].time_to_obs_target = None
    eo_tools_dict[actor_name].offnadir_at_obs_target = None
    att_models_dict[actor_name]._new_target_attitude_deg = np.asarray(eul_default, float)


def claim_best_global_task(actor, r_vec, v_vec, t_datetime, tasked_targets, all_targets, eo_tools_dict, satellite_specs, avg_time_delay, elevation_min, omega_max_rad, alpha_max_rad, zeta, wn_rad, offnadir_margin, sim_step_seconds, delay_transmission_TC, point_eci_to_geodetic_fn):
    """Claim the best shared task for one Cue satellite using cheap geo filtering first."""
    actor_name = actor.name

    sat_lat, sat_lon, sat_alt = point_eci_to_geodetic_fn(r_vec[0], r_vec[1], r_vec[2], t_datetime).flatten()
    sat_lat = float(sat_lat)
    sat_lon = float(sat_lon)
    sat_alt = float(sat_alt)

    candidate_tasks = []

    for whale_idx, whale in tasked_targets.items():
        if whale.state_observing >= 2:
            continue
        if whale.state_confirming >= 2:
            continue
        if whale.t_confirmed_tip is None:
            continue
        if t_datetime <= whale.t_confirmed_tip + timedelta(seconds=delay_transmission_TC):
            continue
        if whale.coord_observed is None:
            continue
        if whale.assigned_cue is not None and whale.assigned_cue != actor_name:
            continue

        tgt_lat, tgt_lon, tgt_alt = whale.coord_observed

        keep_task, central_deg = quick_task_geo_filter(
            sat_lat_deg=sat_lat,
            sat_lon_deg=sat_lon,
            tgt_lat_deg=tgt_lat,
            tgt_lon_deg=tgt_lon,
            max_dlat_deg=20.0,
            max_dlon_deg=20.0,
            max_central_angle_deg=20.0
        )
        if not keep_task:
            continue

        will_be_visible, _ = eo_tools_dict[actor_name].will_be_visible_within(
            whale.coord_observed,
            r_vec,
            v_vec,
            t_datetime,
            avg_time_delay,
            el_min_deg=elevation_min,
            step=60.0
        )
        if not will_be_visible:
            continue

        offnadir_limit_local = float(satellite_specs[actor_name]["offnadir_limit"])

        target_eul_deg, offnadir_at_obs, offnadir_unbound, time_to_obs, pointing_vec_lvlh_target = eo_tools_dict[actor_name].compute_optimal_future_attitude(
            r_eci=r_vec,
            v_eci=v_vec,
            target_geodetic=whale.coord_observed,
            t_datetime=t_datetime,
            omega_max_rad=omega_max_rad,
            alpha_max_rad=alpha_max_rad,
            zeta=zeta,
            wn_rad=wn_rad,
            offnadir_max=offnadir_limit_local,
            offnadir_margin=offnadir_margin,
            dt_step_coarse=max(sim_step_seconds, 2.0),
            dt_step_fine=max(sim_step_seconds / 5.0, 0.25),
            dt_max=avg_time_delay,
            mode="per_axis"
        )

        if target_eul_deg is None or time_to_obs is None:
            continue

        candidate_tasks.append({
            "target_id": whale_idx,
            "coord": whale.coord_observed,
            "target_eul_deg": np.asarray(target_eul_deg, float),
            "offnadir_at_obs": float(offnadir_at_obs),
            "offnadir_unbound": float(offnadir_unbound),
            "time_to_obs": float(time_to_obs),
            "pointing_vec_lvlh_target": np.asarray(pointing_vec_lvlh_target, float),
            "central_angle_deg": float(central_deg if central_deg is not None else 999.0),
        })

    if not candidate_tasks:
        return None

    best_task = min(
        candidate_tasks,
        key=lambda task: (
            task["time_to_obs"],
            task["offnadir_at_obs"],
            task["central_angle_deg"]
        )
    )

    task_id = best_task["target_id"]
    all_targets[task_id].assigned_cue = actor_name

    eo_tools_dict[actor_name].current_task = best_task
    eo_tools_dict[actor_name].t_task_assigned = t_datetime
    eo_tools_dict[actor_name].t_to_obs_expected = float(best_task["time_to_obs"])
    eo_tools_dict[actor_name].pointing_vec_lvlh_target = best_task["pointing_vec_lvlh_target"]
    eo_tools_dict[actor_name].offnadir_unbound_target = best_task["offnadir_unbound"]
    eo_tools_dict[actor_name].time_to_obs_target = best_task["time_to_obs"]
    eo_tools_dict[actor_name].offnadir_at_obs_target = best_task["offnadir_at_obs"]
    eo_tools_dict[actor_name].move_set = True

    return best_task