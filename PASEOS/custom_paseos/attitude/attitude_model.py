# attitude_model.py
# -----------------------------------------------------------------------------
# Attitude dynamics & helpers. Owns Euler angles and pointing math (degrees).
# -----------------------------------------------------------------------------

import numpy as np
import pykep as pk
from custom_paseos.utils.constants import R_earth

from custom_paseos.attitude.disturbance_calculations import (
    calculate_aero_torque,
    calculate_magnetic_torque,
    calculate_grav_torque,
)
from custom_paseos.utils.reference_frame_transformation import (
    IRF2LVLH,
    LVLH2IRF,
    IRF2BRF_eul,
    BRF2IRF_eul,
    RotMat_LVLH_to_BRF_by_eul,
    RotMat_by_quat,
    rotation_matrix_to_ypr,
    rodrigues_rotation,
    get_rpy_angles_brf,
    rotate_body_vectors,
)

# -----------------------------------------------------------------------------
# Shared helpers (no duplication)
# -----------------------------------------------------------------------------

def _axis_slew_dynamics(delta_rad, w_max, a_max, dt=None):
    """
    Compute trapezoidal/triangular slew profile for one axis.

    Parameters
    ----------
    delta_rad : float
        Total rotation [rad].
    w_max : float
        Max angular velocity [rad/s].
    a_max : float
        Max angular acceleration [rad/s^2].
    dt : float or None
        If None -> only return durations (t_total, t_acc, t_const).
        If float -> return full profile (times [s], angles [rad]).

    Returns
    -------
    If dt is None:
        (t_total, t_acc, t_const)
    If dt given:
        (times, angles) trajectory in radians.
    """
    d = abs(float(delta_rad))  # ensure scalar float
    sign = np.sign(float(delta_rad))

    if d == 0.0:
        if dt is None:
            return 0.0, 0.0, 0.0
        else:
            return np.array([0.0]), np.array([0.0])

    # triangular
    if d <= (w_max**2) / a_max:
        t_acc = np.sqrt(d / a_max)
        t_total = 2.0 * t_acc
        t_const = 0.0

        if dt is None:
            return t_total, t_acc, t_const

        times = np.arange(0.0, t_total + dt, dt)
        angles = []
        for t in times:
            if t <= t_acc:
                theta = 0.5 * a_max * t**2
            else:
                tau = t - t_acc
                theta = 0.5 * a_max * t_acc**2 + (w_max * tau - 0.5 * a_max * tau**2)
            angles.append(theta)
        return times, sign * np.array(angles)

    # trapezoidal
    t_acc = w_max / a_max
    d_accdec = (w_max**2) / a_max
    t_const = max(0.0, (d - d_accdec) / w_max)
    t_total = 2.0 * t_acc + t_const

    if dt is None:
        return t_total, t_acc, t_const

    times = np.arange(0.0, t_total + dt, dt)
    angles = []
    for t in times:
        if t <= t_acc:
            theta = 0.5 * a_max * t**2
        elif t <= t_acc + t_const:
            theta = 0.5 * a_max * t_acc**2 + w_max * (t - t_acc)
        else:
            tau = t - (t_acc + t_const)
            theta = (0.5 * a_max * t_acc**2 + w_max * t_const +
                     w_max * tau - 0.5 * a_max * tau**2)
        angles.append(theta)
    return times, sign * np.array(angles)


def _wrap180(a_deg):
    return (a_deg + 180.0) % 360.0 - 180.0


def _compute_delta(current_eul, target_eul):
    """Return delta in deg and rad (wrapped to [-180,180])."""
    cur = np.asarray(current_eul, float)
    tgt = np.asarray(target_eul, float)
    delta_deg = _wrap180(tgt - cur)
    delta_rad = np.deg2rad(delta_deg)
    return cur, tgt, delta_deg, delta_rad


def _vector_error_angle(cur_eul_deg, tgt_eul_deg):
    """
    Compute the shortest rotation angle between two attitudes using rotation matrices,
    matching the original behavior in your 'working' method.
    """
    R_cur = RotMat_LVLH_to_BRF_by_eul(np.asarray(cur_eul_deg, float))
    R_tgt = RotMat_LVLH_to_BRF_by_eul(np.asarray(tgt_eul_deg, float))
    R_err = R_tgt @ R_cur.T
    tr = float(np.trace(R_err))
    tr = np.clip(tr, -1.0, 3.0)  # numerical safety
    angle = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    return angle


def _aggregate_profiles(delta_rad, wmax, amax, dt=None, settle_seconds=0.0, mode="per_axis",
                        cur_eul_deg=None, tgt_eul_deg=None):
    """
    Aggregate axis or vector slew profiles.
    If dt=None → return t_slew (duration only).
    If dt>0   → return (times, angles_rad[N,3]).
    """
    if mode == "per_axis":
        if dt is None:
            t_axes = []
            for d, w, a in zip(delta_rad, wmax, amax):
                t_total, _, _ = _axis_slew_dynamics(d, float(w), float(a), dt=None)
                t_axes.append(t_total)
            t_slew = max(t_axes)
            return t_slew
        else:
            profiles = []
            for d, w, a in zip(delta_rad, wmax, amax):
                times_i, angles_i = _axis_slew_dynamics(d, float(w), float(a), dt=dt)
                profiles.append((times_i, angles_i))
            T_max = max(times[-1] for times, _ in profiles) + float(settle_seconds)
            times = np.arange(0.0, T_max + dt, dt)
            angles_all = np.zeros((len(times), 3))
            for i, (times_i, angles_i) in enumerate(profiles):
                angles_all[:, i] = np.interp(times, times_i, angles_i,
                                             left=0.0, right=float(delta_rad[i]))
            return times, angles_all

    elif mode == "vector":
        # Compute true shortest rotation angle (same as the old method)
        if (cur_eul_deg is None) or (tgt_eul_deg is None):
            raise ValueError("vector mode requires cur_eul_deg and tgt_eul_deg.")
        angle = _vector_error_angle(cur_eul_deg, tgt_eul_deg)
        wmax_s = float(np.max(np.atleast_1d(wmax)))
        amax_s = float(np.max(np.atleast_1d(amax)))

        if dt is None:
            t_slew, _, _ = _axis_slew_dynamics(angle, wmax_s, amax_s, dt=None)
            return t_slew
        else:
            times_i, angles_i = _axis_slew_dynamics(angle, wmax_s, amax_s, dt=dt)
            T_max = times_i[-1] + float(settle_seconds)
            times = np.arange(0.0, T_max + dt, dt)
            frac = np.interp(times, times_i, angles_i, left=0.0, right=angle) / (angle if angle > 0 else 1.0)
            # Distribute along the rotation vector in Euler-delta space (approximate)
            # If you want exact interpolation, replace with quaternion SLERP and back to Euler.
            dir_vec = np.asarray(delta_rad, float)
            if np.allclose(dir_vec, 0.0):
                angles_all = np.zeros((len(times), 3))
            else:
                angles_all = frac[:, None] * dir_vec[None, :]
            return times, angles_all

    else:
        raise ValueError("mode must be 'per_axis' or 'vector'")


# -----------------------------------------------------------------------------
# AttitudeModel
# -----------------------------------------------------------------------------

class AttitudeModel:
    """
    Simulates roll/pitch/yaw attitude evolution with simple disturbance torques.
    Euler angles are stored and returned in degrees.
    """

    def __init__(
        self,
        local_actor,
        actor_initial_attitude_deg=(0.0, 0.0, 0.0),
        actor_initial_angular_velocity=(0.0, 0.0, 0.0),
        actor_pointing_vector_body=(0.0, 0.0, 1.0),
        actor_residual_magnetic_field=(0.0, 0.0, 0.0),
    ):
        self._actor = local_actor

        # State in degrees
        self._actor_attitude_deg = np.array(actor_initial_attitude_deg, float)
        self._actor_angular_velocity = np.array(actor_initial_angular_velocity, float)
        self._actor_angular_acceleration = np.zeros(3)

        # Pointing vector (body frame, normalized)
        self._actor_pointing_vector_body = np.array(actor_pointing_vector_body, float)
        self._actor_pointing_vector_body /= np.linalg.norm(self._actor_pointing_vector_body)

        # Initial transforms
        self._actor_pointing_vector_eci = None
        self._actor_angular_velocity_eci = None
        self._update_initial_vectors()

        # Disturbance model
        self._actor_residual_magnetic_field = np.array(actor_residual_magnetic_field, float)

        # Target attitude (deg)
        self._target_attitude_deg = None

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------
    def _update_initial_vectors(self):
        t = self._actor.local_time
        r = np.array(self._actor.get_position(t))
        v = np.array(self._actor.get_position_velocity(t)[1])

        self._actor_pointing_vector_eci = LVLH2IRF(
            self._actor_pointing_vector_body, r, v
        )
        self._actor_angular_velocity_eci = LVLH2IRF(
            self._actor_angular_velocity, r, v
        )

    # -------------------------------------------------------------------------
    # Disturbances
    # -------------------------------------------------------------------------
    def _calculate_disturbance_torque(self):
        T = np.zeros(3)
        if self._actor.has_attitude_disturbances:
            if "aerodynamic" in self._actor.get_disturbances():
                T += calculate_aero_torque()
            if "gravitational" in self._actor.get_disturbances():
                T += calculate_grav_torque()
            if "magnetic" in self._actor.get_disturbances():
                time = self._actor.local_time
                T += calculate_magnetic_torque(
                    m_earth=self._actor.central_body.magnetic_dipole_moment(time),
                    m_sat=self._actor_residual_magnetic_field,
                    position=self._actor.get_position(time),
                    velocity=self._actor.get_position_velocity(time)[1],
                    eul_deg=self._actor_attitude_deg,
                )
        return T

    def _calculate_angular_acceleration(self):
        I = self._actor._moment_of_inertia()
        self._actor_angular_acceleration = np.linalg.inv(I) @ (
            self._calculate_disturbance_torque()
            - np.cross(self._actor_angular_velocity, I @ self._actor_angular_velocity)
        )

    # -------------------------------------------------------------------------
    # Frame & body dynamics
    # -------------------------------------------------------------------------
    def _body_rotation(self, dt):
        self._calculate_angular_acceleration()
        self._actor_angular_velocity += self._actor_angular_acceleration * dt
        return self._actor_angular_velocity * dt  # rad

    @staticmethod
    def _frame_rotation(r, r_next, v):
        h = np.cross(r, v)
        h /= np.linalg.norm(h)
        angle = np.arccos(np.clip(np.dot(r, r_next) / (np.linalg.norm(r) * np.linalg.norm(r_next)), -1.0, 1.0))
        return -IRF2LVLH(h * angle, r, v)

    def _body_axes_in_lvlh(self):
        roll, pitch, yaw = self._actor_attitude_deg
        t = self._actor.local_time
        r = np.array(self._actor.get_position(t))
        v = np.array(self._actor.get_position_velocity(t)[1])

        x = BRF2IRF_eul([1, 0, 0], r, v, [roll, pitch, yaw])
        y = BRF2IRF_eul([0, 1, 0], r, v, [roll, pitch, yaw])
        z = BRF2IRF_eul([0, 0, 1], r, v, [roll, pitch, yaw])
        p = BRF2IRF_eul(self._actor_pointing_vector_body, r, v, [roll, pitch, yaw])
        return x, y, z, p

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def update_attitude(self, dt):
        """Advance the attitude by dt seconds under disturbance torques."""
        t = self._actor.local_time
        r = np.array(self._actor.get_position(t))
        r_next = np.array(self._actor.get_position(pk.epoch(t.mjd2000 + dt * pk.SEC2DAY, "mjd2000")))
        v = np.array(self._actor.get_position_velocity(t)[1])

        xb, yb, zb, pb = self._body_axes_in_lvlh()

        theta_frame = self._frame_rotation(r, r_next, v)
        theta_body = self._body_rotation(dt)

        xb, yb, zb, pb = rotate_body_vectors(xb, yb, zb, pb, theta_frame)
        theta_body = rodrigues_rotation(theta_body, theta_frame)
        xb, yb, zb, pb = rotate_body_vectors(xb, yb, zb, pb, theta_body)

        # Extract updated Euler angles (deg)
        self._actor_attitude_deg = get_rpy_angles_brf(xb, yb, zb, r_next, v)

        self._actor_angular_velocity_eci = LVLH2IRF(self._actor_angular_velocity, r_next, v)
        self._actor_pointing_vector_eci = LVLH2IRF(pb, r_next, v)

    def set_target_euler(self, target_eul_deg):
        """Set desired Euler angles [deg]."""
        self._target_attitude_deg = np.array(target_eul_deg, float)

    def set_actor_euler(self, eul_deg):
        """Set desired Euler angles [deg]."""
        self._actor_attitude_deg = np.array(eul_deg, float)

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------
    @staticmethod
    def offnadir_from_euler(eul_deg):
        """Compute off-nadir angle [deg] given Euler angles [deg]."""
        boresight_ref = np.array([0.0, 0.0, 1.0])
        R = RotMat_LVLH_to_BRF_by_eul(eul_deg)
        boresight_dir = R @ boresight_ref
        boresight_dir /= np.linalg.norm(boresight_dir)
        dot = np.clip(np.dot(boresight_ref, boresight_dir), -1.0, 1.0)
        return np.degrees(np.arccos(dot)), boresight_dir

    @staticmethod
    def pointing_attitude_brf(pointing_vec_brf_target):
        """Euler [deg] to rotate LVLH boresight [0,0,1] onto target vec in BRF."""
        eul_ref = [0.0, 0.0, 0.0]
        l1 = np.array([0.0, 0.0, 1.0])
        l2 = np.asarray(pointing_vec_brf_target, float).reshape(3)
        l1 /= np.linalg.norm(l1)
        l2 /= np.linalg.norm(l2)

        dot = np.clip(np.dot(l1, l2), -1.0, 1.0)
        cross = np.cross(l2, l1)

        q_vec = cross
        q_scalar = (1 + dot)
        quat = np.concatenate((q_vec, [q_scalar])) / np.sqrt(2 * (1 + dot))

        R_align = RotMat_by_quat(quat)
        R_total = RotMat_LVLH_to_BRF_by_eul(eul_ref) @ R_align

        roll, pitch, yaw = rotation_matrix_to_ypr(R_total)
        angles = np.array([roll, pitch, yaw], float)
        angles = (angles + 180.0) % 360.0 - 180.0
        return angles.tolist()

    def get_pointing_stabilization_time(self,
                                        current_eul=None,
                                        target_eul=None,
                                        omega_max_rad=0.05,
                                        alpha_max_rad=0.01,
                                        settle_seconds=None,
                                        zeta=None,
                                        wn_rad=None,
                                        mode="per_axis"):
        """
        Compute slew + stabilization time between two Euler attitudes.

        Returns
        -------
        (t_total, t_slew, t_settle) in seconds.
        """
        if current_eul is None:
            current_eul = self._actor_attitude_deg
        if target_eul is None:
            target_eul = self._target_attitude_deg

        cur, tgt, delta_deg, delta_rad = _compute_delta(current_eul, target_eul)
        wmax = np.broadcast_to(np.asarray(omega_max_rad, float), 3)
        amax = np.broadcast_to(np.asarray(alpha_max_rad, float), 3)

        t_slew = _aggregate_profiles(
            delta_rad, wmax, amax, dt=None, mode=mode,
            cur_eul_deg=cur, tgt_eul_deg=tgt
        )

        # Settling time (use explicit None checks to match working version)
        if settle_seconds is not None:
            t_settle = float(settle_seconds)
        elif (zeta is not None) and (wn_rad is not None) and (zeta > 0) and (wn_rad > 0):
            t_settle = 4.0 / (zeta * wn_rad)  # 2% settling time
        else:
            t_settle = 5.0  # default buffer

        t_total = t_slew + t_settle
        return t_total, t_slew, t_settle

    def generate_euler_trajectory(self,
                                  current_eul=None,
                                  target_eul=None,
                                  dt=0.1,
                                  omega_max_rad=0.05,
                                  alpha_max_rad=0.01,
                                  settle_seconds=5.0,
                                  mode="per_axis"):
        """
        Generate Euler angle trajectory (deg) for a slew maneuver.

        Returns
        -------
        times : np.ndarray [s]
        eul_traj : np.ndarray [N,3] in deg
        """
        if current_eul is None:
            current_eul = self._actor_attitude_deg
        if target_eul is None:
            target_eul = self._target_attitude_deg

        cur, tgt, delta_deg, delta_rad = _compute_delta(current_eul, target_eul)
        wmax = np.broadcast_to(np.asarray(omega_max_rad, float), 3)
        amax = np.broadcast_to(np.asarray(alpha_max_rad, float), 3)

        times, angles_rad = _aggregate_profiles(
            delta_rad, wmax, amax, dt=float(dt),
            settle_seconds=float(settle_seconds),
            mode=mode, cur_eul_deg=cur, tgt_eul_deg=tgt
        )
        eul_traj = np.rad2deg(angles_rad) + cur  # add offset in deg
        return times, eul_traj


# -----------------------------------------------------------------------------
# Controller estimator (unchanged behavior)
# -----------------------------------------------------------------------------

def estimate_wv3_like_controller(omega_rad_max=np.deg2rad(3.86),  # WV-2 datasheet rate
                                 alpha_rad_max=np.deg2rad(1.43),  # WV-2 datasheet accel
                                 ground_sep_km=200.0,  # requirement: 200 km retarget
                                 orbit_altitude_km=617.0,  # WV-3 altitude class
                                 total_time_req_s=12.0,  # “time to slew 200 km: 12 s”
                                 zeta_guess=0.8,  # well-damped
                                 mode="vector"):
    """
    Estimate (zeta, wn_rad) so that total_time ~= 12 s for a 200 km retarget at ~617 km altitude.
    """
    R_earth_km = R_earth
    theta_c = ground_sep_km / R_earth_km
    delta_theta = (R_earth_km / (R_earth_km + orbit_altitude_km)) * theta_c  # [rad]

    def _axis_slew_time(delta_rad, w_max, a_max):
        d = abs(float(delta_rad))
        if d == 0.0:
            return 0.0
        if d <= (w_max ** 2) / a_max:
            return 2.0 * np.sqrt(d / a_max)
        t_acc = w_max / a_max
        d_accdec = (w_max ** 2) / a_max
        t_const = max(0.0, (d - d_accdec) / w_max)
        return 2.0 * t_acc + t_const

    if mode == "vector":
        t_slew = _axis_slew_time(delta_theta, omega_rad_max, alpha_rad_max)
    elif mode == "per_axis":
        t_axes = [
            _axis_slew_time(delta_theta, omega_rad_max, alpha_rad_max),
            _axis_slew_time(delta_theta, omega_rad_max, alpha_rad_max),
            _axis_slew_time(delta_theta, omega_rad_max, alpha_rad_max),
        ]
        t_slew = float(np.max(t_axes))
    else:
        raise ValueError("mode must be 'vector' or 'per_axis'.")

    t_settle = max(0.0, total_time_req_s - t_slew)
    wn_rad = 4.0 / (zeta_guess * t_settle) if (zeta_guess > 0 and t_settle > 0) else np.inf
    return zeta_guess, wn_rad


if __name__ == "__main__":
    zeta, wn_rad = estimate_wv3_like_controller()
    print(f"zeta={zeta}, wn_rad={wn_rad}")
