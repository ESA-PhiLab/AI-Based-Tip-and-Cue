import numpy as np
import pykep as pk

from ..utils.constants import R_earth

from ..attitude.disturbance_calculations import (
    calculate_aero_torque,
    calculate_magnetic_torque,
    calculate_grav_torque,
)

from ..utils.reference_frame_transformation import (
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
# Shared helpers
# -----------------------------------------------------------------------------

def _axis_slew_dynamics(delta_rad, w_max, a_max, dt=None, w0=0.0, a0=0.0):
    """
    Compute trapezoidal/triangular slew profile for one axis,
    starting at angular velocity w0 and angular acceleration a0.

    Parameters
    ----------
    delta_rad : float
        Remaining rotation [rad].
    w_max : float
        Max angular velocity [rad/s].
    a_max : float
        Max angular acceleration [rad/s^2].
    dt : float or None
        If None → only return durations.
        If float → return trajectory arrays.
    w0 : float
        Initial angular velocity [rad/s].
    a0 : float
        Initial angular acceleration [rad/s^2].

    Returns
    -------
    If dt is None:
        (t_total, t_acc, t_const, t_dec)
    If dt given:
        (times, angles) trajectory arrays.
    """
    sign = np.sign(delta_rad) if delta_rad != 0 else 1.0
    d = abs(float(delta_rad))

    if d == 0.0 and w0 == 0.0:
        if dt is None:
            return 0.0, 0.0, 0.0, 0.0
        else:
            return np.array([0.0]), np.array([0.0])

    # Distance required to brake from current velocity
    d_stop = w0**2 / (2 * a_max) if a_max > 0 else np.inf

    # Already braking and within stop distance
    if d <= d_stop and a0 < 0:
        t_dec = w0 / a_max
        if dt is None:
            return t_dec, 0.0, 0.0, t_dec
        times = np.arange(0, t_dec + dt, dt)
        angles = w0 * times - 0.5 * a_max * times**2
        return times, sign * angles

    # Otherwise, plan a trapezoidal/triangular profile
    w_peak = min(w_max, np.sqrt(max(0.0, a_max * d + 0.5 * w0**2)))

    # Accelerate from w0 → w_peak
    t_acc = max(0.0, (w_peak - w0) / a_max)
    d_acc = w0 * t_acc + 0.5 * a_max * t_acc**2

    # Decelerate w_peak → 0
    t_dec = w_peak / a_max
    d_dec = w_peak**2 / (2 * a_max)

    if d_acc + d_dec >= d:
        # Triangular profile
        t_acc = (-w0 + np.sqrt(w0**2 + 2 * a_max * d)) / a_max
        t_dec = (w0 + a_max * t_acc) / a_max
        t_const = 0.0
        t_total = t_acc + t_dec
    else:
        # Trapezoidal profile
        d_const = d - (d_acc + d_dec)
        t_const = d_const / w_peak
        t_total = t_acc + t_const + t_dec

    if dt is None:
        return t_total, t_acc, t_const, t_dec

    # --- Build trajectory ---
    times = np.arange(0, t_total + dt, dt)
    angles = []
    for t in times:
        if t <= t_acc:
            theta = w0 * t + 0.5 * a_max * t**2
        elif t <= t_acc + t_const:
            tau = t - t_acc
            theta = d_acc + w_peak * tau
        else:
            tau = t - (t_acc + t_const)
            theta = d_acc + w_peak * t_const + w_peak * tau - 0.5 * a_max * tau**2
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
    """Shortest rotation angle between two attitudes via rotation matrices."""
    R_cur = RotMat_LVLH_to_BRF_by_eul(np.asarray(cur_eul_deg, float))
    R_tgt = RotMat_LVLH_to_BRF_by_eul(np.asarray(tgt_eul_deg, float))
    R_err = R_tgt @ R_cur.T
    tr = float(np.trace(R_err))
    tr = np.clip(tr, -1.0, 3.0)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def _aggregate_profiles(delta_rad, wmax, amax, dt=None, settle_seconds=0.0, mode="per_axis",
                        cur_eul_deg=None, tgt_eul_deg=None, w0=None, a0=None):
    """
    Aggregate axis or vector slew profiles.
    If dt=None → return t_slew (duration only).
    If dt>0   → return (times, angles_rad[N,3]).
    """
    if mode == "per_axis":
        if dt is None:
            t_axes = []
            for i, (d, w, a) in enumerate(zip(delta_rad, wmax, amax)):
                wi = 0.0 if w0 is None else float(w0[i])
                ai = 0.0 if a0 is None else float(a0[i])
                t_total, *_ = _axis_slew_dynamics(d, float(w), float(a), dt=None, w0=wi, a0=ai)
                t_axes.append(t_total)
            return max(t_axes)
        else:
            profiles = []
            for i, (d, w, a) in enumerate(zip(delta_rad, wmax, amax)):
                wi = 0.0 if w0 is None else float(w0[i])
                ai = 0.0 if a0 is None else float(a0[i])
                times_i, angles_i = _axis_slew_dynamics(d, float(w), float(a), dt=dt, w0=wi, a0=ai)
                profiles.append((times_i, angles_i))
            T_max = max(times[-1] for times, _ in profiles) + float(settle_seconds)
            times = np.arange(0.0, T_max + dt, dt)
            angles_all = np.zeros((len(times), 3))
            for i, (times_i, angles_i) in enumerate(profiles):
                angles_all[:, i] = np.interp(times, times_i, angles_i,
                                             left=0.0, right=float(delta_rad[i]))
            return times, angles_all

    elif mode == "vector":
        if (cur_eul_deg is None) or (tgt_eul_deg is None):
            raise ValueError("vector mode requires cur_eul_deg and tgt_eul_deg.")
        angle = _vector_error_angle(cur_eul_deg, tgt_eul_deg)
        wmax_s = float(np.max(np.atleast_1d(wmax)))
        amax_s = float(np.max(np.atleast_1d(amax)))
        w0_s = 0.0 if w0 is None else float(np.linalg.norm(w0))
        a0_s = 0.0 if a0 is None else float(np.linalg.norm(a0))

        if dt is None:
            t_slew, *_ = _axis_slew_dynamics(angle, wmax_s, amax_s, dt=None, w0=w0_s, a0=a0_s)
            return t_slew
        else:
            times_i, angles_i = _axis_slew_dynamics(angle, wmax_s, amax_s, dt=dt, w0=w0_s, a0=a0_s)
            T_max = times_i[-1] + float(settle_seconds)
            times = np.arange(0.0, T_max + dt, dt)
            frac = np.interp(times, times_i, angles_i, left=0.0, right=angle) / (angle if angle > 0 else 1.0)
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
        self._new_target_attitude_deg = actor_initial_attitude_deg

        # Command bookkeeping
        self.t_eul_commanded = None
        self.delay_slew_stab = None
        self.slew_stab_time_max = None

        # flag to track planned slews
        self.slew_active = False
        self._planned_start_eul = None  # Euler angles at the start of slew
        self._planned_start_time = None  # elapsed_seconds when slew started

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------
    def _update_initial_vectors(self):
        """Initialize derived vectors in IRF consistently from BRF state."""
        t = self._actor.local_time
        r = np.asarray(self._actor.get_position(t), float).reshape(3)
        v = np.asarray(self._actor.get_position_velocity(t)[1], float).reshape(3)
        eul = np.asarray(self._actor_attitude_deg, float).reshape(3)

        # Pointing vector is defined in BRF -> convert to IRF using current attitude
        self._actor_pointing_vector_eci = BRF2IRF_eul(self._actor_pointing_vector_body, r, v, eul)

        # Angular velocity is stored in BRF (consistent with rigid-body dynamics)
        self._actor_angular_velocity_eci = BRF2IRF_eul(self._actor_angular_velocity, r, v, eul)



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

    def _body_axes_in_irf(self):
        """Return BRF axes expressed in IRF (columns of BRF->IRF rotation)."""
        roll, pitch, yaw = self._actor_attitude_deg
        t = self._actor.local_time
        r = np.asarray(self._actor.get_position(t), float).reshape(3)
        v = np.asarray(self._actor.get_position_velocity(t)[1], float).reshape(3)
        eul = np.asarray([roll, pitch, yaw], float)

        x_irf = BRF2IRF_eul([1.0, 0.0, 0.0], r, v, eul)
        y_irf = BRF2IRF_eul([0.0, 1.0, 0.0], r, v, eul)
        z_irf = BRF2IRF_eul([0.0, 0.0, 1.0], r, v, eul)
        p_irf = BRF2IRF_eul(self._actor_pointing_vector_body, r, v, eul)
        return x_irf, y_irf, z_irf, p_irf


    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def update_attitude(self, dt):
        """Advance attitude by dt seconds under disturbance torques (rigid-body in BRF, attitude in IRF)."""
        t = self._actor.local_time
        r = np.asarray(self._actor.get_position(t), float).reshape(3)
        v = np.asarray(self._actor.get_position_velocity(t)[1], float).reshape(3)

        # Next orbit state (for Euler extraction relative to LVLH at t+dt)
        t_next = pk.epoch(t.mjd2000 + float(dt) * pk.SEC2DAY, "mjd2000")
        r_next = np.asarray(self._actor.get_position(t_next), float).reshape(3)
        v_next = np.asarray(self._actor.get_position_velocity(t)[1], float).reshape(3)  # keep consistent with your actor API

        # BRF axes in IRF at current time
        xb, yb, zb, pb = self._body_axes_in_irf()

        # --- Rigid-body dynamics in BRF ---
        self._calculate_angular_acceleration()
        self._actor_angular_velocity = self._actor_angular_velocity + self._actor_angular_acceleration * float(dt)  # BRF components

        # Convert incremental rotation (BRF components) to IRF rotation vector:
        # omega_irf = xb*wx + yb*wy + zb*wz
        w_brf = np.asarray(self._actor_angular_velocity, float).reshape(3)
        omega_irf = xb * w_brf[0] + yb * w_brf[1] + zb * w_brf[2]
        theta_body_irf = omega_irf * float(dt)  # axis*angle (small-angle integration)

        # Rotate body axes and pointing vector in IRF
        xb, yb, zb, pb = rotate_body_vectors(xb, yb, zb, pb, theta_body_irf)

        # Extract updated Euler angles for LVLH->BRF at t+dt
        self._actor_attitude_deg = get_rpy_angles_brf(xb, yb, zb, r_next, v_next)

        # Update derived vectors in IRF
        eul_next = np.asarray(self._actor_attitude_deg, float).reshape(3)
        self._actor_angular_velocity_eci = BRF2IRF_eul(self._actor_angular_velocity, r_next, v_next, eul_next)
        self._actor_pointing_vector_eci = pb

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
    def offnadir_from_euler(eul_deg, boresight_brf=(0.0, 0.0, 1.0)) -> tuple[float, np.ndarray]:
        """offnadir_from_euler(eul_deg,boresight_brf=(0,0,1)) -> tuple[float,np.ndarray]: Off-nadir [deg] between LVLH +Z and BRF boresight (returned in LVLH)."""
        eul = np.asarray(eul_deg, float).reshape(3)
        R_lvlh_to_brf = RotMat_LVLH_to_BRF_by_eul(eul)

        b_brf = np.asarray(boresight_brf, float).reshape(3)
        nb = float(np.linalg.norm(b_brf))
        if nb <= 0.0:
            raise ValueError("boresight_brf must be non-zero.")
        b_brf = b_brf / nb

        b_lvlh = R_lvlh_to_brf.T @ b_brf  # BRF->LVLH
        nl = float(np.linalg.norm(b_lvlh))
        if nl <= 0.0:
            raise ValueError("boresight_brf produced near-zero b_lvlh.")
        b_lvlh = b_lvlh / nl

        nadir_lvlh = np.array([0.0, 0.0, 1.0], float)  # your convention: LVLH +Z = nadir
        dot = float(np.clip(np.dot(nadir_lvlh, b_lvlh), -1.0, 1.0))
        off_deg = float(np.degrees(np.arccos(dot)))
        return off_deg, b_lvlh

    @staticmethod
    def pointing_attitude_lvlh(target_vec_lvlh, boresight_brf=(0.0, 0.0, 1.0), up_lvlh=(1.0, 0.0, 0.0), up_brf=(1.0, 0.0, 0.0)) -> list[float]:
        """pointing_attitude_lvlh(target_vec_lvlh,boresight_brf=(0,0,1),up_lvlh=(1,0,0),up_brf=(1,0,0)) -> list[float]: Euler [deg] for LVLH->BRF so that R@target_vec_lvlh=boresight_brf with a defined roll via up vectors."""
        import numpy as np

        def _unit(x):
            x = np.asarray(x, float).reshape(3)
            n = float(np.linalg.norm(x))
            if n <= 0.0:
                raise ValueError("Vector must be non-zero.")
            return x / n

        def _orthonormal_basis(z_hat, up_hint):
            z_hat = _unit(z_hat)
            up_hint = _unit(up_hint)
            x_hat = np.cross(up_hint, z_hat)
            nx = float(np.linalg.norm(x_hat))
            if nx < 1e-9:
                # up_hint parallel to z_hat -> pick another
                alt = np.array([0.0, 1.0, 0.0], float) if abs(z_hat[0]) > 0.9 else np.array([1.0, 0.0, 0.0], float)
                x_hat = np.cross(alt, z_hat)
                nx = float(np.linalg.norm(x_hat))
                if nx < 1e-9:
                    raise ValueError("Failed to build basis.")
            x_hat /= nx
            y_hat = np.cross(z_hat, x_hat)
            y_hat /= float(np.linalg.norm(y_hat))
            return np.column_stack((x_hat, y_hat, z_hat))  # 3x3

        t_lvlh = _unit(target_vec_lvlh)
        b_brf = _unit(boresight_brf)

        B_lvlh = _orthonormal_basis(t_lvlh, up_lvlh)  # columns = LVLH basis vectors, with z=t
        B_brf = _orthonormal_basis(b_brf, up_brf)  # columns = BRF basis vectors, with z=b

        # Map LVLH coords -> BRF coords: R * (basis in LVLH) = (basis in BRF)
        R = B_brf @ B_lvlh.T

        roll, pitch, yaw = rotation_matrix_to_ypr(R)
        angles = np.array([roll, pitch, yaw], float)
        angles = (angles + 180.0) % 360.0 - 180.0
        return angles.tolist()

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
                                        mode="per_axis",
                                        current_w_rad=None,
                                        current_a_rad=None):
        """Compute slew + stabilization time between two Euler attitudes."""
        if current_eul is None:
            current_eul = self._actor_attitude_deg
        if target_eul is None:
            target_eul = self._target_attitude_deg

        cur, tgt, delta_deg, delta_rad = _compute_delta(current_eul, target_eul)
        wmax = np.broadcast_to(np.asarray(omega_max_rad, float), 3)
        amax = np.broadcast_to(np.asarray(alpha_max_rad, float), 3)

        t_slew = _aggregate_profiles(
            delta_rad, wmax, amax, dt=None, mode=mode,
            cur_eul_deg=cur, tgt_eul_deg=tgt, w0=current_w_rad, a0=current_a_rad
        )

        delta_angle_deg = np.linalg.norm(delta_deg)


        if settle_seconds is not None:
            t_settle = float(settle_seconds)
        elif (zeta is not None) and (wn_rad is not None) and (zeta > 0) and (wn_rad > 0):
            base_settle = 4.0 / (zeta * wn_rad)
            # proportional to commanded angle (e.g. 45° move → full base_settle)
            t_settle = base_settle * (delta_angle_deg / 45.0)
            t_settle = min(t_settle, base_settle)
        else:
            t_settle = max(0.5, 0.1 * delta_angle_deg)

        t_total = t_slew + t_settle
        return t_total, t_slew, t_settle

    def generate_euler_trajectory(self,
                                  current_eul=None,
                                  target_eul=None,
                                  dt=0.1,
                                  omega_max_rad=0.05,
                                  alpha_max_rad=0.01,
                                  settle_seconds=0.0,
                                  mode="per_axis",
                                  current_w_rad=None,
                                  current_a_rad=None):
        """Generate Euler trajectory (deg) for slew + optional stabilization hold."""
        if current_eul is None:
            current_eul = self._actor_attitude_deg
        if target_eul is None:
            target_eul = self._target_attitude_deg

        cur, tgt, delta_deg, delta_rad = _compute_delta(current_eul, target_eul)

        wmax = np.broadcast_to(np.asarray(omega_max_rad, float), 3)
        amax = np.broadcast_to(np.asarray(alpha_max_rad, float), 3)

        # --- Slew part ---
        times, angles_rad = _aggregate_profiles(
            delta_rad, wmax, amax, dt=float(dt),
            settle_seconds=0.0,  # only slew here
            mode=mode, cur_eul_deg=cur, tgt_eul_deg=tgt,
            w0=current_w_rad, a0=current_a_rad
        )
        eul_traj = np.rad2deg(angles_rad) + cur

        # --- Stabilization part (constant attitude) ---
        if settle_seconds > 0.0:
            t_end = times[-1]
            extra_times = np.arange(dt, settle_seconds + dt / 2, dt) + t_end
            eul_final = eul_traj[-1]
            extra_eul = np.tile(eul_final, (len(extra_times), 1))

            times = np.concatenate([times, extra_times])
            eul_traj = np.vstack([eul_traj, extra_eul])

        return eul_traj, times

    def plan_slew(self,
                  start_eul_deg,
                  target_eul_deg,
                  omega_max_rad,
                  alpha_max_rad,
                  zeta=0.8,
                  wn_rad=0.42,
                  dt=1.0,
                  mode="per_axis",
                  t_start=0.0,
                  w_stab_res=None,
                  a_stab_res=None):
        """
        Plan a slew + stabilization trajectory.
        Stores [(abs_time, eul_deg, vel_rad, acc_rad), ...].
        """
        # Normalize residuals: allow scalar or vector
        if w_stab_res is not None:
            w_stab_res = np.broadcast_to(np.atleast_1d(w_stab_res), (3,))
        if a_stab_res is not None:
            a_stab_res = np.broadcast_to(np.atleast_1d(a_stab_res), (3,))

        # Compute slew + stabilization durations
        delay_slew_stab, delay_slew, delay_stab = self.get_pointing_stabilization_time(
            current_eul=start_eul_deg,
            target_eul=target_eul_deg,
            omega_max_rad=omega_max_rad,
            alpha_max_rad=alpha_max_rad,
            zeta=zeta,
            wn_rad=wn_rad,
            mode=mode,
            current_w_rad=[0.0, 0.0, 0.0],
            current_a_rad=[0.0, 0.0, 0.0]
        )
        self.delay_slew_stab = delay_slew_stab
        self._planned_start_time = t_start

        # Generate trajectory (includes stabilization hold)
        euler_traj, times = self.generate_euler_trajectory(
            current_eul=start_eul_deg,
            target_eul=target_eul_deg,
            dt=dt,
            omega_max_rad=omega_max_rad,
            alpha_max_rad=alpha_max_rad,
            settle_seconds=delay_stab,
            mode=mode,
            current_w_rad=None,
            current_a_rad=None
        )

        # Compute velocities and accelerations
        euler_rad = np.deg2rad(euler_traj)
        vel = np.gradient(euler_rad, times, axis=0)
        acc = np.gradient(vel, times, axis=0)

        # Overwrite stabilization portion with given residuals
        if (w_stab_res is not None) and (a_stab_res is not None):
            mask = times >= delay_slew
            vel[mask] = w_stab_res
            acc[mask] = a_stab_res

        # Ensure last point is rest (override)
        vel[-1] = np.zeros(3)
        acc[-1] = np.zeros(3)

        # Store final trajectory
        self._planned_traj = [
            (t_start + t_rel, eul, v, a)
            for t_rel, eul, v, a in zip(times, euler_traj, vel, acc)
        ]

        self.slew_active = True
        self.set_target_euler(target_eul_deg)

    def follow_planned_slew(self, elapsed_seconds):
        """
        Follow the planned slew trajectory at the given simulation time.
        Updates Euler angles [deg], angular velocity [rad/s],
        and angular acceleration [rad/s²].
        """

        if not self.slew_active or not self._planned_traj:
            return

        traj_times = [t for t, *_ in self._planned_traj]

        # If finished, snap to target
        if elapsed_seconds >= traj_times[-1]:
            self.set_actor_euler(self._target_attitude_deg)
            self._actor_angular_velocity = np.zeros(3)
            self._actor_angular_acceleration = np.zeros(3)
            self.slew_active = False

            return

        # Otherwise, interpolate between two trajectory points
        for i in range(len(traj_times) - 1):
            t0, eul0, v0, a0 = self._planned_traj[i]
            t1, eul1, v1, a1 = self._planned_traj[i + 1]
            if t0 <= elapsed_seconds < t1:
                frac = (elapsed_seconds - t0) / (t1 - t0)

                # Linear interpolation
                eul = (1 - frac) * eul0 + frac * eul1  # deg
                vel = (1 - frac) * v0 + frac * v1  # rad/s
                acc = (1 - frac) * a0 + frac * a1  # rad/s²

                self.set_actor_euler(eul)
                self._actor_angular_velocity = vel
                self._actor_angular_acceleration = acc
                break

    def _extra_delay_pause_equivalent(self,
                                      eul_current_target,
                                      eul_new_target,
                                      omega_max_rad,
                                      alpha_max_rad,
                                      zeta,
                                      wn_rad,
                                      mode="per_axis"):
        """
        Computes the SAME extra delay term you already use in 'pause' mode:
          delay_extra = delta_eul / deg(omega_max_rad) + delay_stab_extra
        where delay_stab_extra is computed via get_pointing_stabilization_time
        for (current_target -> new_target).

        This helper does NOT multiply by sim_step_seconds. Keep that multiplication
        outside (exactly as your original pause code does).
        """
        eul_current_target = np.asarray(eul_current_target, float)
        eul_new_target = np.asarray(eul_new_target, float)

        # Angular delta in degrees (same as your original)
        delta_eul = np.linalg.norm(eul_new_target - eul_current_target)

        # Stabilization component for the additional delta (same call you use)
        _, _, delay_stab_extra = self.get_pointing_stabilization_time(
            current_eul=eul_current_target,
            target_eul=eul_new_target,
            omega_max_rad=omega_max_rad,
            alpha_max_rad=alpha_max_rad,
            zeta=zeta,
            wn_rad=wn_rad,
            mode=mode
        )

        # Same extra delay formula you use in 'pause'
        delay_extra = delta_eul / np.rad2deg(omega_max_rad) + delay_stab_extra
        return delay_extra


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
