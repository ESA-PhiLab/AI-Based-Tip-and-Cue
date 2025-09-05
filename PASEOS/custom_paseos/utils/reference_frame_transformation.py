"""
Attitude/geometry transforms between IRF (ECI), LVLH, and BRF.

Frame conventions (planet-observation LVLH):
- LVLH z-axis: points toward planet center (nadir)  => ẑ = - r̂
- LVLH y-axis: opposite orbital angular momentum     => ŷ = - ĥ,  where h = r × v
- LVLH x-axis: velocity projected onto the local horizon
               (i.e., the along-track direction)     => x̂ = normalize( v - (v·ẑ) ẑ )

Euler angles:
- Always handled as (roll, pitch, yaw).
- Convention: intrinsic ZYX rotation (first roll about x, then pitch about y, then yaw about z).
- Composition: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

import numpy as np
# ------------------------- helpers -------------------------

def _normalize(vec, eps=1e-12):
    v = np.asarray(vec, dtype=float).flatten()
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero (or near-zero) magnitude encountered during normalization.")
    return v / n

def _asvec3(a):
    v = np.asarray(a, dtype=float).flatten()
    if v.size != 3:
        raise ValueError("Expected a 3-element vector.")
    return v

# --------------------- core rotation matrices ---------------------

def RotMat_IRF_to_LVLH(r, v):
    """Rotation matrix from IRF (ECI) to LVLH."""
    r = _asvec3(r)
    v = _asvec3(v)

    z_dir = -_normalize(r)
    h = np.cross(r, v)
    y_dir = -_normalize(h)

    v_proj = v - np.dot(v, z_dir) * z_dir
    try:
        x_dir = _normalize(v_proj)
    except ValueError:
        x_dir = _normalize(np.cross(y_dir, z_dir))

    y_dir = _normalize(np.cross(z_dir, x_dir))
    T = np.vstack((x_dir, y_dir, z_dir))
    return T

def RotMat_LVLH_to_BRF_by_eul(eul_ang):
    """
    Rotation matrix LVLH -> BRF using intrinsic ZYX (roll, pitch, yaw).
    Input order: (roll_deg, pitch_deg, yaw_deg).
    """
    roll_deg, pitch_deg, yaw_deg = eul_ang
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0.0, np.sin(pitch)],
        [ 0.0,           1.0, 0.0          ],
        [-np.sin(pitch), 0.0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0]
    ])

    return Rz @ Ry @ Rx  # intrinsic ZYX

def RotMat_by_quat(q):
    """Rotation matrix IRF -> BRF from quaternion [qx,qy,qz,qw] (scalar last)."""
    q = np.asarray(q, dtype=float).flatten()
    if q.size != 4:
        raise ValueError("Quaternion must have 4 elements [qx, qy, qz, qw].")
    qx, qy, qz, qw = q
    n2 = qx*qx + qy*qy + qz*qz + qw*qw
    if n2 < 1e-16:
        raise ValueError("Zero-norm quaternion.")
    s = 2.0 / n2

    xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
    xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s

    T = np.array([
        [1.0 - (yy + zz),     xy - wz,            xz + wy],
        [xy + wz,             1.0 - (xx + zz),    yz - wx],
        [xz - wy,             yz + wx,            1.0 - (xx + yy)]
    ])
    return T

# --------------------- vector transforms ---------------------

def IRF2LVLH(u, r, v):
    return RotMat_IRF_to_LVLH(r, v) @ _asvec3(u)

def LVLH2IRF(u, r, v):
    return RotMat_IRF_to_LVLH(r, v).T @ _asvec3(u)

def LVLH2BRF_eul(u, eul_ang):
    return RotMat_LVLH_to_BRF_by_eul(eul_ang) @ _asvec3(u)

def BRF2LVLH_eul(u, eul_ang):
    return RotMat_LVLH_to_BRF_by_eul(eul_ang).T @ _asvec3(u)

def IRF2BRF_quat(u, q):
    return RotMat_by_quat(q) @ _asvec3(u)

def BRF2IRF_quat(u, q):
    return RotMat_by_quat(q).T @ _asvec3(u)

def RotMat_IRF_to_BRF(r, v, eul_ang):
    return RotMat_LVLH_to_BRF_by_eul(eul_ang) @ RotMat_IRF_to_LVLH(r, v)

def IRF2BRF_eul(u, r, v, eul_ang):
    return RotMat_IRF_to_BRF(r, v, eul_ang) @ _asvec3(u)

def BRF2IRF_eul(u, r, v, eul_ang):
    return RotMat_IRF_to_BRF(r, v, eul_ang).T @ _asvec3(u)

# --------------------- Euler extraction ---------------------

def rotation_matrix_to_ypr(R):
    """
    Extract (roll, pitch, yaw) from R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Returns radians.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("Input must be 3x3")

    # pitch from -asin(r20)
    r20 = R[2, 0]

    if np.isclose(r20, -1.0, atol=1e-12):
        pitch = np.pi / 2
        roll = np.arctan2(R[0, 1], R[0, 2])
        yaw = 0.0
        return roll, pitch, yaw

    if np.isclose(r20, 1.0, atol=1e-12):
        pitch = -np.pi / 2
        roll = np.arctan2(-R[0, 1], -R[0, 2])
        yaw = 0.0
        return roll, pitch, yaw

    pitch = np.arcsin(-r20)
    cp = np.cos(pitch)

    roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
    yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    return roll, pitch, yaw



