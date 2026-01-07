"""
Attitude/geometry transforms between IRF (ECI), LVLH, and BRF.

Frame conventions (planet-observation LVLH):
- LVLH z-axis: nadir pointing (−r̂)
- LVLH y-axis: opposite orbital angular momentum (−ĥ, h = r × v)
- LVLH x-axis: completes right-handed system (along-track)
Euler angles:
- Convention: intrinsic ZYX (roll about x, pitch about y, yaw about z).
- Order: (roll, pitch, yaw), always in degrees.
"""

import numpy as np

# ------------------------- helpers -------------------------

def _normalize(vec, eps=1e-12):
    v = np.asarray(vec, float).flatten()
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero (or near-zero) magnitude encountered.")
    return v / n

def _asvec3(a):
    v = np.asarray(a, float).flatten()
    if v.size != 3:
        raise ValueError("Expected 3-element vector.")
    return v

# --------------------- core rotation matrices ---------------------

def RotMat_IRF_to_LVLH(r, v):
    """RotMat_IRF_to_LVLH(r,v) -> np.ndarray: IRF->LVLH with z=-rhat, y=-hhat, x=y×z."""
    r = _asvec3(r); v = _asvec3(v)
    z_dir = -_normalize(r)
    y_dir = -_normalize(np.cross(r, v))
    x_dir = _normalize(np.cross(y_dir, z_dir))
    return np.vstack((x_dir, y_dir, z_dir))


def RotMat_LVLH_to_BRF_by_eul(eul_deg):
    """Rotation matrix LVLH → BRF from (roll, pitch, yaw) in degrees (ZYX order)."""
    roll, pitch, yaw = np.radians(eul_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx  # intrinsic ZYX

def RotMat_by_quat(q):
    """Rotation matrix IRF → BRF from quaternion [qx, qy, qz, qw] (scalar last)."""
    qx, qy, qz, qw = np.asarray(q, float).flatten()
    n2 = qx*qx + qy*qy + qz*qz + qw*qw
    if n2 < 1e-16:
        raise ValueError("Zero-norm quaternion.")
    s = 2.0 / n2

    xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
    xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s

    return np.array([[1 - (yy + zz), xy - wz, xz + wy],
                     [xy + wz, 1 - (xx + zz), yz - wx],
                     [xz - wy, yz + wx, 1 - (xx + yy)]])

# --------------------- vector transforms ---------------------

def IRF2LVLH(u, r, v): return RotMat_IRF_to_LVLH(r, v) @ _asvec3(u)
def LVLH2IRF(u, r, v): return RotMat_IRF_to_LVLH(r, v).T @ _asvec3(u)
def LVLH2BRF_eul(u, eul_deg): return RotMat_LVLH_to_BRF_by_eul(eul_deg) @ _asvec3(u)
def BRF2LVLH_eul(u, eul_deg): return RotMat_LVLH_to_BRF_by_eul(eul_deg).T @ _asvec3(u)

def RotMat_IRF_to_BRF(r, v, eul_deg): return RotMat_LVLH_to_BRF_by_eul(eul_deg) @ RotMat_IRF_to_LVLH(r, v)
def IRF2BRF_eul(u, r, v, eul_deg): return RotMat_IRF_to_BRF(r, v, eul_deg) @ _asvec3(u)
def BRF2IRF_eul(u, r, v, eul_deg): return RotMat_IRF_to_BRF(r, v, eul_deg).T @ _asvec3(u)

# --------------------- Euler extraction ---------------------

def rotation_matrix_to_ypr(R):
    """Extract (roll, pitch, yaw) from R (intrinsic ZYX). Returns degrees."""
    R = np.asarray(R, float).reshape(3, 3)
    r20 = R[2, 0]

    if np.isclose(r20, -1.0, atol=1e-12):
        return [np.degrees(np.arctan2(R[0, 1], R[0, 2])), 90.0, 0.0]
    if np.isclose(r20, 1.0, atol=1e-12):
        return [np.degrees(np.arctan2(-R[0, 1], -R[0, 2])), -90.0, 0.0]

    pitch = np.arcsin(-r20)
    cp = np.cos(pitch)
    roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
    yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    return np.degrees([roll, pitch, yaw])

# --------------------- Attitude helper utilities ---------------------

def rodrigues_rotation(p, rotvec_rad) -> np.ndarray:
    """rodrigues_rotation_rad(p,rotvec_rad) -> np.ndarray: Rotate vector p by rotation vector (axis*angle) in radians."""
    p = np.asarray(p, float).reshape(3)
    w = np.asarray(rotvec_rad, float).reshape(3)
    theta = float(np.linalg.norm(w))
    if theta == 0.0:
        return p
    k = w / theta
    return (p*np.cos(theta) +
            np.cross(k, p)*np.sin(theta) +
            k*np.dot(k, p)*(1 - np.cos(theta)))

def rotate_body_vectors(x, y, z, p, rotvec_rad):
    """rotate_body_vectors_rad(x,y,z,p,rotvec_rad) -> tuple: Rotate x,y,z,p by same rotation vector in radians."""
    return (rodrigues_rotation(x, rotvec_rad),
            rodrigues_rotation(y, rotvec_rad),
            rodrigues_rotation(z, rotvec_rad),
            rodrigues_rotation(p, rotvec_rad))


def get_rpy_angles_irf(x, y, z):
    """Roll, pitch, yaw [deg] of BRF wrt IRF."""
    R_brf_in_irf = np.c_[x, y, z]  # columns are BRF axes in IRF
    return rotation_matrix_to_ypr(R_brf_in_irf)


def get_rpy_angles_brf(x, y, z, r, v):
    """Roll, pitch, yaw [deg] of BRF wrt LVLH (0,0,0 = nadir)."""
    R_brf_in_irf = np.c_[x, y, z]
    R_lvlh_in_irf = RotMat_IRF_to_LVLH(r, v).T
    R_rel = R_lvlh_in_irf.T @ R_brf_in_irf
    return rotation_matrix_to_ypr(R_rel)