"""Disturbance torque calculations in the spacecraft body frame."""

import numpy as np
from ..utils.reference_frame_transformation import IRF2BRF_eul


def calculate_aero_torque():
    """Placeholder: aerodynamic torque in BRF."""
    return np.zeros(3)


def calculate_grav_torque():
    """Placeholder: gravity gradient torque in BRF."""
    return np.zeros(3)


def calculate_magnetic_torque(m_earth, m_sat, position, velocity, eul_deg):
    """
    Disturbance torque due to Earth’s magnetic field (dipole model).

    Args:
        m_earth (np.ndarray): Earth dipole moment [Am²] in IRF.
        m_sat (np.ndarray): spacecraft residual dipole moment [Am²] in BRF.
        position (np.ndarray): spacecraft ECI position [m].
        velocity (np.ndarray): spacecraft ECI velocity [m/s].
        eul_deg (array-like): current Euler angles [deg], (roll, pitch, yaw).

    Returns:
        np.ndarray: disturbance torque [Nm] in BRF.
    """
    position = np.asarray(position, float)
    r = np.linalg.norm(position)
    if r < 1e-6:
        raise ValueError("Position vector too small for magnetic torque computation.")

    r_hat = position / r

    # Magnetic flux density in IRF (dipole model)
    B_eci = 1e-7 * (3 * np.dot(m_earth, r_hat) * r_hat - m_earth) / (r ** 3)

    # Transform field to BRF using spacecraft attitude (deg)
    B_brf = IRF2BRF_eul(B_eci, position, velocity, eul_deg)

    # Torque = spacecraft dipole × local B-field
    return np.cross(m_sat, B_brf)
