""" Module containing the functions to generate rotation matrix between reference frames 
Three Reference Frames have been identified:

LVLH (Local Vertical Local Horizon) :
- x-axis in the direction of the spacecraft's velocity vector;
- z-axis in the opposite direction of the spacecraft's position vector;
- y-axis completes the right triad, with a direction opposite to the angular momentum vector h;

Note: 

The LVLH reference frame has been defined in such a way that the z-axis points toward the center of the planet (usefull for Planet-Observation)

BRF (Body Reference Frame) has been designed as a reference frame nominally aligned with the LVLH.

IRF (Inertial Reference Frame) is instead nominally the Earth Centered Inertial reference frame. 
"""


import numpy as np

def RotMat_IRF_to_LVLH(r,v):
    
    """ The function creates the transformation matrix from IRF to LVLH. 
    Args:
    - r (np.ndarray) [3x1] : position vector of the spacecraft's CG expressed in the IRF;
    - v (np.ndarray) [3x1] : velocity vector of the spacecraft's CG expressed in the IRF;
    Returns
    - R (np.ndarray) [3x3] : Rotation Matrix from IRF to LVLH
    Note: The rotation matrix from LVLH to IRF can be obtained through a transposition of the R matrix 
    """
    #import pdb; pdb.set_trace()
    r = r.flatten()
    v = v.flatten()
    z_dir = - r / np.linalg.norm(r)
    
    # LVLH x-axis: Along velocity direction
    x_dir = v / np.linalg.norm(v)

    # LVLH y-axis: Completes the right-handed frame
    y_dir = np.cross(z_dir, x_dir)
    y_dir = y_dir / np.linalg.norm(y_dir)
    
    T = np.array([x_dir, y_dir, z_dir])
    #import pdb; pdb.set_trace()
    return T


def RotMat_LVLH_to_BRF_by_eul(eul_ang):
    
    """ The function creates the transformation matrix from LVLH to BRF. 
    Args:
    - euler_angles (np.ndarray) [3x1] : array of euler angles in the order roll-pitch-yaw;
    Returns
    - T (np.ndarray) [3x3] : Rotation Matrix from IRF to LVLH
    Note: the rotation matrix has been built according to a 3-2-1 matrix of rotations
    """
    roll_deg, pitch_deg, yaw_deg = eul_ang 
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    A = np.array(
    [
      [1,0,0],
      [0, np.cos(roll), np.sin(roll)],
      [0, -np.sin(roll), np.cos(roll)],
      
    ]
    )

    B = np.array(
    [
    [np.cos(pitch), 0, -np.sin(pitch)],
    [0, 1, 0],
    [np.sin(pitch), 0, np.cos(pitch)],
    ]
    )

    C = np.array(
    [
    [np.cos(yaw), np.sin(yaw), 0],
    [-np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1],
    ]
    )

    T = A @ B @ C

    return T


def RotMat_by_quat(q):
    """ The function creates the transformation matrix from IRF to BRF using quaternion coming up from the quaternion kinematic propagation. 
    Args:
    - q (np.ndarray) [4x1] : quaternion vector relating the BRF to the IRF [NB: scalar last convention adopted!]
    Returns
    - R (np.ndarray) [3x3] : Rotation Matrix from BRF to LVLH
    Note: The rotation matrix from IRF to BRF can be obtained through a transposition of the R matrix.

    Rationale: given the quaternion relating Reference Frame 1 (RF1) to Reference Frame 2 (RF2) following a scalar last convetion q = [q1 q2 q3 q0], 
    it is possible to define the rotation matrix between the two reference frames as:

                |1 - 2*(2*q2 + 2*q3)     2*(q1*q2 - q0*q3)    2*(q1*q3 + q0*q2)|
    T_RF1_RF2 = |2*(q1*q2 + q0*q3)       1-2*(2*q1 + 2*q3)    2*(q2*q3 - q0*q1)|
                |2*(q1*q3 - q0*q2)       2*(q2*q3 + q0*q1)    1-2*(2*q1 + 2*q2)| 

    """

    q1, q2, q3, q0 = q
    T = np.array([
    [1 - 2*(q2**2 + q3**2),  2*(q1*q2 - q0*q3),  2*(q1*q3 + q0*q2)],
    [2*(q1*q2 + q0*q3),  1 - 2*(q1**2 + q3**2),  2*(q2*q3 - q0*q1)],
    [2*(q1*q3 - q0*q2),  2*(q2*q3 + q0*q1),  1 - 2*(q1**2 + q2**2)]
    ])

    return T


def IRF2LVLH(u, r, v):
    """  
    Description: The function creates a vector trasposed in the LVLH reference frame from the IRF.    
    Note: The function uses the "RotMat_IRF_to_LVLH" 
    """

    T = RotMat_IRF_to_LVLH(r,v)

    u =  T @ u

    return u

def LVLH2IRF(u, r, v):
    """  
    Description: The function creates a vector trasposed in the IRF reference frame from the LVLH.    
    Note: The function uses the "RotMat_IRF_to_LVLH" 
    """
    T = RotMat_IRF_to_LVLH(r, v)

    u = np.linalg.inv(T) @ u

    return u

def LVLH2BRF_eul(u,eul_ang):
    """  
    Description: The function creates a vector trasposed in the BRF reference frame from the LVLH.    
    Note: - The function uses the "RotMat_LVLH_to_BRF_by_EUL
          - Eul angle in deg" 
    """
    T = RotMat_LVLH_to_BRF_by_eul(eul_ang)

    u = T @ u

    return u


def BRF2LVLH_eul(u,eul_ang):
    """ 
    Description: The function creates a vector trasposed in the LVLH Reference Frame from the BRF.    
    Note: - The function uses the "RotMat_LVLH_to_BRF_by_EUL
          - Eul angle in deg" 
    """
    T = np.linalg.inv(RotMat_LVLH_to_BRF_by_eul(eul_ang))

    u = T @ u

    return u

def IRF2BRF_quat(u,q):

    """  
    Description: The function creates a vector trasposed in the BRF from the IRF.    
    Note: The function uses the "RotMat_LVLH_to_BRF_b" 
    """
    T = RotMat_by_quat(q)

    u = T @ u,

    return u

def BRF2IRF_quat(u,q):
    """  
    Description: The function creates a vector trasposed in the BRF from the IRF.    
    Note: The function uses the "RotMat_LVLH_to_BRF_b" 
    """
    T = np.linalg.inv(RotMat_by_quat(q))

    u = T @ u,

    return u

def RotMat_IRF_to_BRF(r,v,eul_ang):

    """  
    Description: The function creates the rotation matrix from IRF to the BRF
    Note: 
          - The function uses the "RotMat_LVLH_to_BRF_by_eul(r,v) and "RotMat_IRF_to_LVLH(r, v);
          - The eul_ang are in deg;
    """
    #print(f"r.shape: {r.shape}, v.shape: {v.shape}")
    T1 = RotMat_IRF_to_LVLH(r,v)
    T2 = RotMat_LVLH_to_BRF_by_eul(eul_ang)

    T = T2 @ T1

    
    return T

def IRF2BRF_eul(u,r,v,eul_ang):
    
    """  
    Description: The function transposes a vector in the BRF from the IRF
    Note: 
          - The function uses the "RotMat_LVLH_to_BRF_by_eul(r,v) and "RotMat_IRF_to_LVLH(r, v);
          - The eul_ang are in deg;
    """
#    print(f"T.Shape: {T.shape}")
#    print(f"u.shape: {u.shape}")  # Deve essere (3,) o (3,1)
    
    T = RotMat_IRF_to_BRF(r,v,eul_ang)
    #import pdb; pdb.set_trace()
    u = T @ u
    
    return u

def BRF2IRF_eul(u,r,v,eul_ang):
    
    """  
    Description: The function transposes a vector in the BRF from the IRF
    Note: 
          - The function uses the "RotMat_LVLH_to_BRF_by_eul(r,v) and "RotMat_IRF_to_LVLH(r, v);
          - The eul_ang are in deg;
    """
    #import pdb; pdb.set_trace()
    T = np.linalg.inv(RotMat_IRF_to_BRF(r,v,eul_ang))
    u = T @ u
    return u

def rotation_matrix_to_ypr(R):
    """
    Convert a 3x3 rotation matrix into yaw, pitch, roll (ZYX order).

    Parameters
    ----------
    R : np.ndarray
        3x3 rotation matrix

    Returns
    -------
    yaw : float
        Rotation around Z axis [rad]
    pitch : float
        Rotation around Y axis [rad]
    roll : float
        Rotation around X axis [rad]
    """

    if not (R.shape == (3, 3)):
        raise ValueError("Input must be a 3x3 rotation matrix")

    # Check for gimbal lock (pitch = ±90°)
    if np.isclose(R[2, 0], -1.0):
        pitch = np.pi / 2
        yaw = np.arctan2(R[0, 1], R[0, 2])
        roll = 0.0
    elif np.isclose(R[2, 0], 1.0):
        pitch = -np.pi / 2
        yaw = np.arctan2(-R[0, 1], -R[0, 2])
        roll = 0.0
    else:
        pitch = -np.arcsin(R[2, 0])
        cos_pitch = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cos_pitch, R[2, 2] / cos_pitch)
        yaw = np.arctan2(R[1, 0] / cos_pitch, R[0, 0] / cos_pitch)

    return yaw, pitch, roll










