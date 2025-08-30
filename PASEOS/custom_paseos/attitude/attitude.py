import numpy as np
import pykep as pk
import pymap3d as pm

from paseos.actors.spacecraft_actor import SpacecraftActor
from loguru import logger

from ..utils.reference_frame_transformation import(
    LVLH2IRF,
    IRF2LVLH,
    LVLH2BRF_eul,
    BRF2LVLH_eul,
    BRF2IRF_eul,
    IRF2BRF_eul,
)

from ..utils.point_transformation import(
    Point_Geodetic2ECI,
    )

class Attitude:
     #Spaecraft_actor.
     _actor = None
     #Actor attitude in deg
     _actor_attitude_in_deg = None
     #Actor angular velocity in body frame deg/s
     _actor_angular_velocity_deg_s = None
     #Actor angular acceleration in body frame is deg/s^2
     _actor_angular_acceleration = None
     #Actor pointing vector expressed in inertial frame.
     _actor_pointing_vector_eci = None
     #Actor pointing vector in body reference frame
     _actor_pointing_vector_body = None
     """
     This classe is provided with all the functions needed to perform the dedicated Earth-Observation activity
     """
     def __init__(
        self,
        local_actor,
        # initial conditions:
        actor_initial_attitude_in_deg: list[float] = [0.0, 0.0, 0.0],
        actor_initial_angular_velocity_deg_s: list[float] = [0.0, 0.0, 0.0],
        # pointing vector in body frame: (defaults to body z-axis)
        actor_pointing_vector_body: list[float] = [0.0, 0.0, 1.0],
        #actor_residual_magnetic_field_body: list[float] = None,
        #accommodation_coefficient: float = None,
     ):
        """Creates an attitude model to model actor attitude based on
        initial conditions (initial attitude and angular velocity) and
        external disturbance torques.
        Args:
            actor (SpacecraftActor): Actor to model.
            actor_initial_attitude_in_deg (list of floats, optional): Actor's initial attitude ([roll, pitch, yaw]) angles.
            actor_initial_angular_velocity (list of floats, optional) in the body frame: Actor's initial angular velocity
            actor_pointing_vector_body (list of floats, optional): User defined vector in the Actor body.
            actor_residual_magnetic_field_body (list of floats, optional): Actor's own magnetic field modeled
        """
        assert isinstance(local_actor, SpacecraftActor), (
            "local_actor must be a " "SpacecraftActor" "."
        )

        logger.trace("Initializing Attitude Function.")
        self._actor = local_actor
        # convert to np.ndarray
        self._actor_attitude_in_deg = np.array(actor_initial_attitude_in_deg)
        self._actor_angular_velocity_deg_s = np.array(actor_initial_angular_velocity_deg_s)

        # normalize inputted pointing vector & convert to np.ndarray
        self._actor_pointing_vector_body = np.array(actor_pointing_vector_body) / np.linalg.norm(
            np.array(actor_pointing_vector_body)
        )

        # pointing vector expressed in Earth-centered inertial frame from the BRF
        self._actor_pointing_vector_eci = BRF2IRF_eul(
            self._actor_pointing_vector_body,
            np.array(self._actor.get_position(self._actor.local_time)),
            np.array(self._actor.get_position_velocity(self._actor.local_time)[1]),
            np.array(self._actor_attitude_in_deg),
        )
      
     
     def compute_obs_angle(self, target_lat, target_lon, target_alt, datetime_utc, eul_ang):
        target_eci = np.array(Point_Geodetic2ECI(target_lat,target_lon,target_alt,datetime_utc))
        sat_p_eci = np.array(self._actor.get_position(self._actor.local_time))
        sat_v_eci = np.array(self._actor.get_position_velocity(self._actor.local_time)[1])
        target_eci = np.array(target_eci).reshape(3)
        sat_p_eci = np.array(sat_p_eci).reshape(3)
        sat_v_eci = np.array(sat_v_eci).reshape(3)
        vector_eci = np.array(target_eci - sat_p_eci)
        #import pdb; pdb.set_trace()
        direction_eci = np.array(vector_eci/np.linalg.norm(vector_eci))
        #import pdb; pdb.set_trace()
        #print(direction_eci)
        direction_BRF = IRF2BRF_eul(direction_eci,sat_p_eci,sat_v_eci,eul_ang)
        #import pdb; pdb.set_trace()
        cos_theta = np.dot(direction_BRF,self._actor_pointing_vector_body)
        angle = np.degrees(np.arccos(np.clip(cos_theta,-1.0,1.0)))
        
        return angle
        
        



             








        







