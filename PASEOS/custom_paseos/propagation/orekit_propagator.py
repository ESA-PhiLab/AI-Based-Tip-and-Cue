from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.orbits import OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.utils import PVCoordinates
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.sampling import PythonOrekitFixedStepHandler
from org.orekit.propagation.sampling import OrekitStepNormalizer
from org.orekit.orbits import OrbitType
from org.orekit.utils import IERSConventions
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.propagation.events import EclipseDetector
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity import ThirdBodyAttraction
from org.orekit.forces.gravity import OceanTides, SolidTides
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.orbits import CartesianOrbit
from org.orekit.data import DataContext
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from java.io import File
from orekit import JArray_double

import numpy as np
import sys, os


from org.orekit.propagation.sampling import PythonOrekitFixedStepHandler, OrekitStepNormalizer

class StepHandler(PythonOrekitFixedStepHandler):
    def __init__(self, step_sec):
        super(StepHandler, self).__init__()
        self.step_sec = step_sec
        self.states = []

    def init(self, s0, t, step):
        pass

    def handleStep(self, currentState, isLast):
        # Save each sampled state
        self.states.append(currentState)

def propagate_with_fixed_step(propagator, initialDate, duration_sec, step_sec):
    """Run once, sample all states at fixed step_sec."""
    handler = StepHandler(step_sec)
    normalizer = OrekitStepNormalizer(step_sec, handler)
    propagator.setMasterMode(normalizer)
    propagator.propagate(initialDate.shiftedBy(duration_sec))
    return handler.states


class OrekitPropagator:
    """This class serves as a wrapper to orekit. It initializes the orekit
    virtual machine and provides a method to propagate a satellite orbit.

    It follows the example from the orekit documentation:

    https://gitlab.orekit.org/orekit-labs/python-wrapper/-/blob/master/examples/Propagation.ipynb
    """

    # Constants for the numerical propagator, see orekit docs for details
    minStep = 0.0001
    maxstep = 1000.0
    initStep = 60.0
    positionTolerance = 1e-3

    def __init__(self, orbital_elements: list, epoch: AbsoluteDate, satellite_mass: float, area_s: float, cr_s: float, area_d: float, cd: float) -> None:
        """Initialize the propagator.

        Args:
            orbital_elements (list): List of orbital elements.
            epoch (AbsoluteDate): Epoch of the orbit.
            satellite_mass (float): Mass of the satellite.
        """

        # Inertial frame where the satellite is defined
        inertialFrame = FramesFactory.getEME2000()

        # Unpack the orbital elements
        a, e, i, omega, raan, lv = orbital_elements

        self.initialDate = epoch

        # Orbit construction as Keplerian
        initialOrbit = KeplerianOrbit(
            a,
            e,
            i,
            omega,
            raan,
            lv,
            PositionAngleType.TRUE,
            inertialFrame,
            epoch,
            Constants.WGS84_EARTH_MU,
        )

        # Set up the numerical propagator tolerance
        tolerances = NumericalPropagator.tolerances(
            self.positionTolerance, initialOrbit, initialOrbit.getType()
        )

        # Set up the numerical integrator
        integrator = DormandPrince853Integrator(
            self.minStep,
            self.maxstep,
            JArray_double.cast_(
                tolerances[0]
            ),  # Double array of doubles needs to be casted in Python
            JArray_double.cast_(tolerances[1]),
        )
        integrator.setInitialStepSize(self.initStep)

        # Define the initial state of the spacecraft
        self.initialState = SpacecraftState(initialOrbit, satellite_mass)


        # Set up the numerical propagator
        self.propagator_num = NumericalPropagator(integrator)
        self.propagator_num.setOrbitType(OrbitType.CARTESIAN)
        self.propagator_num.setInitialState(self.initialState)

        self.currentState = self.propagator_num.getInitialState()
        self.currentDate = self.initialDate

        # Add the force models
        gravityProvider = GravityFieldFactory.getNormalizedProvider(10, 10)
        self.propagator_num.addForceModel(
            HolmesFeatherstoneAttractionModel(
                FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider
            )
        )

        moon = CelestialBodyFactory.getMoon()
        self.propagator_num.addForceModel(ThirdBodyAttraction(moon))

        sun = CelestialBodyFactory.getSun()
        self.propagator_num.addForceModel(ThirdBodyAttraction(sun))

        wgs84_ellipsoid = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            FramesFactory.getITRF(IERSConventions.IERS_2010, True),
        )

        radiation_model = IsotropicRadiationSingleCoefficient(area_s, cr_s)
        srp_model = SolarRadiationPressure(sun, wgs84_ellipsoid, radiation_model)
        self.propagator_num.addForceModel(srp_model)

        # Set up space weather and atmosphere
        # Set up space weather and atmosphere
        site_packages = next(p for p in sys.path if "site-packages" in p)
        orekit_data_path = os.path.join(site_packages, "orekitdata")
        orekitData = File(orekit_data_path)

        manager = DataContext.getDefault().getDataProvidersManager()
        manager.addProvider(DirectoryCrawler(orekitData))
        utc = TimeScalesFactory.getUTC()
        # cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES, manager, utc)
        cssiSpaceWeatherData = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
        # atmosphere = NRLMSISE00(cssiSpaceWeatherData, sun, wgs84_ellipsoid).withSwitch(9, -1)
        atmosphere = NRLMSISE00(cssiSpaceWeatherData, sun, wgs84_ellipsoid)
        self.atmosphere = atmosphere
        # import pdb; pdb.set_trace()
        drag_model = DragForce(atmosphere, IsotropicDrag(area_d, cd))
        self.propagator_num.addForceModel(drag_model)

    def eph(self, time_since_epoch_in_seconds: float):
        targetDate = self.initialDate.shiftedBy(time_since_epoch_in_seconds)

        # Only propagate forward from current date
        if targetDate.compareTo(self.currentDate) >= 0:
            self.currentState = self.propagator_num.propagate(targetDate)
            self.currentDate = targetDate
        else:
            # If asked to go backwards in time, must restart
            self.currentState = self.propagator_num.propagate(targetDate)
            self.currentDate = targetDate

        return self.currentState

    def propagate_at_fixed_times(self, duration_sec: float, step_sec: float = 10):
        t_array = [
            self.initialDate.shiftedBy(float(dt))
            for dt in np.arange(0, duration_sec, step_sec)
        ]

        results = []
        for date in t_array:
            state = self.propagator_num.propagate(date)
            wgs84_ellipsoid = OneAxisEllipsoid(
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                Constants.WGS84_EARTH_FLATTENING,
                FramesFactory.getITRF(IERSConventions.IERS_2010, True),
            )
            sun = CelestialBodyFactory.getSun()
            # eclipse_detector = EclipseDetector(sun, 696000.0, wgs84_ellipsoid)
            # in_shadow = eclipse_detector.g(state) <= 0  # True se in ombra
            orbit = KeplerianOrbit(state.getOrbit())
            pv = state.getPVCoordinates()
            pos = pv.getPosition()
            dens = self.atmosphere.getDensity(date, pos, state.getFrame())

            results.append({
                'date': state.getDate(),
                'position': pos,
                'velocity': pv.getVelocity(),
                'a': orbit.getA(),
                'e': orbit.getE(),
                'i': orbit.getI(),
                'raan': orbit.getRightAscensionOfAscendingNode(),
                'pa': orbit.getPerigeeArgument(),
                'ta': orbit.getTrueAnomaly(),
                'density': dens  #
                # 'in_sunlight': not in_shadow
            })
        return results
