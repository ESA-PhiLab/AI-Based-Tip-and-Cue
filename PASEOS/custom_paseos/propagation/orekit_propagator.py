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


import pykep as pk

import os, sys

import numpy as np

class OrekitPropagator:
    minStep = 1e-3
    maxstep = 86400.0
    initStep = 60.0
    positionTolerance = 1e-13

    def __init__(self, orbital_elements: list, epoch: AbsoluteDate, satellite_mass: float, area_s: float, cr_s: float, area_d: float, cd: float) -> None:
        inertialFrame = FramesFactory.getEME2000()
        a, e, i, raan, argp, M = orbital_elements
        self.initialDate = epoch

        # build temporary orbit with mean anomaly
        temp_orbit = KeplerianOrbit(
            a, e, i, argp, raan, M,
            PositionAngleType.MEAN,
            inertialFrame, epoch, Constants.WGS84_EARTH_MU
        )

        # then extract the true anomaly
        true_anomaly = temp_orbit.getTrueAnomaly()

        initialOrbit = KeplerianOrbit(
            a, e, i, argp, raan, true_anomaly,
            PositionAngleType.TRUE,
            inertialFrame, epoch, Constants.WGS84_EARTH_MU
        )
		
        #position = Vector3D(-5585307.01800000, -3984841.87800000, -81.7070000000000)  # [m]
        #velocity = Vector3D(-563.668065000000, 804.376494000000, 7559.36260000000)  # [m/s]
        #pv = PVCoordinates(position, velocity)
        #initialOrbit = CartesianOrbit(pv, FramesFactory.getEME2000(), epoch, Constants.WGS84_EARTH_MU)

        tolerances = NumericalPropagator.tolerances(
            self.positionTolerance, initialOrbit, initialOrbit.getType()
        )

        #integrator = DormandPrince853Integrator(
        #    self.minStep,
        #    self.maxstep,
        #    JArray_double.cast_(tolerances[0]),
        #    JArray_double.cast_(tolerances[1]),
        #)

        integrator = DormandPrince853Integrator(
            self.minStep,
            self.maxstep,
            1e-13,
            1e-13,
        )
        integrator.setInitialStepSize(self.initStep)

        initialState = SpacecraftState(initialOrbit, satellite_mass)

        self.propagator_num = NumericalPropagator(integrator)
        self.propagator_num.setOrbitType(OrbitType.CARTESIAN)
        self.propagator_num.setInitialState(initialState)

        gravityProvider = GravityFieldFactory.getNormalizedProvider(100, 100)
        self.propagator_num.addForceModel(
            HolmesFeatherstoneAttractionModel(
                FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider
            )
        )

      # propagator.addForceModel(
        #self.propagator_num.addForceModel(OceanTides(FramesFactory.getEME2000()))
        #self.propagator_num.addForceModel(SolidTides(FramesFactory.getEME2000()))

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
        site_packages = next(p for p in sys.path if "site-packages" in p)
        orekit_data_path = os.path.join(site_packages, "orekitdata")  # ✅ correct

        print(orekit_data_path)

        orekitData = File(orekit_data_path)
        manager = DataContext.getDefault().getDataProvidersManager()
        manager.addProvider(DirectoryCrawler(orekitData))
        utc = TimeScalesFactory.getUTC()
        #cssi = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES, manager, utc)
        cssiSpaceWeatherData = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
        #atmosphere = NRLMSISE00(cssiSpaceWeatherData, sun, wgs84_ellipsoid).withSwitch(9, -1)
        atmosphere = NRLMSISE00(cssiSpaceWeatherData, sun, wgs84_ellipsoid)
        self.atmosphere = atmosphere
        #import pdb; pdb.set_trace()
        drag_model = DragForce(atmosphere, IsotropicDrag(area_d, cd))
        self.propagator_num.addForceModel(drag_model)


    def eph(self, time_since_epoch_in_seconds: float):
        state = self.propagator_num.propagate(
            self.initialDate, self.initialDate.shiftedBy(time_since_epoch_in_seconds)
        )
        return state

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
            #eclipse_detector = EclipseDetector(sun, 696000.0, wgs84_ellipsoid)
            #in_shadow = eclipse_detector.g(state) <= 0  # True se in ombra
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
                'density': dens  # 👈 Densità salvata
                #'in_sunlight': not in_shadow  # 👈 aggiunto
            })
        return results



