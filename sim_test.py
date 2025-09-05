from settings import *

import orekit
import pyvista as pv
import numpy as np
import pykep as pk
from datetime import datetime, timedelta
import time
import gc
import os

from orekit.pyhelpers import setup_orekit_curdir

from paseos import ActorBuilder, SpacecraftActor
import paseos

from custom_paseos.propagation.orekit_propagator import OrekitPropagator
from simulation.constellation import build_constellation
from simulation.simulation_functions import propagate_actor

from org.orekit.time import AbsoluteDate, TimeScalesFactory

# -----------------------------
# Config
# -----------------------------
RESET_INTERVAL = 50  # steps between propagator resets
show_orbits = False

# -----------------------------
# Init Orekit
# -----------------------------
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

utc = TimeScalesFactory.getUTC()
t0_orekit = AbsoluteDate(
    t0.year, t0.month, t0.day,
    t0.hour, t0.minute,
    t0.second + t0.microsecond / 1e6,
    utc
)
t0_pykep = pk.epoch_from_string(t0.strftime("%Y-%m-%d %H:%M:%S"))

# -----------------------------
# Build Tip + Cue constellations
# -----------------------------
planet_lst_tip, sats_tip, _ = build_constellation(params_tip, "Tip", t0_pykep)
planet_lst_cue, sats_cue, _ = build_constellation(params_cue, "Cue", t0_pykep)
all_planets = planet_lst_tip + planet_lst_cue

# -----------------------------
# Create actors
# -----------------------------
tip_actors, cue_actors = [], []
for planet in all_planets:
    propagator = OrekitPropagator(
        orbital_elements=planet.orbital_elements,
        epoch=t0_orekit,
        satellite_mass=satellite_mass,
        area_s=area_s, cr_s=cr_s, area_d=area_d, cd=cd
    )
    actor = ActorBuilder.get_actor_scaffold(
        name=planet.name, actor_type=SpacecraftActor, epoch=t0_pykep
    )
    ActorBuilder.set_custom_orbit(
        actor,
        lambda t, p=propagator, t0=t0_pykep: (
            list(p.eph((t.mjd2000 - t0.mjd2000) * pk.DAY2SEC).getPVCoordinates().getPosition().toArray()),
            list(p.eph((t.mjd2000 - t0.mjd2000) * pk.DAY2SEC).getPVCoordinates().getVelocity().toArray())
        ),
        t0_pykep
    )
    ActorBuilder.set_central_body(actor, pk.planet.jpl_lp("earth"), radius=R_earth)
    (tip_actors if "Tip" in planet.name else cue_actors).append(actor)

# -----------------------------
# Init sim
# -----------------------------
if tip_actors:
    sim = paseos.init_sim(local_actor=tip_actors[0])
    for actor in tip_actors[1:] + cue_actors:
        sim.add_known_actor(actor)
else:
    sim = paseos.init_sim(local_actor=cue_actors[0])
    for actor in cue_actors[1:]:
        sim.add_known_actor(actor)

n_steps_total = int(sim_duration_seconds / sim_step_seconds) + 1
print("Total number of simulation steps:", n_steps_total)

trajectories = None  # disable orbit storage

# -----------------------------
# Loop
# -----------------------------
elapsed_time, n_steps = 0.0, 0
while elapsed_time <= sim_duration_seconds:

    t_start = time.time()
    t_pykep = sim.local_time
    tip_positions, cue_positions = [], []

    # -------------------------
    # Reset propagators every N steps
    # -------------------------
    if n_steps % RESET_INTERVAL == 0 and n_steps > 0:
        current_absdate = t0_orekit.shiftedBy(float(elapsed_time))
        current_pykep = pk.epoch(t0_pykep.mjd2000 + elapsed_time / pk.DAY2SEC)


        for planet, actor in zip(all_planets, tip_actors + cue_actors):
            new_propagator = OrekitPropagator(
                orbital_elements=planet.orbital_elements,
                epoch=current_absdate,
                satellite_mass=satellite_mass,
                area_s=area_s, cr_s=cr_s, a# -----------------------------
# Loop
# -----------------------------
elapsed_time, n_steps = 0.0, 0
while elapsed_time <= sim_duration_seconds:

    t_start = time.time()
    t_pykep = sim.local_time
    tip_positions, cue_positions = [], []

    # -------------------------
    # Reset propagators every 10 steps using current state
    # -------------------------
    if n_steps % 10 == 0 and n_steps > 0:
        current_absdate = t0_orekit.shiftedBy(float(elapsed_time))
        current_pykep = pk.epoch(t0_pykep.mjd2000 + elapsed_time / pk.DAY2SEC)

        for actor in tip_actors + cue_actors:
            # Get current state
            r, v = actor.get_position_velocity(current_pykep)

            # Create new propagator from state vector
            new_propagator = OrekitPropagator.from_cartesian(
                position=r, velocity=v,
                epoch=current_absdate,
                satellite_mass=satellite_mass,
                area_s=area_s, cr_s=cr_s, area_d=area_d, cd=cd
            )

            # Wrap orbit function
            def new_orbit(t, p=new_propagator, t0=current_pykep):
                pv = p.eph((t.mjd2000 - t0.mjd2000) * pk.DAY2SEC).getPVCoordinates()
                return (
                    list(pv.getPosition().toArray()),
                    list(pv.getVelocity().toArray())
                )

            ActorBuilder.set_custom_orbit(actor, new_orbit, current_pykep)

    # -------------------------
    # Propagate Tip
    # -------------------------
    for actor in tip_actors:
        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        tip_positions.append(r)

    # -------------------------
    # Propagate Cue
    # -------------------------
    for actor in cue_actors:
        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        cue_positions.append(r)

    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f"\t {n_steps} Time iteration: {t_elapsed:.3f}")

    if n_steps % 10 == 0:
        gc.collect()

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)
    elapsed_time += sim_step_seconds
    n_steps += 1
rea_d=area_d, cd=cd
            )
            def new_orbit(t, p=new_propagator, t0=current_pykep):
                pv = p.eph((t.mjd2000 - t0.mjd2000) * pk.DAY2SEC).getPVCoordinates()
                return (
                    list(pv.getPosition().toArray()),
                    list(pv.getVelocity().toArray())
                )
            ActorBuilder.set_custom_orbit(actor, new_orbit, current_pykep)

    # -------------------------
    # Propagate Tip
    # -------------------------
    for actor in tip_actors:
        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        tip_positions.append(r)

    # -------------------------
    # Propagate Cue
    # -------------------------
    for actor in cue_actors:
        r_vec, v_vec, r, v = propagate_actor(actor, t_pykep, trajectories, n_steps, show_orbits)
        cue_positions.append(r)

    t_end = time.time()
    t_elapsed = t_end - t_start
    print(f"\t {n_steps} Time iteration: {t_elapsed:.3f}")

    if n_steps % 10 == 0:
        gc.collect()

    sim.advance_time(time_to_advance=sim_step_seconds, current_power_consumption_in_W=0.0)
    elapsed_time += sim_step_seconds
    n_steps += 1
