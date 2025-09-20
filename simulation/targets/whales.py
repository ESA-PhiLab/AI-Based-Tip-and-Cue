# whales.py
# Whale simulation with land avoidance, geodesic motion, and unified Whale class.

import os
import math
import random
import numpy as np

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from settings import R_earth, max_abs_lat
import uuid

from .water_target_utils import direct_geodesic, is_water


# --------------------------- WHALE CLASS ---------------------------

@dataclass
class Whale:
    id: int
    lat: float
    lon: float
    alt: float = 0.0
    speed: float = field(default_factory=lambda: random.uniform(0.5, 3.0))
    heading: float = field(default_factory=lambda: random.uniform(0.0, 360.0))

    ai_class_true: Optional[str] = None
    ai_class_predicted: Optional[str] = None
    running_ai: Optional[bool] = False

    tip_actor: Optional[str] = None
    cue_actor: Optional[str] = None

    t_observed_tip: Optional[datetime] = None
    t_confirmed_tip: Optional[datetime] = None
    t_tasked_tip: Optional[datetime] = None
    t_tasked_cue: Optional[datetime] = None
    t_observed_cue: Optional[datetime] = None
    t_confirmed_cue: Optional[datetime] = None

    coord_observed: Optional[tuple[float, float, float]] = None

    state_observing: int = 0
    state_tasked:     int = 0
    state_confirming: int = 0

    confirmed_tip: Optional[bool] = None
    confirmed_cue: Optional[bool] = None

    assigned_cue: Optional[str] = None
    delay_confirmation_tip: float = 0.0
    delay_confirmation_cue: float = 0.0

    tip_observation_counter: int = 0
    cue_observation_counter: int = 0

    detection_id: Optional[str] = None

    def step(self, mask: np.ndarray, res_deg: float, dt_sec: float, whale_propagation: dict):
        """Advance whale position with OU speed, diffusive heading, and land avoidance."""
        if dt_sec <= 0.0:
            return

        v = self.speed
        k = whale_propagation["speed_mean_reversion_per_s"]
        noise = random.gauss(0.0, whale_propagation["speed_noise_sigma"] * math.sqrt(dt_sec))
        v = v + k * (whale_propagation["speed_mean"] - v) * dt_sec + noise
        v = max(whale_propagation["speed_min"], min(whale_propagation["speed_max"], v))

        h = (self.heading + random.gauss(0.0, whale_propagation["turn_std_deg_per_sqrt_s"] * math.sqrt(dt_sec))) % 360.0

        dist = v * dt_sec
        lat0, lon0 = self.lat, self.lon
        lat1, lon1 = direct_geodesic(lat0, lon0, h, dist)

        if abs(lat1) > max_abs_lat:
            h = (h + 180.0) % 360.0
            lat1, lon1 = direct_geodesic(lat0, lon0, h, dist)

        tries = 0
        while (not is_water(lat1, lon1, mask, res_deg)) or (abs(lat1) > 89.9):
            delta = random.uniform(30.0, 150.0) * (1 if random.random() < 0.5 else -1)
            h = (h + delta) % 360.0
            dist *= 0.7
            lat1, lon1 = direct_geodesic(lat0, lon0, h, dist)
            tries += 1
            if tries >= whale_propagation["land_avoid_max_tries"]:
                lat1, lon1 = lat0, lon0
                break

        self.lat, self.lon, self.speed, self.heading = lat1, lon1, v, h

    def position(self) -> tuple[float, float, float]:
        return self.lat, self.lon, self.alt

    def update_detection_id(self):
        """Assign or reset detection_id based on AI classification & state."""

        # Assign at first observation
        if self.detection_id is None and (self.t_observed_tip or self.t_observed_cue):
            self.detection_id = str(uuid.uuid4())

        # Reset if TIP confirmed negative (predicted not-whale)
        if self.t_confirmed_tip and self.ai_class_predicted == "not-whale":
            self.detection_id = None

        # Reset if CUE confirmation happened (always ends cycle)
        if self.t_confirmed_cue:
            self.detection_id = None

        # Reset if target was dropped before confirmation
        if self.state_observing == 0 and self.state_tasked == 0 and self.state_confirming == 0:
            if self.t_observed_tip or self.t_observed_cue:
                self.detection_id = None

# --------------------------- INITIAL TARGETS ---------------------------


def init_whales(known_targets: list[tuple[float, float, float]],  seed_val: Optional[int] = None, pos_fraction: float = 1.0) -> dict[int, Whale]:
    """
    Initialize Whale objects with class labels.

    Parameters
    ----------
    known_targets : list
        List of (lat, lon, alt) tuples.
    seed_val : int, optional
        Random seed for reproducibility.
    pos_fraction : float
        Fraction of whales to be assigned 'positive' class.
    """

    if seed_val is not None:
        random.seed(seed_val)

    n = len(known_targets)
    n_pos = int(round(n * pos_fraction))
    labels = ["whale"] * n_pos + ["not-whale"] * (n - n_pos)
    random.shuffle(labels)

    whales: dict[int, Whale] = {}
    for idx, (lat, lon, alt_m) in enumerate(known_targets):
        whales[idx] = Whale(id=idx, lat=lat, lon=lon, alt=alt_m, ai_class_true=labels[idx])
    return whales


def update_whales(all_targets: dict[int, Whale], mask: np.ndarray, res_deg: float, dt: float, whale_propagation: dict):
    for w in all_targets.values():
        w.step(mask, res_deg, dt_sec=dt, whale_propagation=whale_propagation)




