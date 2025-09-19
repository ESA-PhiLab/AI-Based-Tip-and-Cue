import random
from simulation.whales import Whale


def cue_ai_decision(whale: Whale, tpr: float, tnr: float, rng: random.Random) -> (bool, str):
    """Confirm whale with prob tpr if true whale, else with prob (1-tnr)."""

    """Forward whale to Cue with prob tpr if true whale, else with prob (1-tnr)."""

    if whale.ai_class_true == "whale":
        prediction = rng.random() < tpr
    else:
        prediction = rng.random() < (1 - tnr)

    if prediction:
        return prediction, "whale"
    else:
        return prediction, "not-whale"




