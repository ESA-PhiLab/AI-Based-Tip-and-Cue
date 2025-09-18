import random
from ..propagate_whales import Whale


def cue_ai_decision(whale: Whale, tpr: float, tnr: float, rng: random.Random) -> bool:
    """Confirm whale with prob tpr if true whale, else with prob (1-tnr)."""


    if whale.ai_class_true == "whale":
        return rng.random() < tpr
    else:
        return rng.random() < (1 - tnr)



