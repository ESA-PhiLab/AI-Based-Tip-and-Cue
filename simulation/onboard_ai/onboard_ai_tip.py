import random
from ..propagate_whales import Whale
import random


def tip_ai_decision(whale: Whale, tpr: float, tnr: float, rng: random.Random) -> bool:
    """Forward whale to Cue with prob tpr if true whale, else with prob (1-tnr)."""
    if whale.ai_class_true == "whale":
        return rng.random() < tpr
    else:
        return rng.random() < (1 - tnr)



