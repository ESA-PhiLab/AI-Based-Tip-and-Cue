from simulation.targets.whales import Whale
import random


def tip_ai_decision(whale: Whale, tpr: float, tnr: float, rng: random.Random) -> (bool, str):
    """Forward whale to Cue with prob tpr if true whale, else with prob (1-tnr)."""
    if whale.ai_class_true == "whale":
        prediction = rng.random() < tpr
    else:
        prediction = rng.random() < (1 - tnr)

    if prediction:
        return prediction, "whale-tipped"
    else:
        return prediction, "not-whale"



