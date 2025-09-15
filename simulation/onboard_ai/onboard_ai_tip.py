import random
from ..propagate_whales import Whale


def tip_ai_decision(whale: Whale, tpr: float, tnr: float, seed_ai: float) -> bool:
    """
    Simulate Tip AI model decision whether to forward whale to Cue.

    Parameters
    ----------
    whale : Whale
        Whale object with ai_class 'positive' or 'negative'
    tpr : float
        True positive rate (sensitivity)
    tnr : float
        True negative rate (specificity)

    Returns
    -------
    bool
        True if forwarded to Cue, False if filtered out
    """

    random.seed(seed_ai)

    if whale.ai_class_true == "whale":
        return random.random() < tpr  # forward with probability TPR
    else:  # whale.ai_class == "not-whale"
        return random.random() > tnr  # false alarm with probability (1 - TNR)



