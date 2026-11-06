"""SteadyHand 2: a hand that looks before it grabs.

EMG says when to grab and which grip; a wrist camera says which grip the
object affords. The hand moves only when both agree with enough confidence,
and the misfire promise comes from SteadyHand's conformal / adaptive cutoffs.
"""
from .fusion import fuse, act, misfires, summarize, Decisions
from .calibrate import conformal_cutoff, run_adaptive

__version__ = "0.1.0"
__all__ = ["fuse", "act", "misfires", "summarize", "Decisions", "conformal_cutoff", "run_adaptive"]
