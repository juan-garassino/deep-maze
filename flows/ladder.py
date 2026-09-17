"""The convergence ladder for the ephemeral Cloud Run training job.

Each rung is one Cloud Run Job execution running scripts/train_runpod.py with a
single-stage CURRICULUM, warm-started from the previous rung's best model. The
ladder starts at the easiest config that provably learns (small full-obs maze,
single treasure) and hardens toward the partial-obs DRQN/DTQN memory thesis.
`has_learned` is the pure climb gate the flow consults after each rung — kept
dependency-free (only dataclasses) so it is trivially unit-testable.

Curriculum field order (see scripts/train_runpod.py): W,H,n_treasures,episodes,
max_steps[,seq_len]. Collect-all tours follow max_steps = 3*(w+h)*n_treasures.
All values are STARTING points — tune the rungs against real eval curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rung:
    name: str
    agent: str           # 'drqn' | 'dtqn'
    curriculum: str      # single stage: "W,H,nt,eps,max_steps[,seq_len]"
    partial: int         # egocentric window; 0 = full observability
    collect_all: bool
    learn_gate: float    # climb when max(eval, best) success >= this
    extra_env: dict = field(default_factory=dict)


# Rung 0 proves the pipeline learns at all (full-obs single-treasure DRQN
# converges readily); each rung hardens one axis — observability, then
# multi-treasure collect-all, then maze size + memory window (seq_len).
# Sizes are odd on both axes (a DFS perfect maze tiles its odd-coord lattice
# flush to the border only at odd dimensions — even sizes strand a wall band).
LADDER: list[Rung] = [
    Rung("r0-7x7-1t-full",  "drqn", "7,7,1,600,80",          partial=0, collect_all=False, learn_gate=0.60),
    Rung("r1-9x9-1t",       "drqn", "9,9,1,1000,120,8",      partial=5, collect_all=False, learn_gate=0.50),
    Rung("r2-11x19-2t",     "drqn", "11,19,2,1500,192,8",    partial=5, collect_all=True,  learn_gate=0.50),
    Rung("r3-19x39-4t",     "dtqn", "19,39,4,2000,720,16",   partial=5, collect_all=True,  learn_gate=0.40),
    Rung("r4-29x59-6t",     "dtqn", "29,59,6,2500,1620,32",  partial=5, collect_all=True,  learn_gate=0.35),
]

# The CPU smoke rung: tiny nets (NANO), 60 episodes, gate 0 so it always
# "learns" — exercises submit→result→promote→resync end-to-end without a GPU.
NANO_RUNG = Rung("smoke-nano", "drqn", "7,7,1,60,40", partial=2, collect_all=False,
                 learn_gate=0.0, extra_env={"NANO": "true"})


def has_learned(agent_result: dict | None, gate: float) -> bool:
    """True if the rung's eval/best success cleared its learn-gate."""
    if not agent_result:
        return False
    succ = max(agent_result.get("eval_success_rate", 0.0) or 0.0,
               agent_result.get("best_success_rate", 0.0) or 0.0)
    return succ >= gate
