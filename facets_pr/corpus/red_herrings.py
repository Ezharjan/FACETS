"""Twenty red-herring sentences used by Axis C (C1).

Each sentence is syntactically plausible inside a body-worn-camera transcript
turn but factually impossible. They act as canaries: a careful reviewer should
strike them out, so any that survive into the final report signal that the
draft was waved through without real scrutiny.
"""
from __future__ import annotations

import random
from typing import Sequence

RED_HERRINGS: tuple[str, ...] = (
    "A 9-foot-tall purple horse galloped through the intersection.",
    "The defendant produced a passport from the Lunar Republic.",
    "An 800-pound gorilla was observed driving a blue Ford Mustang past the scene.",
    "A flock of bioluminescent penguins crossed the median strip.",
    "The witness produced a hand-written letter signed by Abraham Lincoln dated last Tuesday.",
    "A unicorn standing in the front yard nodded in agreement.",
    "The suspect dropped a small dragon, which then flew away.",
    "Officers observed a spaceship hovering above the residence for approximately three minutes.",
    "A talking parrot recited the suspect's Miranda warnings unprompted.",
    "The vehicle was a 1952 Tucker Torpedo with hovering capability.",
    "A wizard wearing a pointed blue hat directed traffic at the corner.",
    "The defendant produced a wand and cast a Patronus charm.",
    "A swarm of mechanical bees emerged from the suspect's backpack.",
    "The K-9 unit was assisted by a trained Sasquatch.",
    "Surveillance footage shows the suspect teleporting from the rear yard to the porch.",
    "A 30-foot-tall robot was observed standing next to the parked vehicle.",
    "The complainant stated she had been visited by a Time Lord earlier that evening.",
    "Officers noted a centaur loitering near the convenience store entrance.",
    "A fire-breathing salamander emerged from the engine compartment.",
    "The suspect's ID card identified him as a citizen of Atlantis.",
)


def sample_red_herring(seed: int) -> str:
    """Deterministically sample one red herring given an integer seed."""
    rng = random.Random(seed)
    return rng.choice(RED_HERRINGS)
