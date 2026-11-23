"""Opt-in joint PAE reranker. Importing these modules never changes an existing run."""

from .model import PAEPoseReranker, antibody_antigen_pair_mask
from .training import (
    JointLoss, acceptable_mass_loss, joint_objective,
    shared_gradient_stats, alpha_for_shared_gradient_ratio,
)

__all__ = [
    "PAEPoseReranker", "antibody_antigen_pair_mask",
    "JointLoss", "acceptable_mass_loss", "joint_objective",
    "shared_gradient_stats", "alpha_for_shared_gradient_ratio",
]
