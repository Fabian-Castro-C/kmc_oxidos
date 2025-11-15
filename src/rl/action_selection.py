"""
Action selection mechanisms for Swarm-RL.

Implements scalable action selection strategies like Gumbel-Max
to handle large, discrete action spaces efficiently, as described in
the SwarmThinkers paper (arXiv:2505.20094v1).
"""
import torch
from torch.distributions import Gumbel


def select_action_gumbel_max(logits: torch.Tensor) -> int:
    """
    Selects an action using the Gumbel-Max trick.

    This is equivalent to sampling from a categorical distribution defined
    by the softmax of the logits, but it is computationally much more
    efficient for very large action spaces as it avoids the explicit
    softmax calculation.

    Args:
        logits: A 1D tensor of raw, unnormalized scores (logits) for each
                possible action.

    Returns:
        The index of the selected action.
    """
    # The Gumbel-Max trick: argmax(logits + g) is equivalent to sampling
    # from softmax(logits).
    # Create a Gumbel distribution with the same shape as logits.
    # loc=0, scale=1 is the standard Gumbel distribution.
    gumbel_dist = Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    gumbel_noise = gumbel_dist.sample()

    # Add Gumbel noise to the logits to get the perturbed scores
    perturbed_logits = logits + gumbel_noise

    # The action with the highest score is the chosen one
    return torch.argmax(perturbed_logits).item()
