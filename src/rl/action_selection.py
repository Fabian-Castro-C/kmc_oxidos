"""
Action selection mechanisms for Swarm-RL.

Implements scalable action selection strategies like Gumbel-Max
to handle large, discrete action spaces efficiently, as described in
the SwarmThinkers paper (arXiv:2505.20094v1).
"""
import torch
from torch.distributions import Gumbel


def select_action_gumbel_max(logits: torch.Tensor) -> tuple[int, torch.Tensor]:
    """
    Selects an action using the Gumbel-Max trick.

    This is equivalent to sampling from a categorical distribution defined
    by the softmax of the logits, but it is computationally much more

    Args:
        logits: A 1D tensor of raw, unnormalized scores (logits) for each
                possible action.

    Returns:
        A tuple containing:
        - The index of the selected action.
        - The log probability of the selected action.
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
    selected_action = torch.argmax(perturbed_logits).item()

    # Calculate log probability for PPO update
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_prob = log_probs[selected_action]

    return selected_action, selected_log_prob
