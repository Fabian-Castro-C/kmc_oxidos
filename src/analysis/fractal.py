"""
Fractal dimension calculation using box-counting method.
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import linregress


def calculate_fractal_dimension(
    height_profile: npt.NDArray[np.float64], min_box_size: int = 2, max_box_size: int | None = None
) -> float:
    """
    Calculate fractal dimension using box-counting method.

    Args:
        height_profile: 2D array of surface heights.
        min_box_size: Minimum box size.
        max_box_size: Maximum box size (None = auto).

    Returns:
        Fractal dimension D_f.
    """
    nx, ny = height_profile.shape

    # Set default max_box_size if None
    effective_max_box_size = max_box_size if max_box_size is not None else min(nx, ny) // 4

    # Box sizes to test
    box_sizes: list[int] = []
    counts: list[int] = []

    for box_size in range(
        min_box_size,
        effective_max_box_size + 1,
        max(1, (effective_max_box_size - min_box_size) // 10),
    ):
        n_boxes = 0

        for i in range(0, nx - box_size + 1, box_size):
            for j in range(0, ny - box_size + 1, box_size):
                window = height_profile[i : i + box_size, j : j + box_size]
                h_range = np.max(window) - np.min(window)

                # Count boxes needed to cover height range
                if h_range > 0:
                    n_boxes += int(np.ceil(h_range / box_size)) + 1

        box_sizes.append(box_size)
        counts.append(n_boxes)

    # Fit: log(N) = D_f * log(1/ε) + const
    log_sizes = np.log(1.0 / np.array(box_sizes))
    log_counts = np.log(np.array(counts))

    # Get linear regression result
    result = linregress(log_sizes, log_counts)

    # Extract slope (Pylance may show warnings, but this is correct)
    return float(result.slope)  # type: ignore[attr-defined]
