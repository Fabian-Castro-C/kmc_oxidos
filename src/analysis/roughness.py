"""
Roughness analysis and Family-Vicsek scaling.

Calculate W(L,t) and determine scaling exponents α and β.
"""

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


def calculate_roughness(
    height_profile: npt.NDArray[np.float64], window_size: int | None = None
) -> float:
    """
    Calculate interface width (roughness) W(L,t).

    W(L,t) = sqrt(<[h(x,t) - <h>]²>)

    Args:
        height_profile: 2D array of surface heights.
        window_size: Size of window for local roughness (None = global).

    Returns:
        Roughness value.
    """
    if window_size is None:
        mean_height = np.mean(height_profile)
        return float(np.sqrt(np.mean((height_profile - mean_height) ** 2)))

    # Local roughness calculation
    roughnesses = []
    nx, ny = height_profile.shape

    for i in range(0, nx - window_size + 1, window_size):
        for j in range(0, ny - window_size + 1, window_size):
            window = height_profile[i : i + window_size, j : j + window_size]
            mean_h = np.mean(window)
            w = np.sqrt(np.mean((window - mean_h) ** 2))
            roughnesses.append(w)

    return float(np.mean(roughnesses))


def fit_family_vicsek(
    times: npt.NDArray[np.float64],
    roughnesses: npt.NDArray[np.float64],
    system_size: float,
) -> dict[str, float]:
    """
    Fit Family-Vicsek scaling relation to determine α and β.

    W(L,t) = L^α f(t/L^z) where f(u) ~ u^β for small u

    Args:
        times: Array of time points.
        roughnesses: Array of roughness values.
        system_size: System size L.

    Returns:
        Dictionary with fitted parameters: {'alpha': α, 'beta': β}.
    """

    # For early times: W ~ t^β
    def power_law(t: npt.NDArray[np.float64], beta: float, a: float) -> npt.NDArray[np.float64]:
        return a * np.power(t, beta)

    # Fit to early time data
    early_idx = len(times) // 3
    result = curve_fit(power_law, times[:early_idx], roughnesses[:early_idx])
    popt = result[0]
    beta = popt[0]

    # For late times: W ~ L^α (saturated)
    alpha = np.log(roughnesses[-1]) / np.log(system_size)

    return {"alpha": float(alpha), "beta": float(beta)}
