"""
Visualization tools for simulation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_surface(
    height_profile: npt.NDArray[np.float64],
    title: str = "Surface Profile",
    save_path: str | None = None,
) -> None:
    """
    Plot 3D surface profile.

    Args:
        height_profile: 2D array of heights.
        title: Plot title.
        save_path: Path to save figure (None = display).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    nx, ny = height_profile.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)  # pylint: disable=invalid-name

    ax.plot_surface(X, Y, height_profile.T, cmap="viridis")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_roughness_evolution(
    times: npt.NDArray[np.float64],
    roughnesses: npt.NDArray[np.float64],
    title: str = "Roughness Evolution",
    save_path: str | None = None,
) -> None:
    """
    Plot roughness vs time.

    Args:
        times: Time array.
        roughnesses: Roughness array.
        title: Plot title.
        save_path: Path to save figure.
    """
    plt.figure(figsize=(8, 6))
    plt.loglog(times, roughnesses, "o-")
    plt.xlabel("Time (s)")
    plt.ylabel("Roughness W(L,t)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
