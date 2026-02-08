from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from colorspacious import cspace_convert


def _fig_to_array(fig: Figure) -> npt.NDArray[np.floating]:
    """Converts a matplotlib figure to a 3D numpy array (RGB)."""
    # Force a draw so the buffer is populated
    fig.canvas.draw()
    res = fig.canvas.buffer_rgba()
    img = np.asarray(res)

    return img[:, :, :3] / 255.0


def _simulate(
    img_array: npt.NDArray[np.floating],
    cvd_type: str,
    severity: int = 100,
) -> npt.NDArray[np.floating]:
    cvd_space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": severity}
    _simulated = cspace_convert(img_array, cvd_space, "sRGB1")

    # Force all values to stay within the [0.0, 1.0] range
    return np.clip(_simulated, 0, 1)


def _desaturate(img_array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """_simulates complete achromatopsia (greyscale)."""
    jch = cspace_convert(img_array, "sRGB1", "JCh")
    jch[..., 1] = 0  # Set Chroma to 0
    return cspace_convert(jch, "JCh", "sRGB1")


def as_colorblind_fig(fig: Figure, figsize: tuple[float, float] = (8, 8)) -> Figure:
    img = _fig_to_array(fig)

    simulations = [
        ("Deuteranopia", _simulate(img, "deuteranomaly")),
        ("Protanopia", _simulate(img, "protanomaly")),
        ("Tritanopia", _simulate(img, "tritanomaly")),
        ("_desaturated", _desaturate(img)),
    ]

    new_fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axes = axes.flatten()

    for ax, (title, sim_img) in zip(axes, simulations):
        ax.imshow(np.clip(sim_img, 0, 1))
        ax.set_title(title.title())
        ax.axis("off")

    new_fig.tight_layout()
    return new_fig
