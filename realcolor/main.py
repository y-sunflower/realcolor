from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from colorspacious import cspace_convert


def _fig_to_array(fig: Figure) -> npt.NDArray[np.floating]:
    """Converts a matplotlib figure to a 3D numpy array (RGB)."""
    # Force a draw so the buffer is populated
    fig.canvas.draw()
    res = fig.canvas.buffer_rgba()  # ty: ignore[unresolved-attribute]
    img = np.asarray(res)

    return img[:, :, :3] / 255.0


def _simulate(img_array, cvd_type, severity=100):
    cvd_space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": severity}
    simulated = cspace_convert(img_array, cvd_space, "sRGB1")

    # Force all values to stay within the [0.0, 1.0] range
    return np.clip(simulated, 0, 1)


def _desaturate(img_array):
    """Simulates complete achromatopsia (greyscale)."""
    jch = cspace_convert(img_array, "sRGB1", "JCh")
    jch[..., 1] = 0  # Set Chroma to 0
    return cspace_convert(jch, "JCh", "sRGB1")


def simulate_colorblindness(
    plot_object,
    figsize: tuple[float, float] = (8, 8),
) -> Figure:
    if not isinstance(plot_object, Figure):
        fig = plot_object.draw()
    else:
        fig = plot_object
    img = _fig_to_array(fig)

    simulations = [
        ("Deuteranopia", _simulate(img, "deuteranomaly")),
        ("Protanopia", _simulate(img, "protanomaly")),
        ("Tritanopia", _simulate(img, "tritanomaly")),
        ("Desaturated", _desaturate(img)),
    ]

    new_fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    axes = axes.flatten()

    for ax, (title, sim_img) in zip(axes, simulations):
        ax.imshow(np.clip(sim_img, 0, 1))
        ax.set_title(title)
        ax.axis("off")

    new_fig.tight_layout()
    return new_fig
