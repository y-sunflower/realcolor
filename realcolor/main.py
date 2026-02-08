from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from realcolor.colors import (
    _LUMINANCE_WEIGHTS,
    _machado_matrix,
    _srgb_to_linear,
    _linear_to_srgb,
)


def _fig_to_array(fig: Figure) -> npt.NDArray[np.floating]:
    """Converts a matplotlib figure to a 3D numpy array (RGB)."""
    # Force a draw so the buffer is populated
    fig.canvas.draw()
    res = fig.canvas.buffer_rgba()  # ty: ignore[unresolved-attribute]
    img = np.asarray(res)

    return img[:, :, :3] / 255.0


def _simulate(
    img_array: npt.NDArray[np.floating],
    cvd_type: str,
    severity: int = 100,
) -> npt.NDArray[np.floating]:
    matrix = _machado_matrix(cvd_type, severity)
    linear = _srgb_to_linear(img_array)
    simulated = np.einsum("...j,ij->...i", linear, matrix)
    return np.clip(_linear_to_srgb(simulated), 0, 1)


def _desaturate(img_array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Simulates complete achromatopsia (greyscale)."""
    linear = _srgb_to_linear(img_array)
    luminance = np.dot(linear[..., :3], _LUMINANCE_WEIGHTS)
    gray_linear = np.stack([luminance] * 3, axis=-1)
    return np.clip(_linear_to_srgb(gray_linear), 0, 1)


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
