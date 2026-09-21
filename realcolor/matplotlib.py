from __future__ import annotations

import numpy as np
import numpy.typing as npt

from realcolor.main import _all_simulations


def fig_to_array(fig) -> npt.NDArray[np.float64]:
    """Convert a Matplotlib figure to a 3D RGB array."""
    if not hasattr(fig.canvas, "buffer_rgba"):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        FigureCanvasAgg(fig)
    fig.canvas.draw()
    res = fig.canvas.buffer_rgba()
    img = np.asarray(res, dtype=np.float64)
    return img[:, :, :3] / 255.0


def _source_figure(plot_object):
    from matplotlib.figure import Figure

    if isinstance(plot_object, Figure):
        return plot_object
    if hasattr(plot_object, "draw"):
        return plot_object.draw()
    raise TypeError(
        "Unsupported plot object. Expected a Matplotlib figure, a plotnine "
        "plot, or a Plotly figure."
    )


def simulate_matplotlib(
    plot_object,
    *,
    figsize: tuple[float, float],
    severity: float,
    kind: str | None,
):
    """Render simulations as a Matplotlib figure."""
    import matplotlib.pyplot as plt

    fig = _source_figure(plot_object)
    img = fig_to_array(fig)
    all_simulations = _all_simulations(img, severity)

    if kind is not None:
        title, sim_img = all_simulations[kind]
        new_fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.clip(sim_img, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    else:
        new_fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        for ax, (title, sim_img) in zip(axes.flat, all_simulations.values()):
            ax.imshow(np.clip(sim_img, 0, 1))
            ax.set_title(title)
            ax.axis("off")

    new_fig.tight_layout()
    return new_fig
