from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from realcolor.main import _all_simulations


def fig_to_array(fig):
    """Convert a Plotly figure to a 3D RGB array using its static renderer."""
    image_bytes = fig.to_image(format="png")
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def simulate_plotly(
    plot_object,
    *,
    width: float,
    height: float,
    severity: float,
    kind: str | None,
):
    """Render simulations as a Plotly figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    img = fig_to_array(plot_object)
    all_simulations = _all_simulations(img, severity)
    simulations = (
        [all_simulations[kind]] if kind is not None else list(all_simulations.values())
    )

    is_grid = kind is None
    rows, columns = (2, 2) if is_grid else (1, 1)
    result = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[title for title, _ in simulations],
        horizontal_spacing=0.04,
        vertical_spacing=0.08 if is_grid else 0.0,
    )

    for index, (_, sim_img) in enumerate(simulations):
        row = index // columns + 1
        column = index % columns + 1
        result.add_trace(
            go.Image(z=(np.clip(sim_img, 0, 1) * 255).astype(np.uint8)),
            row=row,
            col=column,
        )

    result.update_layout(
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "t": 40 if is_grid else 30, "b": 0},
    )
    result.update_xaxes(visible=False)
    result.update_yaxes(visible=False, autorange="reversed")
    return result
