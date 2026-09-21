from io import BytesIO
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest
from PIL import Image

from realcolor import simulate_colorblindness
from realcolor.main import VALID_KINDS


def _make_plotly_figure():
    return go.Figure(
        data=[
            go.Scatter(x=[0, 1, 2], y=[0, 1, 0], line={"color": "red"}),
            go.Scatter(x=[0, 1, 2], y=[1, 0, 1], line={"color": "green"}),
        ]
    )


def _source_png():
    pixels = np.zeros((24, 32, 3), dtype=np.uint8)
    pixels[:, :16] = [255, 0, 0]
    pixels[:, 16:] = [0, 128, 0]
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


class TestPlotlyBackend:
    def test_returns_plotly_figure(self):
        fig = _make_plotly_figure()
        with patch.object(fig, "to_image", return_value=_source_png()):
            result = simulate_colorblindness(fig)
        assert isinstance(result, go.Figure)

    def test_all_kinds_use_a_2_by_2_grid(self):
        fig = _make_plotly_figure()
        with patch.object(fig, "to_image", return_value=_source_png()):
            result = simulate_colorblindness(fig)

        assert len(result.data) == 4
        assert [trace.xaxis for trace in result.data] == ["x", "x2", "x3", "x4"]
        assert [annotation.text for annotation in result.layout.annotations] == [
            "Deuteranopia",
            "Protanopia",
            "Tritanopia",
            "Desaturated",
        ]

    def test_each_panel_contains_an_image(self):
        fig = _make_plotly_figure()
        with patch.object(fig, "to_image", return_value=_source_png()):
            result = simulate_colorblindness(fig)

        assert all(trace.type == "image" for trace in result.data)

    @pytest.mark.parametrize("kind", VALID_KINDS)
    def test_single_kind_returns_one_panel(self, kind):
        fig = _make_plotly_figure()
        with patch.object(fig, "to_image", return_value=_source_png()):
            result = simulate_colorblindness(fig, kind=kind)

        assert len(result.data) == 1
        assert len(result.layout.annotations) == 1
        assert result.layout.annotations[0].text == kind.title()

    def test_custom_native_dimensions(self):
        fig = _make_plotly_figure()
        with patch.object(fig, "to_image", return_value=_source_png()):
            result = simulate_colorblindness(fig, width=1200, height=900)

        assert result.layout.width == 1200
        assert result.layout.height == 900

    def test_figsize_is_rejected_for_plotly(self):
        fig = _make_plotly_figure()
        with pytest.raises(TypeError, match="width and height"):
            simulate_colorblindness(fig, figsize=(12, 9))
