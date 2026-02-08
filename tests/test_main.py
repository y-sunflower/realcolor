import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from realcolor.main import _fig_to_array, _simulate, _desaturate, as_colorblind_fig


def _make_simple_fig():
    """Helper: create a simple matplotlib figure with colored patches."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0], color="red")
    ax.plot([0, 1, 2], [1, 0, 1], color="green")
    return fig


class TestFigToArray:
    def test_returns_3d_array(self):
        fig = _make_simple_fig()
        arr = _fig_to_array(fig)
        assert arr.ndim == 3
        plt.close(fig)

    def test_has_3_channels(self):
        fig = _make_simple_fig()
        arr = _fig_to_array(fig)
        assert arr.shape[2] == 3
        plt.close(fig)

    def test_values_between_0_and_1(self):
        fig = _make_simple_fig()
        arr = _fig_to_array(fig)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0
        plt.close(fig)

    def test_dtype_is_float(self):
        fig = _make_simple_fig()
        arr = _fig_to_array(fig)
        assert np.issubdtype(arr.dtype, np.floating)
        plt.close(fig)

    def test_shape_matches_figure_size(self):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([0, 1], [0, 1])
        arr = _fig_to_array(fig)
        dpi = fig.dpi
        expected_h = int(3 * dpi)
        expected_w = int(4 * dpi)
        assert arr.shape[0] == expected_h
        assert arr.shape[1] == expected_w
        plt.close(fig)


class TestSimulate:
    @pytest.fixture()
    def sample_image(self):
        np.random.seed(42)
        return np.random.rand(10, 10, 3)

    @pytest.mark.parametrize(
        "cvd_type", ["deuteranomaly", "protanomaly", "tritanomaly"]
    )
    def test_output_shape_matches_input(self, sample_image, cvd_type):
        result = _simulate(sample_image, cvd_type)
        assert result.shape == sample_image.shape

    @pytest.mark.parametrize(
        "cvd_type", ["deuteranomaly", "protanomaly", "tritanomaly"]
    )
    def test_output_clipped_to_01(self, sample_image, cvd_type):
        result = _simulate(sample_image, cvd_type)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize(
        "cvd_type", ["deuteranomaly", "protanomaly", "tritanomaly"]
    )
    def test_severity_0_returns_original(self, sample_image, cvd_type):
        result = _simulate(sample_image, cvd_type, severity=0)
        np.testing.assert_allclose(result, sample_image, atol=1e-5)

    def test_different_cvd_types_give_different_results(self, sample_image):
        deut = _simulate(sample_image, "deuteranomaly")
        prot = _simulate(sample_image, "protanomaly")
        trit = _simulate(sample_image, "tritanomaly")
        assert not np.allclose(deut, prot)
        assert not np.allclose(deut, trit)
        assert not np.allclose(prot, trit)

    def test_dtype_is_float(self, sample_image):
        result = _simulate(sample_image, "deuteranomaly")
        assert np.issubdtype(result.dtype, np.floating)

    def test_pure_gray_unchanged(self):
        gray = np.full((5, 5, 3), 0.5)
        for cvd_type in ["deuteranomaly", "protanomaly", "tritanomaly"]:
            result = _simulate(gray, cvd_type)
            np.testing.assert_allclose(result, gray, atol=0.05)


class TestDesaturate:
    @pytest.fixture()
    def sample_image(self):
        np.random.seed(42)
        return np.random.rand(10, 10, 3)

    def test_output_shape_matches_input(self, sample_image):
        result = _desaturate(sample_image)
        assert result.shape == sample_image.shape

    def test_already_gray_stays_gray(self):
        gray = np.full((5, 5, 3), 0.5)
        result = _desaturate(gray)
        np.testing.assert_allclose(result, gray, atol=0.05)

    def test_channels_are_approximately_equal(self, sample_image):
        result = _desaturate(sample_image)
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        np.testing.assert_allclose(r, g, atol=0.05)
        np.testing.assert_allclose(r, b, atol=0.05)

    def test_dtype_is_float(self, sample_image):
        result = _desaturate(sample_image)
        assert np.issubdtype(result.dtype, np.floating)

    def test_pure_red_becomes_gray(self):
        red = np.zeros((5, 5, 3))
        red[:, :, 0] = 1.0
        result = _desaturate(red)
        assert np.allclose(result[:, :, 0], result[:, :, 1], atol=0.05)
        assert np.allclose(result[:, :, 0], result[:, :, 2], atol=0.05)


class TestAsColorblindFig:
    def test_returns_figure(self):
        fig = _make_simple_fig()
        result = as_colorblind_fig(fig)
        assert isinstance(result, plt.Figure)
        plt.close(fig)
        plt.close(result)

    def test_has_4_axes(self):
        fig = _make_simple_fig()
        result = as_colorblind_fig(fig)
        assert len(result.axes) == 4
        plt.close(fig)
        plt.close(result)

    def test_axes_titles(self):
        fig = _make_simple_fig()
        result = as_colorblind_fig(fig)
        titles = [ax.get_title() for ax in result.axes]
        assert titles == ["Deuteranopia", "Protanopia", "Tritanopia", "_Desaturated"]
        plt.close(fig)
        plt.close(result)

    def test_axes_have_no_ticks(self):
        fig = _make_simple_fig()
        result = as_colorblind_fig(fig)
        for ax in result.axes:
            assert not ax.axison
        plt.close(fig)
        plt.close(result)

    def test_custom_figsize(self):
        fig = _make_simple_fig()
        w, h = 12, 10
        result = as_colorblind_fig(fig, figsize=(w, h))
        actual_w, actual_h = result.get_size_inches()
        assert actual_w == pytest.approx(w)
        assert actual_h == pytest.approx(h)
        plt.close(fig)
        plt.close(result)

    def test_returns_new_figure(self):
        fig = _make_simple_fig()
        result = as_colorblind_fig(fig)
        assert result is not fig
        plt.close(fig)
        plt.close(result)

    def test_each_subplot_has_image(self):
        fig = _make_simple_fig()
        result = as_colorblind_fig(fig)
        for ax in result.axes:
            assert len(ax.images) == 1
        plt.close(fig)
        plt.close(result)
