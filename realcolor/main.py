import itertools
import math
import numbers
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from colorspacious import cspace_convert, deltaE

VALID_KINDS = ("deuteranopia", "protanopia", "tritanopia", "desaturated")

_CVD_TYPE_MAP = {
    "deuteranopia": "deuteranomaly",
    "protanopia": "protanomaly",
    "tritanopia": "tritanomaly",
}

_SCORE_KINDS = ("deuteranopia", "protanopia", "tritanopia")

_DELTAE_THRESHOLD = 25.0

_COLOR_NAMES = {
    "black": (0.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "fuchsia": (1.0, 0.0, 1.0),
    "gray": (0.5019607843, 0.5019607843, 0.5019607843),
    "green": (0.0, 0.5019607843, 0.0),
    "grey": (0.5019607843, 0.5019607843, 0.5019607843),
    "lime": (0.0, 1.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "maroon": (0.5019607843, 0.0, 0.0),
    "navy": (0.0, 0.0, 0.5019607843),
    "olive": (0.5019607843, 0.5019607843, 0.0),
    "orange": (1.0, 0.6470588235, 0.0),
    "purple": (0.5019607843, 0.0, 0.5019607843),
    "red": (1.0, 0.0, 0.0),
    "silver": (0.7529411765, 0.7529411765, 0.7529411765),
    "teal": (0.0, 0.5019607843, 0.5019607843),
    "white": (1.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def _validate_rgb(values) -> tuple[float, float, float]:
    if len(values) not in (3, 4):
        raise ValueError("RGB colors must contain 3 or 4 values.")

    try:
        rgb = tuple(float(value) for value in values[:3])
    except (TypeError, ValueError) as error:
        raise ValueError("RGB values must be numbers between 0 and 1.") from error

    if not all(math.isfinite(value) and 0 <= value <= 1 for value in rgb):
        raise ValueError("RGB values must be finite numbers between 0 and 1.")
    return rgb[0], rgb[1], rgb[2]


def _parse_color(color) -> tuple[float, float, float]:
    """Parse a color into an sRGB tuple without a plotting dependency."""
    if isinstance(color, str):
        value = color.strip().lower()
        if value in _COLOR_NAMES:
            return _COLOR_NAMES[value]

        if value.startswith("#"):
            hex_value = value[1:]
            if len(hex_value) in (3, 4):
                hex_value = "".join(char * 2 for char in hex_value)
            if len(hex_value) not in (6, 8):
                raise ValueError(f"Invalid color value: {color!r}")
            try:
                channels = tuple(
                    int(hex_value[index : index + 2], 16) / 255
                    for index in range(0, len(hex_value), 2)
                )
            except ValueError as error:
                raise ValueError(f"Invalid color value: {color!r}") from error
            return _validate_rgb(channels)

        try:
            grayscale = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid color value: {color!r}") from error
        if math.isfinite(grayscale) and 0 <= grayscale <= 1:
            return (grayscale, grayscale, grayscale)
        raise ValueError(f"Invalid color value: {color!r}")

    if isinstance(color, (tuple, list, np.ndarray)):
        return _validate_rgb(color)

    raise ValueError(f"Invalid color value: {color!r}")


def _fig_to_array(fig) -> npt.NDArray[np.floating]:
    """Convert a Matplotlib figure to RGB without importing Matplotlib eagerly."""
    from realcolor.matplotlib import fig_to_array

    return fig_to_array(fig)


def _simulate(img_array, cvd_type, severity):
    cvd_space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": severity}
    simulated = cspace_convert(img_array, cvd_space, "sRGB1")
    return np.clip(simulated, 0, 1)


def _desaturate(img_array):
    """Simulate complete achromatopsia (greyscale)."""
    jch = cspace_convert(img_array, "sRGB1", "JCh")
    jch[..., 1] = 0
    return cspace_convert(jch, "JCh", "sRGB1")


def _validate_dimensions(width, height) -> tuple[float, float]:
    for name, value in (("width", width), ("height", height)):
        if (
            not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive number.")
    return float(width), float(height)


def _validate_figsize(figsize) -> tuple[float, float]:
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError("figsize must contain a positive width and height.")
    return _validate_dimensions(figsize[0], figsize[1])


def _validate_kind(kind: str | None) -> None:
    if kind is not None and kind not in VALID_KINDS:
        raise ValueError(
            f"Invalid kind {kind!r}. Must be one of {VALID_KINDS} or None for all."
        )


def _all_simulations(img, severity: float) -> dict[str, tuple[str, npt.NDArray]]:
    return {
        "deuteranopia": ("Deuteranopia", _simulate(img, "deuteranomaly", severity)),
        "protanopia": ("Protanopia", _simulate(img, "protanomaly", severity)),
        "tritanopia": ("Tritanopia", _simulate(img, "tritanomaly", severity)),
        "desaturated": ("Desaturated", _desaturate(img)),
    }


def _is_plotly_figure(plot_object) -> bool:
    try:
        from plotly.graph_objects import Figure
    except ModuleNotFoundError:
        return False
    return isinstance(plot_object, Figure)


def simulate_colorblindness(
    plot_object,
    *,
    figsize: tuple[float, float] | None = None,
    width: float | None = None,
    height: float | None = None,
    severity: float = 100,
    kind: str | None = None,
):
    """Create a colorblindness simulation for a supported plotting figure.

    Matplotlib and Matplotlib-based objects use ``figsize`` in inches. Plotly
    figures use native ``width`` and ``height`` values in pixels.
    """
    _validate_kind(kind)

    if _is_plotly_figure(plot_object):
        if figsize is not None:
            raise TypeError("Plotly figures use width and height, not figsize.")
        width, height = _validate_dimensions(
            800 if width is None else width,
            800 if height is None else height,
        )
        from realcolor.plotly import simulate_plotly

        return simulate_plotly(
            plot_object,
            width=width,
            height=height,
            severity=severity,
            kind=kind,
        )

    if width is not None or height is not None:
        raise TypeError("Matplotlib figures use figsize, not width and height.")
    figsize = _validate_figsize((8, 8) if figsize is None else figsize)
    try:
        from realcolor.matplotlib import simulate_matplotlib
    except ModuleNotFoundError as error:
        module_name = error.name or ""
        if module_name == "matplotlib" or module_name.startswith("matplotlib."):
            raise ImportError(
                "Matplotlib support requires the 'matplotlib' optional dependency. "
                "Install it with `pip install realcolor[matplotlib]`."
            ) from error
        raise

    return simulate_matplotlib(
        plot_object,
        figsize=figsize,
        severity=severity,
        kind=kind,
    )


def _colors_to_array(colors):
    """Parse colors to a (N, 1, 3) sRGB1 numpy array."""
    rgb_list = [_parse_color(color) for color in colors]
    return np.array(rgb_list).reshape(-1, 1, 3)


def _min_pairwise_deltaE(simulated_colors):
    """Return (min_deltaE, (i, j)) for the closest pair of simulated colors."""
    n = simulated_colors.shape[0]
    min_de = float("inf")
    worst_pair = (0, 1)
    for i, j in itertools.combinations(range(n), 2):
        de = float(
            deltaE(simulated_colors[i, 0], simulated_colors[j, 0], input_space="sRGB1")
        )
        if de < min_de:
            min_de = de
            worst_pair = (i, j)
    return min_de, worst_pair


class ColorblindScoreResult:
    """Result of a colorblind friendliness score."""

    deuteranopia: dict
    protanopia: dict
    tritanopia: dict

    def __init__(self, overall: float, per_type: dict[str, dict]):
        self.overall = overall
        self.deuteranopia = per_type["deuteranopia"]
        self.protanopia = per_type["protanopia"]
        self.tritanopia = per_type["tritanopia"]

    def __repr__(self):
        return (
            f"ColorblindScoreResult(overall={self.overall}, "
            f"deuteranopia={self.deuteranopia}, "
            f"protanopia={self.protanopia}, "
            f"tritanopia={self.tritanopia})"
        )


def colorblind_score(
    colors: Sequence[str | tuple[float, float, float]],
    severity: float = 100,
) -> ColorblindScoreResult:
    """Score how distinguishable a set of colors is under colorblind simulation."""
    if len(colors) < 2:
        raise ValueError("At least 2 colors are required.")

    color_array = _colors_to_array(colors)
    hex_colors = [
        f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
        for r, g, b in color_array[:, 0, :]
    ]

    per_type: dict[str, dict] = {}
    for kind in _SCORE_KINDS:
        cvd_type = _CVD_TYPE_MAP[kind]
        simulated = _simulate(color_array, cvd_type, severity)
        min_de, (i, j) = _min_pairwise_deltaE(simulated)
        score = min(100.0, min_de / _DELTAE_THRESHOLD * 100.0)
        per_type[kind] = {
            "score": round(score, 1),
            "min_deltaE": round(min_de, 1),
            "worst_pair": (hex_colors[i], hex_colors[j]),
        }

    overall = min(info["score"] for info in per_type.values())
    return ColorblindScoreResult(overall=overall, per_type=per_type)
