import itertools

from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from colorspacious import cspace_convert, deltaE


VALID_KINDS = ("deuteranopia", "protanopia", "tritanopia", "desaturated")

_CVD_TYPE_MAP = {
    "deuteranopia": "deuteranomaly",
    "protanopia": "protanomaly",
    "tritanopia": "tritanomaly",
}

_SCORE_KINDS = ("deuteranopia", "protanopia", "tritanopia")

_DELTAE_THRESHOLD = 25.0


def _fig_to_array(fig: Figure) -> npt.NDArray[np.floating]:
    """Converts a matplotlib figure to a 3D numpy array (RGB)."""
    # Force a draw so the buffer is populated
    fig.canvas.draw()
    res = fig.canvas.buffer_rgba()  # ty: ignore[unresolved-attribute]
    img = np.asarray(res)

    return img[:, :, :3] / 255.0


def _simulate(img_array, cvd_type, severity):
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
    severity: int | float = 100,
    kind: str | None = None,
) -> Figure:
    if kind is not None and kind not in VALID_KINDS:
        raise ValueError(
            f"Invalid kind {kind!r}. Must be one of {VALID_KINDS} or None for all."
        )

    if not isinstance(plot_object, Figure):
        fig = plot_object.draw()
    else:
        fig = plot_object
    img = _fig_to_array(fig)

    all_simulations = {
        "deuteranopia": ("Deuteranopia", _simulate(img, "deuteranomaly", severity)),
        "protanopia": ("Protanopia", _simulate(img, "protanomaly", severity)),
        "tritanopia": ("Tritanopia", _simulate(img, "tritanomaly", severity)),
        "desaturated": ("Desaturated", _desaturate(img)),
    }

    if kind is not None:
        title, sim_img = all_simulations[kind]
        new_fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.clip(sim_img, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    else:
        simulations = list(all_simulations.values())
        new_fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        for ax, (title, sim_img) in zip(axes, simulations):
            ax.imshow(np.clip(sim_img, 0, 1))
            ax.set_title(title)
            ax.axis("off")

    new_fig.tight_layout()
    return new_fig


def _colors_to_array(colors):
    """Parse a list of colors to a (N, 1, 3) sRGB1 numpy array."""
    rgb_list = [to_rgb(c) for c in colors]
    return np.array(rgb_list).reshape(-1, 1, 3)


def _min_pairwise_deltaE(simulated_colors):
    """Return (min_deltaE, (i, j)) for the closest pair of simulated colors.

    simulated_colors is (N, 1, 3) sRGB1 array.
    """
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
    """Result of a colorblind friendliness score.

    Attributes
    ----------
    overall : float
        Minimum score across all CVD types (0-100).
    deuteranopia : dict
        Score details for deuteranopia.
    protanopia : dict
        Score details for protanopia.
    tritanopia : dict
        Score details for tritanopia.
    """

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
    colors: list[str | tuple[float, float, float]],
    severity: int | float = 100,
) -> ColorblindScoreResult:
    """Score how distinguishable a set of colors is under colorblind simulation.

    Parameters
    ----------
    colors : list of str or RGB tuples
        Two or more colors in any format matplotlib understands.
    severity : int or float, default 100
        CVD severity from 0 (no deficiency) to 100 (full).

    Returns
    -------
    ColorblindScoreResult with overall score and per-type details accessible
    as attributes (e.g. result.overall, result.deuteranopia).
    """
    if len(colors) < 2:
        raise ValueError("At least 2 colors are required.")

    color_array = _colors_to_array(colors)

    # Normalise every input color to its hex representation for worst_pair output
    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(
            int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
        )
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
