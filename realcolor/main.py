import numpy as np
import matplotlib.pyplot as plt
from colorspacious import cspace_convert


def fig_to_array(fig):
    """Converts a matplotlib figure to a 3D numpy array (RGB)."""
    # Force a draw so the buffer is populated
    fig.canvas.draw()
    res = fig.canvas.buffer_rgba()
    img = np.asarray(res)

    return img[:, :, :3] / 255.0


def simulate(img_array, cvd_type, severity=100):
    cvd_space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": severity}
    simulated = cspace_convert(img_array, cvd_space, "sRGB1")

    # Force all values to stay within the [0.0, 1.0] range
    return np.clip(simulated, 0, 1)


def desaturate(img_array):
    """Simulates complete achromatopsia (greyscale)."""
    jch = cspace_convert(img_array, "sRGB1", "JCh")
    jch[..., 1] = 0  # Set Chroma to 0
    return cspace_convert(jch, "JCh", "sRGB1")


def as_colorblind_fig(fig, figsize=(12, 8)):
    img = fig_to_array(fig)

    simulations = [
        ("Original", img),
        ("Deuteranopia", simulate(img, "deuteranomaly")),
        ("Protanopia", simulate(img, "protanomaly")),
        ("Tritanopia", simulate(img, "tritanomaly")),
        ("Desaturated", desaturate(img)),
    ]

    # Create the grid
    new_fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, (title, sim_img) in zip(axes, simulations):
        ax.imshow(sim_img)
        ax.set_title(title)
        ax.axis("off")

    # Remove the extra subplot if 5 images are shown
    if len(simulations) < len(axes):
        axes[-1].axis("off")

    new_fig.tight_layout()
    return new_fig
