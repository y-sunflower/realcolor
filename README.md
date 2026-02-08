# realcolor: how colorblind-friendly your charts are?

`realcolor` is a lightweight Python package meant to easily see colorblind people will see your charts. It simulates all kind of colorblindness and shows you the results.

<br>

## Installation

```bash
pip install git+https://github.com/y-sunflower/realcolor.git
```

<br>

## Quick start

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 2, 5], label="Group A", lw=4)
ax.plot([1, 2, 3], [2, 5, 3], label="Group B", lw=4)
ax.legend()
```

![](./img/1.png)

```python
from realcolor import as_colorblind_fig

as_colorblind_fig(fig)
```

![](./img/2.png)
