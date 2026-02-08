# realcolor: simulate colorblindness in Python charts

`realcolor` is a lightweight Python package designed to show **how colorblind people see your graphs**. It simulates all types of colorblindness (deuteranopia, protanopia, tritanopia) by showing you your graphic as seen by a colorblind person. It works with _matplotlib and everything built on top of it_ (seaborn, plotnine, etc).

> [!NOTE]
> Colorblindness affects up to 1 in 12 males (8%) and 1 in 200 females (0.5%)[^1]

<br>

## Installation

```bash
pip install realcolor
```

<br>

## Quick start

### Matplotlib

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 2, 5], label="Group A", lw=4)
ax.plot([1, 2, 3], [2, 5, 3], label="Group B", lw=4)
ax.legend()
```

![](./img/1.png)

```python
from realcolor import simulate_colorblindness

simulate_colorblindness(fig)
```

![](./img/2.png)

### Plotnine

```python
from plotnine import ggplot, geom_point, aes
from plotnine.data import anscombe_quartet

ggp = ggplot(anscombe_quartet, aes(x="x", y="y", color="dataset")) + geom_point(size=10)
```

![](./img/3.png)

```python
from realcolor import simulate_colorblindness

simulate_colorblindness(ggp)
```

![](./img/4.png)

> [!TIP]
> Looking for support of other data visualization libraries? [Open an issue](https://github.com/y-sunflower/realcolor/issues).

<br>

[^1]: Deane B. Judd, "Facts of Color-Blindness\*," J. Opt. Soc. Am. 33, 294-307 (1943)
