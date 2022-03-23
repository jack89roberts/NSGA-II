import contextily as ctx
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


def set_matplotlib_style():
    plt.style.use("fivethirtyeight")
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["axes.grid"] = False


def load_oa_data():
    """Load Camden output area data.

    Returns
    -------
    pandas.DataFrame
        Camden output area shapes and population statistics
    """
    oa_stats = gpd.read_file("data/oa_shape")
    oa_stats = oa_stats.set_index("oa11cd").sort_index()

    oa_centroids = pd.read_csv("data/centroids.csv")
    oa_centroids = oa_centroids.set_index("oa11cd").sort_index()
    oa_stats = oa_stats.join(oa_centroids).sort_index()

    oa_population = pd.read_csv("data/population_total.csv")
    oa_population = oa_population.set_index("oa11cd").sort_index()
    oa_population = oa_population["population"]
    oa_population.name = "total_population"
    oa_stats = oa_stats.join(oa_population).sort_index()
    oa_stats["total_density"] = oa_stats["total_population"] / (
        oa_stats.geometry.area / 1e6
    )

    oa_ages = pd.read_csv("data/population_ages.csv")
    oa_ages = oa_ages.set_index("oa11cd").sort_index()
    oa_ages.columns = oa_ages.columns.astype(int)
    over65 = oa_ages.loc[:, oa_ages.columns > 65].sum(axis=1)
    over65.name = "over65"
    oa_stats = oa_stats.join(over65).sort_index()
    oa_stats["over65_density"] = oa_stats["over65"] / (oa_stats.geometry.area / 1e6)

    workers = pd.read_csv("data/workplace.csv")
    workers = workers.set_index("oa11cd").sort_index()
    workers = workers["workers"]
    oa_stats = oa_stats.join(workers).sort_index()
    oa_stats["workers_density"] = oa_stats["workers"] / (oa_stats.geometry.area / 1e6)

    return oa_stats


oa_stats = load_oa_data()


def plot_oa(
    values=None,
    show_bl=True,
    vmax=None,
    ax=None,
    figsize=(10, 10),
    legend=False,
    alpha=0.75,
    cax_label=None,
    title=None,
    basemap=True,
    edgecolor=None,
):
    """Plot a map of Camden's output areas.

    Parameters
    ----------
    values : np.array or pd.Series, optional
        Values to use for output area colour scale, by default None
    show_bl : bool, optional
        Show the location of the British Library, by default True
    vmax : _type_, optional
       Max value for the output area colour scale, by default None
    ax : plt.axes, optional
       Axes to plot to, by default None
    figsize : tuple, optional
        Size of figure to create if ax is None, by default (10, 10)
    legend : bool, optional
        Show a legend, by default False
    alpha : float, optional
        Transparency of otuput area shading, by default 0.75
    cax_label : str, optional
        Label for colorbar, by default None
    title : str, optional
        Figure title, by default None
    basemap : bool, optional
        Show the basemap, by default True
    edgecolor : str or list, optional
        Colour of output area boundary lines, by default None
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if cax_label:
            cax.set_title(cax_label)
    else:
        cax = None

    if values is None:
        oa_stats.plot(ax=ax, facecolor="None", edgecolor=edgecolor or "k", linewidth=1)
    else:
        oa_stats["values"] = values
        oa_stats.plot(
            ax=ax,
            column="values",
            vmax=vmax,
            legend=legend,
            alpha=alpha,
            cax=cax,
            edgecolor=edgecolor,
        )

    if show_bl:
        bl = (529906, 182892)  # British library
        ax.plot(bl[0], bl[1], "ro", markersize=16, label="British Library")
        ax.legend()

    if basemap:
        ctx.add_basemap(
            ax,
            source="http://a.tile.stamen.com/toner/{z}/{x}/{y}.png",
            crs=oa_stats.crs.to_epsg(),
        )

    if title:
        ax.set_title(title)

    ax.axis("off")


def singledim(func):
    """
    Decorator that modifies the input function func, a function
    of two arguments usually expecting a 2D array as its second
    argument, to work if a 1D array is given instead.
    """

    def _compute(self, x):
        if x.ndim == 1:
            result = func(self, x[np.newaxis, :])
            return result[0, :]
        return func(self, x)

    return _compute


def plot_hypervolume(points, ref_point=(0, 0)):
    """Plot the 2D hypervolume for am array of 2 objective fitness values (points)

    Parameters
    ----------
    points : np.array
        Fitness values of shape (n, 2)
    ref_point : tuple, optional
        Reference point for hypervolume calculation, by default (0, 0)
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(points[:, 0], points[:, 1], "ro", label="Fitness", markersize=12)

    ref_point = [0, 0]
    ax.plot(0, 0, "kx", markersize=18, label="Reference point")

    coords = [ref_point]
    for i in range(len(points)):
        coords += [
            [coords[-1][0], points[i, 1]],
            [points[i, 0], points[i, 1]],
        ]
    coords += [[points[i, 0], ref_point[1]]]

    ax.add_patch(
        Polygon(
            np.array(coords),
            facecolor="orange",
            edgecolor="k",
            label="Hypervolume",
            linewidth=2,
        )
    )
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.legend()


def plot_circle_fitness(pop, fitness_fn, radius, suptitle=None):
    """Show fitness values and candidate solution positions for the quarter-
    circle multi-objective fitness example.

    Parameters
    ----------
    pop : np.array
        (n, 2) array of candidate solutions (x and y positions)
    fitness_fn : callable
        Fitness function - takes popultation as only argumrnt
    radius : float
        Radius of circle, outside which fitness is 0
    suptitle : _type_, optional
        Figure title, by default None
    """
    delta = 0.1
    x = np.arange(0, radius + 2, delta)
    X, Y = np.meshgrid(x, x)
    cf = fitness_fn(np.array([X.flatten(), Y.flatten()]).T)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    obj_label = ["x", "y"]
    for i in range(2):
        _ = ax[i].imshow(
            cf[:, i].reshape(len(x), len(x)),
            origin="lower",
            cmap="Blues",
            aspect="auto",
            extent=[X.min() - delta, X.max() + delta, Y.min() - delta, Y.max() + delta],
        )
        ax[i].scatter(
            pop[:, 0], pop[:, 1], color="orange", s=64
        )
        ax[i].set_title(
            f"Objective {i + 1}\n${obj_label[i]}^2$ if $(x^2 + y^2) \leq r^2$ else 0"
        )
        ax[i].set_xlabel("x")
        ax[i].set_xticks([])
        ax[i].set_ylabel("y")
        ax[i].set_yticks([])

    ax[2].scatter(fitness_fn(pop)[:, 0], fitness_fn(pop)[:, 1])
    ax[2].set_title("Fitness")
    ax[2].set_xlim([0, radius**2])
    ax[2].set_ylim([0, radius**2])
    ax[2].set_xlabel("Objective 1")
    ax[2].set_ylabel("Objective 2")
    fig.tight_layout()
    if suptitle:
        fig.suptitle(suptitle, y=1.05)
