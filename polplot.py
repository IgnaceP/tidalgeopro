import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, Polygon, \
                             MultiPolygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PolygonPatch
import matplotlib
matplotlib.use('Agg')
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection

"""
def polPlot(XY, XY_inner = None, plot_title = None, show_vertices = False,
            show_vertices_labels = False, show_vertices_labels_interval = 1, plot_on_axis = False,
            empty = False, set_limits = 'True',
            vertices_color = 'darkgray', fill_color = 'silver',
            line_width = 1,
            edge_color = 'turquoise',
            bg_color = 'white', font_size = 10):
    """
    Function to plot polygon

    Args:
        XY: (Required) Shapely (Multi)Polygon, list of coordinate pairs or Numpy Array with dimension n x 2 representing the polygon to plot
        XY_inner: (Optional, defaults no None) If XY is a list or Numpy Array, indicate the inner rings by a list of Numpy n x 2 arrays or a list of coordinate pair lists
        plot_title: (Optional, defaults to None) String describing the title for the plot
        show_vertices: (Optional, defaults to False) True/False boolean to indicate whether the polygon vertices should be shown
        show_verticeslabels: (Optional, defaults to False) True/False boolean to indicate whether the labels of the vertices should be shown
        show_verticeslabels_interval: (Optional, defaults to 1) Interval of which vertices labels should be shown
        plot_on_axis: (Optional, defaults to False) False/plt-axis indicating to plot on an existing axis
        empty: (Optional, defaults to False) True/False boolean indicating whether to fill the polygon or keep it transparent
        set_limits: (Optional, defaults to True) True/False boolean to adapt the limits of the axes.
        vertices_color: (Optional, defaults to 'darkgray') Color string or code to indicate the color of the vertices.
        fill_color: (Optional, defaults to 'silver') Color string or code to indicate the color of the polygon body.
        edge_color: (Optional, defaults to 'turquoise') Color string or code to indicate the color of the polygon edges.
        bg_color: (Optional, defaults to 'white') Color string or code to indicate the color of the background
        font_size: (Optional, defaults to 10) Font size for the node annotations.

    """

    # make sure the function can handle a shapely Polygon, MultiPolygon, Numpy array or list of coordinate pairs
    if type(XY) == Polygon:
        XY_np = np.rot90(XY.exterior.xy)
        XY_inner = [i.xy for i in XY.interiors]
        pols = [[XY_np, XY_inner]]
    elif type(XY) == MultiPolygon:
        pols = []
        for xy in XY:
            XY_np =np.rot90(xy.exterior.xy)
            XY_inner = [i.xy for i in xy.interiors]
            pols.append([XY_np, XY_inner])
    else:
        pols = [[np.asarray(XY), XY_inner]]

    # plot on existing axis if indicated
    if not plot_on_axis: f, a = plt.subplots()
    else: a = plot_on_axis

    # set plot title if indicated
    if plot_title: a.set_title(plottitle, fontweight = 'bold')

    # initialize graph limits
    xmin = float('inf'); ymin = float('inf')
    xmax = float('-inf'); ymax = float('-inf')

    # search min and max coordinates along polygons to determine graph limits
    for pol in pols:
        XY_np = pol[0]
        XY = list(XY_np)
        XY_inner = pol[1]

        if np.min(XY_np[:, 0]) < xmin: xmin = np.min(XY_np[:, 0])
        if np.max(XY_np[:, 0]) > xmax: xmax = np.max(XY_np[:, 0])
        if np.min(XY_np[:, 1]) < ymin: ymin = np.min(XY_np[:, 1])
        if np.max(XY_np[:, 1]) > ymax: ymax = np.max(XY_np[:, 1])

        # determine fill colors
        if empty:
            col = 'none'
            colin = 'none'
        else:
            col = fill_color
            colin = bg_color

        # create a matplotlib patch representing the polygon
        if XY_np.shape[0] <= 2: XY_np = np.rot90(XY_np)
        pol = PolygonPatch(XY_np, facecolor = col, edgecolor = edge_color, lw = line_width) # matplotlib.patches.Polygon

        # if there are inner rings, plot them
        if XY_inner is not None:
            pols_inner = []
            for i in XY_inner:
                i = np.asarray(i)
                if i.shape[0] <= 2: i = np.rot90(i)
                pol_inner = PolygonPatch(i, facecolor=colin, edgecolor = edge_color, lw = line_width)  # matplotlib.patches.Polygon
                pols_inner.append(pol_inner)

        # add the patches to the plot
        a.add_patch(pol)
        if XY_inner is not None:
            for pol_inner in pols_inner:
                a.add_patch(pol_inner)

        # if indicated plot the vertices
        if show_vertices:
            a.scatter(XY_np[:,0], XY_np[:,1], s = 4, edgecolor = vertices_color, facecolor = vertices_color, zorder = 10)
            if XY_inner:
                for i in XY_inner:
                    a.scatter(np.asarray(i)[:, 0], np.asarray(i)[:, 1], s=4, edgecolor= vertices_color, facecolor = vertices_color, zorder = 10)

            # if indicated, show the vertices labels
            if show_vertices_labels:
                # counter
                t = 1
                for x,y in XY:
                    # only plot the vertices labels on a given interval
                    if t%show_vertices_labels_interval == 0: a.annotate(str(t),(x,y), size = font_size)
                    t += 1
                if XY_inner:
                    for i in XY_inner:
                        for j in range(len(i)):
                            x,y = i[j][0],i[j][1]
                            # only plot the vertices labels on a given interval
                            if t%show_vertices_labels_interval == 0: a.annotate(str(t), (x, y), size = font_size)
                            t += 1

    # set the axes limits, if indicated
    if set_limits:
        a.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))
        a.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))
        a.set_aspect('equal')

    # set background color
    a.set_facecolor(bg_color)

"""

# Plots a Polygon to pyplot `ax`
def plotPolygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection

def plotPolygons(ax,mpol,**kwargs):
    collections = []
    for pol in list(mpol.geoms):
        col = plotPolygon(ax, pol, **kwargs)
        collections.append(col)
    return collections