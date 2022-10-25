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
