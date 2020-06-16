""" Rasterize

Module to rasterize a shapely polygon to a Numpy Array

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""

import numpy as np
from rasterio.features import rasterize as rasterioRasterize
from shapely.geometry import Polygon, MultiPolygon, Point

def rasterize(mpol, res = 1, return_minmaxs = False):
    """
    Function to Rasterize

    author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        mpol: (Required) Shapely Multi(Polygon) to rasterize
        res: (Optional, defaults to 1) float/integer representing the cell size
        return_minmaxs: (Optional, defaults to False) True/False boolean to return the coordinates of the min and max of polygon and the coordinates of the top left corner


    Returns:
        Numpy array with dimensions n x m with 1 on cells covering the original polygons
        Coordinate pair with the coordinates of the left top corner of the raster
    """

    # make sure it can handle both a Polygon and a MultiPolygon
    if type(mpol) == Polygon:
        mpol = MultiPolygon([mpol])

    # initialize limits
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')

    # search min and max coordinates along polygons to determine graph limits
    for pol in mpol:
        xy = np.rot90(pol.exterior.xy)

        if np.min(xy[:, 0]) < xmin: xmin = np.min(xy[:, 0])
        if np.max(xy[:, 0]) > xmax: xmax = np.max(xy[:, 0])
        if np.min(xy[:, 1]) < ymin: ymin = np.min(xy[:, 1])
        if np.max(xy[:, 1]) > ymax: ymax = np.max(xy[:, 1])

    # raster dimensions
    rows = int(np.ceil((ymax - ymin + 1)/res))
    cols = int(np.ceil((xmax - xmin +1)/res))
    TL = [xmin, ymax] # coordinates of the top left corner of the top left corner cell

    # initialize original array
    arr = np.zeros((rows, cols))

    # loop over all polygons in MultiPolygon
    for pol in mpol:
        # extract exterior coordinate pairs
        xy = np.rot90(pol.exterior.xy)
        # edit coordinates so that the relative zerpoint will be (0,0)
        xy[:,0] = (xy[:,0] - xmin)/res
        xy[:,1] = (xy[:,1] - ymin)/res

        # use the rasterio function to rasterize the extior polygon
        layer = rasterioRasterize([Polygon(xy)], out_shape=(rows, cols))
        # add this to existing Numpy array storing all rasterized polygons
        arr += layer

        for i in pol.interiors:
            # extract exterior coordinate pairs
            xy = np.rot90(i.xy)
            # edit coordinates so that the relative zeropoint will be (0,0)
            xy[:,0] = (xy[:,0] - xmin)/res
            xy[:,1] = (xy[:,1] - ymin)/res

            # use the rasterio function to rasterize the interior polygon
            layer = rasterioRasterize([Polygon(xy)], out_shape=(rows, cols))
            # substract this to existing Numpy array storing all rasterized polygons
            arr -= layer


    # transform to a boolean
    arr[arr > 0] = 1
    # flip it so north is up if you plot it
    arr = np.flip(arr, axis = 0)

    if return_minmaxs:
        return arr, xmin, ymin, xmax, ymax, TL, res
    else:
        return arr
