""" Reproject a shapely MultiPolygon

This module reprojects a shapely MultiPolygon based on epsg codes

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""

import numpy as np
from pyproj import CRS
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon

def project(XY, orig_epsg, new_epsg):
    """
    Function to (re)project a list of coordinates to another coordinate system/projection

    author: Ignace Pelckmans, University of Antwerp, Belgium

    Args:
        XY: (Required) List of tuples/lists or n x 2/3 Numpy array with coordinate pairs/triplets
        orig_epsg: (Required) int representing the EPSG code of the given coordinates
        new_epsg: (Required) in representing the EPGS code of the coordinate system to (re)project to

    Returns:
        A list of tuples, representing the coordinate pairs of the (re)projected coordinates
    """

    # set crs objects
    crs_orig = CRS.from_epsg(orig_epsg)
    crs_new = CRS.from_epsg(new_epsg)

    # create a transformer
    transformer = Transformer.from_crs(crs_orig, crs_new)

    # initialize empty list to store new coordinates
    XY_repr = []

    # loop over coordinate pairs
    for lonlat in XY:
        # retrieve lat and lon
        lon, lat = lonlat[0:2]
        # transform lat and lon to new coordinate system/projection
        xy_repr = transformer.transform(lat, lon)
        # append to new list
        XY_repr.append(xy_repr[0:2])

    return XY_repr

def pol2Pol(mpol, orig_epsg, new_epsg):
    """
    Function to (re)project a list of coordinates to another coordinate system/projection

    author: Ignace Pelckmans, University of Antwerp, Belgium

    Args:
        mpol: shapely (Multi)Polygon to reproject
        orig_epsg: (Required) int representing the EPSG code of the given coordinates
        new_epsg: (Required) in representing the EPGS code of the coordinate system to (re)project to

    Returns:
        reprojected shapely (Multi)Polygon
    """

    # make sure it can handle both a Polygon and MultiPolygon
    if type(mpol) == Polygon:
        mpol = [mpol]

    # initialize the list to store the projected pols
    pols_repr = []
    # loop over all polygons in MultiPolygon
    for pol in mpol:
        # reproject exterior coordinates of polygon
        ex = project(np.rot90(np.asarray(pol.exterior.xy)), orig_epsg, new_epsg)
        # reproject inner rings of polygon
        inner = []
        for i in pol.interiors:
                inner.append(project(np.rot90(np.asarray(i.xy)), orig_epsg, new_epsg))
        pols_repr.append(Polygon(ex,inner))

    return MultiPolygon(pols_repr)
