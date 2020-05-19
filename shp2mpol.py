""" Shapefile 2 Shapely MultiPolygon

This module loads a shapefile to a shapefile MultiPolygon

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""


from osgeo import ogr
import json
from shapely.geometry import Polygon as shPol
from shapely.geometry import MultiPolygon as shMPol
import utm
import numpy as np
from pyproj import CRS
from pol2pol import project


def shp2Mpol(fn, print_coordinate_system = False, project_to_epsg = False):
    """
    Function to load and (re)project one or multiple polygons from a ESRI shapefile
    ! the shapefile should only have one polygon per feature !

    author: Ignace Pelckmans

    Args:
        fn: (Required) string of .shp file directory
        print_coordinate_system: (Optional, defaults to False) True/False to print the original EPSG code
        project_to_epsg: (Optional, defaults to False) False/int representing the epsg code of the desired projection, indicate False if the data should not be reprojected

    Returns:
        a shapely (Multi)Polygon
    """

    # load shapefile with the ogr toolbox of osgeo
    file = ogr.Open(fn)
    shape = file.GetLayer(0)

    epsg = int(shape.GetSpatialRef().ExportToPrettyWkt().splitlines()[-1].split('"')[3])
    crs = CRS.from_epsg(epsg)
    if print_coordinate_system:
        print("The EPSG code of the coordinate system is: %d" % (crs.to_epsg()))
    # get number of polygons in shapefile
    n_features = shape.GetFeatureCount()

    # initialize new polygon list
    pols = []

    # loop over all polygons
    for i in range(n_features):
        # get feature object
        feature = shape.GetFeature(i)
        # export to JS objects
        feature_JSON = feature.ExportToJson()
        # loads as JS object array
        feature_JSON = json.loads(feature_JSON)

        # extract coordinate attribute from JS object
        # coor is a list of all rings, first one is the outer ring, further elements are coordinate pair lists of the inner rings
        coor = feature_JSON['geometry']['coordinates']

        # if indicated, transform all coordinates to UTM coordinates
        if project_to_epsg:
            ex = project(coor[0], crs.to_epsg(), project_to_epsg)

            inner = []

            if len(coor) > 1:
                for i in coor[1:]:
                    if 2 <= np.shape(np.asarray(i))[1] <= 3:
                        inner.append(project(i, crs.to_epsg(), epsg))

        else: ex = coor[0]; inner = coor[1:]

        # create a shapely polygon
        pol = shPol(ex, inner)
        pols.append(pol)

    # create a shapely MultiPolygon
    mpol = shMPol(pols)

    if len(pols)==1:
        return pol
    else:
        return mpol
