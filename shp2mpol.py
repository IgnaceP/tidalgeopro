from osgeo import ogr
import json
from shapely.geometry import Polygon as shPol
from shapely.geometry import MultiPolygon as shMPol
import utm
import numpy as np

def WGS842UTM(XY, utm_n, utm_l):
    """Function to transform a list of lat lon coordinates in WGS84 to a utm system
    :param XY: (Required) list of coordinate pairs or 2d numpy array
    :param utm_n: (Required) UTM zone number
    :param utm_l: (Required) UTM zone letter
    :return: list of coordinate pairs of projected coordinates
    """

    # initialize empty list to store new coordinates
    XY_utm = []

    # loop over coordinate pairs
    for lonlat in XY:
        # retrieve lat and lon
        lon, lat = lonlat[0:2]
        # transform lat and lon to UTM coordinate one by one
        utmxy = utm.from_latlon(lat, lon, utm_n, utm_l)
        # append to new list
        XY_utm.append(utmxy[0:2])

    return XY_utm

def shp2mpol(fn, print_coordinate_system = False, project_to_UTM = False):
    """ Function to load an ESRI shapefile and transform to shapely MPol
    :param fn:(Required) File path directory
    :param print_coordinate_system: (Optional) True/False to print the original coordinate system of the loaded loadPolygonFromShapefile
    :param project_to_UTM: (Optional) string with UTM number and letter (for instance, '17M') of the desired UTM zone
    :return: a shapely (Multi)Polygon
    """

    # load shapefile with the ogr toolbox of osgeo
    file = ogr.Open(fn)
    shape = file.GetLayer(0)
    if print_coordinate_system:
        print(f'The Coordinate system is: {shape.GetSpatialRef()}')

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
        if project_to_UTM:
            utm_n = int(project_to_UTM[:2])
            utm_l = project_to_UTM[2]

            ex = WGS842UTM(coor[0], utm_n, utm_l)

            inner = []

            if len(coor) > 1:
                for i in coor[1:]:
                    inner.append(WGS842UTM(i, utm_n, utm_l))

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
