""" Numpy array to geotiff

Function to convert a Numpy array to a geotiff (.tif)

author: Ignace Pelckmans
                (University of Antwerp, Belgium)
"""
import numpy as np
from osgeo import gdal, osr

def arr2Geotiff(arr, fname, TL, Res, projEPSG):
    """
    Function to export a 2D numpy array as geotiff

    author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        arr: (Required) 2D Numpy array
        fname: (Required) string output file directory for tiff file
        TL: (Required) Numpy array (2 x 1) tuple or list of x and y coordinate of the top left corner
        Res: (Required) Numpy array (2 x 1) tuple, list of single number with resolution
        projEPSG: (Required) EPSG code

    Returns:

    """

    # Transform array to 64 bit floats (readable in QGIS)
    arr = np.float64(arr)
    # get array dimensions
    rows, cols = np.shape(arr)

    # Create gdal driver in the geo-tiff format
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(fname, cols, rows, 1, gdal.GDT_Float64)

    # get coordinates of topleft corner
    TL_x, TL_y = TL

    # get resolution
    if type(Res) == list or type(Res) == tuple or type(Res) == np.ndarray:
        x_res, y_res = Res
    elif type(Res) == tuple:
        x_res, y_res = Res
    else:
        x_res, y_res = Res, Res

    # set geometry data (TL coordinates, resolution and skew)
    args = (TL_x, x_res, 0, TL_y, 0, y_res)
    ds.SetGeoTransform(args)

    # write band 1 values
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(-9999)

    # set projection based on given EPSG
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(projEPSG)
    ds.SetProjection(srs.ExportToWkt())

    # Close
    ds = None
