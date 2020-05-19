from osgeo import ogr
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import mapping, Polygon
import fiona
from fiona.crs import from_epsg


def mpol2shp(mpol, filename, epsg = 4326):
    """
    Function to save a (MultiPolygon) as a Esri .shp
    :param mpol: (Multi)Polygon to shapefile
    :param filename: filename directory to store the shp
    :param epsg: crs
    """

    # make sure the function can handle both multpolygon and Polygons
    if type(mpol) == Polygon:
        mpol = MultiPolygon([mpol])

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    with fiona.open(filename, 'w', 'ESRI Shapefile', schema = schema, crs = from_epsg(epsg)) as c:
        ## If there are multiple geometries, put the "for" loop here
        for pol in mpol:
            c.write({
                'geometry': mapping(pol),
                'properties': {'id': 123},
            })
