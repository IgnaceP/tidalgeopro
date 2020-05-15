import numpy as np
from shapely.geometry import Point



################################################################################
# channel width ################################################################
################################################################################

def channel_width(coords, mpol):

  """ Compute channel width along skeleton, as two times the distance between skeleton and channel edges

  Required parameters:
  coords (Numpy array of shape (n, 2)): coordinates of the skeleton points
  mpol (MultiPolygon): structure of multiple Polygons describing channel edges

  Returns:
  Numpy array of shape (n): channel width at each skeleton point

  """

  # initialize
  width = np.zeros(coords.shape[0])

  # compute channel width
  for i in range(len(width)):
    point = Point(coords[i, :])
    width[i] = point.distance(mpol.boundary) * 2

  return width


