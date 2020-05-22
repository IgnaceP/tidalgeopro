import numpy as np
from shapely.geometry import Point



################################################################################
# channel width ################################################################
################################################################################

def channel_width(coords, mpol):

  """ Compute channel width along skeleton, as twice the distance between the skeleton and the channel edges.

  !!! Attention !!!
  This is the distance to the closest point defining the channel edges. A more accurate calculation can be found in the channelwidth module. The present function gives satisfactory results for high resolution channel edges.

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


################################################################################
# cross-section metrics ########################################################
################################################################################

def metrics(coords, sections, mpol, slf_fn):

  """ Compute cross-section metrics (channel width, depth and cross-section area) along the skeleton, based on channel edges and a digital elevation model (DEM).

  !!! Attention !!!
  Currently, the DEM must be defined on a Selafin file (format of Telemac output files), which assumes an unstructured triangular grid. Other formats (including for structured rectangular grids) will be considered in the future.

  Required parameters:
  coords (Numpy array of shape (n, 2)): coordinates of the skeleton points
  sections (Numpy array of shape (n)): section index of each skeleton point
  mpol (MultiPolygon): structure of multiple Polygons describing channel edges
  slf_fn (string): name of the Selafin file describing the DEM

  Returns
  Numpy array of shape (n): channel width at each skeleton point
  Numpy array of shape (n): channel depth at each skeleton point
  Numpy array of shape (n): channel cross-section area at each skeleton point

  """

  for i in np.unique(sections):
    print(i)







  #return width, depth, area