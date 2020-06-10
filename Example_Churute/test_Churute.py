"""
Script to test the tidealgeopro module based from a .shp file

author: Ignace Pelckmans
            (University of Antwerp, Belgium)

!!! issue to be solved: at the moment the final skeleton is not working !!!
"""
#--------------------#
#-- import modules --#
#--------------------#

import os
import glob
import time

from remove_vertices import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from shapely.wkt import loads
from shapely.geometry import Point, LineString, MultiLineString, Polygon, \
                             MultiPolygon

import skeleton
from shp2mpol import *
from pol2pol import *
from downstream import *
from streamorder import *
from channelwidth import *
from rasterize import *

from test_Churute_plotfigures import *

#--------------------#
#-- Load shapefile --#
#--------------------#

# path directory of input file
inputfile = './input/Churute_Subregion_Channels.shp'

# load shapefile as shapely MultiPolygon
mpol, epsg = shp2Mpol(inputfile, return_coordinate_system = True)

# transform to UTM 17 S projection (metric coordinates)
mpol = pol2Pol(mpol, epsg, 32717)

# if it is a single multipolygon, transform to a Multipolygon
if type(mpol) == Polygon:
    mpol = MultiPolygon([mpol])

#-----------------------#
#-- Generate skeleton --#
#-----------------------#

# downstream node
dns = [0]

# raw skeleton
skls = skeleton.raw_skeleton(mpol)

# clean skeleton
coords, sections, mls = skeleton.clean_skeleton(skls, mpol, ratio = 1)

# final skeleton
#coords, dist, sections, mls, p_coords, p_dist, p_sections \
#  = skeleton.final_skeleton(coords, sections, mls, mpol, dns, ratio = 5)

#-------------------#
#-- Stream Orders --#
#-------------------#

orders = streamOrder(sections, downstream_nodes = 0)

#-------------------#
#-- Channel Width --#
#-------------------#

# channel width at all skeleton vertices
widths = channelWidth(mpol, mls, stream_orders = orders, segment_connections = sections, node_coordinates = coords, multiprocessing_flag = True)

# rasterized channel width
arr, TL = interpChannelWidth(mpol, widths, res = 5, save_as_geotiff = '/home/ignace/Desktop/test.tif', geotiff_epsg = 32717)

#-----------------------#
#-- Save as txt files --#
#-----------------------#

# if necessary, create new directories
if not os.path.isdir('./txt_files/'):
    os.mkdir('./txt_files/')
if not os.path.isdir('./figures/'):
    os.mkdir('./figures/')

# save skeleton
np.savetxt('./txt_files/skeleton_node_coords.txt', coords)
#np.savetxt('./txt_files/skeleton_node_dist.txt', dist)
np.savetxt('./txt_files/skeleton_sections.txt', sections)
file = open('./txt_files/skeleton.txt', 'w')
file.write(mls.to_wkt())
file.close()
#np.savetxt('./txt_files/skeleton_point_coords.txt', p_coords)
#np.savetxt('./txt_files/skeleton_point_dist.txt', p_dist)
#np.savetxt('./txt_files/skeleton_point_sections.txt', p_sections)

# save stream orders
np.savetxt('./txt_files/stream_orders.txt', orders)

# save channel widths
np.savetxt('./txt_files/channel_widths.txt', widths)
np.savetxt('./txt_files/rasterized_channelwidth.txt', arr)


#------------------------------#
#-- Generate example figures --#
#------------------------------#

# plot clean skeleton
#plotFinalSkeleton(mpol, mls, coords, p_dist, p_coords)

# plot stream orders
plotStreamOrders(mpol, mls, coords, orders)

# plot channel width
plotChannelWidtRas(arr)
