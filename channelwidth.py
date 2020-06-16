""" Channel Width

This module calculates the channel width at nodes located on a Centerline
!!! it assumes that a node is located in the middle of a channel as the channel width is calculated as twice the closest distance to the channel boundaries !!!

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""
import os
import pickle
import numpy as np
import multiprocessing as mp
from shapely.geometry import Point, LineString, MultiLineString, Polygon, \
                             MultiPolygon

from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")

from rasterize import *
from np2Geotiff import *
from pol2npy import *
from progressbar import *

#############################################################################
# Flags

multiprocessing_flag = True
#############################################################################


# function to find the closest vertex of a POI
def getNeighbor(p, pol, kernel_fract = 0.01, return_index = False):
    """
    Function to find the closest vertex of polgyon to a POI.
    The closest neighbor is found by taking all vertices of a polygon which are located
    within a distance r from the POI. The distance r is increased step by step (by the kernel_fract)
    untill a point is found. Therefore, the kernel_fract does not affect the result but it is meant
    to optimize the computational time of the procedure.

    Author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        p: (Required) 1x2 Numpy Array, tuple or list with x- and y-coordinates of the poin of interest (POI)
        pol: (Required) nx2 numpy array with x- and y-coordinates of the polygon's vertices
        kernel_fract: (Optional) stepsize of the kernel size (% of the total width/height of the polygon). The kernel stepsize is the size to increase the searching circle around POI.
                                 It does not affect the final result is meant to optimize the computational time of the procedure.
        return_index: (Optional, defaults to False) return the index of the neighbor as well

    Returns:
        Numpy array of dimensions 1 x 2 with the x- and y-coordinate of the closest vertex of the polygon to the POI

    """
    # get width and height of channel polygon
    width = np.max(pol[:, 0]) - np.min(pol[:, 0])
    height = np.max(pol[:, 1]) - np.min(pol[:, 1])

    # parse to separate x- and y-coordinate
    px, py = p

    # stepsize to increase kernel
    ks = kernel_fract* max(width, height)

    # initialize a flag to feed the while loop and start a counter
    neigh_flag = False; t = 0
    # run while loop as long as there are no neighbouring vertices selected
    while neigh_flag == False:

        # count
        t += 1

        # create a square mask to only calculate the distances between the POI and nearby polygon vertices
        neigh_mask = (pol[:,0] > px - t*ks) * (pol[:,0] < px + t*ks) \
                * (pol[:,1] > py - t*ks) * (pol[:,1] < py + t*ks)
        # mask on all vertices
        neigh = pol[neigh_mask]

        # calculate the distance between the POI en all vertices after masking
        dist = (np.sum((neigh - p) ** 2, axis=1)) ** 0.5
        # mask a circle within the square kernel
        circle_mask = (dist <= t*ks/2)
        neigh = neigh[circle_mask]
        dist = dist[circle_mask]

        # terminate loop if there is at least one vertex in the neighbouring circle, otherwise increase the kernel size
        if np.shape(neigh)[0] > 0:
            neigh_flag = True

    # get the neigbor coordinates which is the vertex within the circle (around POI, radius t*ks)
    neighbor = neigh[np.argmin(dist),:]
    neighbor_index = np.argmin(dist)

    if return_index:
        return neighbor, neighbor_index
    else:
        return neighbor

def channelWidthForOnePoint(xy, p):
    """
    Function to calculate the channel width of a single point in a polygon representing channels

    Author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        xy: (Required) Numpy array with dimensions n x 2, shapely Polygon or MultiPolygon representing the x- and y-coordinates of the polygon's vertices
        p: (Required) 1x2 Numpy Array, tuple or list with x- and y-coordinates of the poin of interest (POI)

    Returns:
        Estimated channel width in metric units (float)
    """

    # make sure it can handle pol being a list of coors, shapely MultiPolygon or a shapely Polygon
    # interior rings can be added as a regular coordinate pairs to the array
    if type(xy) == Polygon:
        xy_np = np.rot90(XY.exterior.xy)
        xy_inner = [np.rot90(i.xy) for i in XY.interiors]

        xy = xy_np
        for i in xy_inner:
            xy = np.vstack((xy, i))

    elif type(xy) == MultiPolygon:
        xy = np.zeros([1,2])
        for xy in XY:
            xy_np = np.rot90(xy.exterior.xy)
            xy = np.vstack((xy, xy_np))
            xy_inner = [i.xy for i in xy.interiors]
            for i in xy_inner:
                xy = np.vstack((xy, i))
        xy = xy[1:,:]
    else:
        pass

    # apply neighbor function and get index of original vertices coordinate pair array
    neigh = getNeighbor(p, xy, kernel_fract= 0.02)
    neigh_loc = np.where(xy == neigh)[0][0]

    # get left and right neigbour of the closest vertex
    # if the neighbor is the last the point of the coordinate list, its right neigbor is the first point of the coordinate list
    p_c = neigh
    p_l = xy[neigh_loc-1]
    if neigh_loc == np.shape(xy)[0]: p_r = xy[0]
    else: p_r = xy[neigh_loc+1]

    # calculate equations of straights between p_c & p_r and p_c & p_r: y = a*x + b
    a_l = (p_c[1] - p_l[1])/(p_c[0]-p_l[0])
    b_l = p_c[1] - a_l*p_c[0]
    a_r = (p_c[1] - p_r[1])/(p_c[0]-p_r[0])
    b_r = p_c[1] - a_r*p_c[0]

    # calculate distance between POI and the two straights
    d_l = abs(a_l*p[0] - p[1] + b_l)/(a_l**2 + 1)**0.5
    d_r = abs(a_r*p[0] - p[1] + b_r)/(a_r**2 + 1)**0.5

    # check if intersection between straight and dist-POI straight lays within the interval of p_c - p_r or p_c - p_l
    # For instance:
    #
    #              /                          /                     I
    #          II /                       II /                  ----------o
    #     I      /                  I       /                              \
    #  ---------o                ----------o                         POI *  \ II
    #                                                                        \
    #        * POI                              * POI
    #
    #
    #   First example: POI is closest to I and thus its distance to this channel edge (I & II) should be the distance between the straight I and POI.
    #   However the distance between POI and the unbounded straight of II is smaller than the distance bewteen POI and straight I. To avoid this,
    #   we check whether the intersection of the straight between POI and a segment is located on the segment.
    #
    #   Second example: now the closest distance to I and II is both at vertex 'o'.
    #
    #   Third example: closes distance is the distance between the straight of II and POI.

    # check if intersection of straigt between POI and left segment and left segment straight is located within the bounds of the left segment
    x_l = (p[1] + (a_l**-1)*p[0]-b_l)/(a_l + 1/a_l)
    y_l = a_l*x_l+b_l
    if min(p_l[0], p_c[0]) < x_l < max(p_l[0], p_c[0]) and \
            min(p_l[1], p_c[1]) < y_l < max(p_l[1], p_c[1]):
        pass
    else: d_l = float('inf')

    # check if intersection of straigt between POI and right segment and right segment straight is located within the bounds of the right segment
    x_r = (p[1] + (a_r**-1)*p[0]-b_r)/(a_r + 1/a_r)
    y_r = a_r*x_r+b_r
    if min(p_r[0], p_c[0]) < x_r < max(p_r[0], p_c[0]) and \
            min(p_r[1], p_c[1]) < y_r < max(p_r[1], p_c[1]):
        pass
    else:
        d_r = float('inf')

    # like in example 2
    if d_l == d_r == float('inf'): channel_width = 2*((p_c[0]-p[0])**2+(p_c[1]-p[1])**2)**0.5
    # like in example 1 or three (if the intersection point is not located within segment (either left or right) bounds, that distance becomes infinite)
    else: channel_width = np.min([d_l, d_r])*2

    return channel_width


def removeBubbles(mls, orders, conn, nodes, widths):
    """
    Function to reorder the channel widths so after interpolation, no bubbles form at confluences

    Author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        mls: (Required) Numpy array or shapely Multilines of dimensions n x 2 storing the node coordinates
        stream_orders: (Required) False/Numpy array with dimensions s x 1 storing the segment orders
        conn: (Required) Numpy array of shape (m, 2) representing section connectivity table (i-th row gives node indices of the i-th section)
        nodes: (Required) False/Numpu array of n x 2 with coordinate pairs of the nodes
        widths: (Required) list of Numpy Arrays with the channel width per segment

    Returns:
        Updated Numpy Array with dimensions n x 3 with the channel width (third column) at the given coordinates (first two columns) (unit depends on the coordinate system, in case of UTM it is meter)
    """

    # total number of nodes
    n_nodes = np.max(conn)+1
    # connectivity list of lists which stores the connected segments per node
    conn_nodes = [np.where(conn == n)[0] for n in range(0, n_nodes)]
    # total number of nodes
    n_nodes = len(conn_nodes)


    # initialize a list to store the new channel width arrays
    widths_upd = widths
    # loop over nodes
    for i in range(n_nodes):
        # only proceed if node is no endnode and if in this node a stream joins a bigger stream
        if len(conn_nodes[i]) == 3\
        and np.count_nonzero(orders[conn_nodes[i]] == np.max(orders[conn_nodes[i]])) == 2:
            # there is one smaller stream (with a lower order than the other two) at this node
            little = conn_nodes[i][np.argmin(orders[conn_nodes[i]])]
            # get channel widths for this segment
            channel_widths = widths[little]
            # determine direction of the segment and make sure that the first coordinate pair of a segment is (closest to) the confluence node
            # the variable coor are the coordinates of the little segment
            if conn[little][0] == i:
                coor = np.flip(np.rot90(mls[little].xy), axis = 0)
                channel_widths = np.flip(channel_widths)
            else:
                coor = np.rot90(mls[little].xy)
            # confluence coordinates
            confl = nodes[i]
            # get distance to confluence node
            dist = ((coor[:,1] -  confl[1])**2 + ((coor[:,0] -  confl[0])**2))**0.5
            # in each node where the channel width / 2 at the confluence is larger than the distance of that node
            # to the confluence point, the channel_width is set as the channel_width of the confluence node
            channel_widths[dist < channel_widths[0]/2] = channel_widths[0]
            # if the channel width array for this segment was flipped, flip it again
            if conn[little][0] == i:
                channel_widths = np.flip(channel_widths)

            # add to the list
            widths_upd[little] = channel_widths

    return widths_upd





def channelWidth(xy, mls, stream_orders = False, segment_connections = False, node_coordinates = False,  multiprocessing_flag = False):
    """
    Function to calculate the channel width at given nodes

    Author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        xy: (Required) shapely (Multi)Polygon or Numpy array of dimensions m x 2 storing the vertices coordinates representing the channel
        mls: (Required) shapely Multilines storing the node coordinates
        stream_orders: (Optional, defaults to False) False/Numpy array with dimensions s x 1 storing the segment orders
        segment_connections: (Optional, defaults to False) False/Numpy array of shape (m, 2) representing section connectivity table (i-th row gives node indices of the i-th section)
        node_coordinates: (Optional, defaults to False) False/Numpu array of n x 2 with coordinate pairs of the nodes
        multiprocessing_flag: (Optional) False/number flag to indicate whether to use multiprocessing. #cpu's is the total number - 2

    Returns:
        Numpy Array with dimensions n x 3 with the channel width (third column) at the given coordinates (first two columns) (unit depends on the coordinate system, in case of UTM it is meter)
    """

    # make sure it can handle pol being a list of coors, shapely MultiPolygon or a shapely Polygon
    # interior rings can be added as a regular coordinate pairs to the array
    if type(xy) == Polygon:
        xy_np = np.rot90(xy.exterior.xy)
        xy_inner = [np.rot90(i.xy) for i in XY.interiors]

        xy = xy_np
        for i in xy_inner:
            xy = np.vstack((xy, i))

    elif type(xy) == MultiPolygon:
        xy_np = np.zeros([1,2])
        for xy_p in xy:
            xy_ex = np.rot90(xy_p.exterior.xy)
            xy_np = np.vstack((xy_np, xy_ex))
            xy_inner = [i.xy for i in xy_p.interiors]
            for i in xy_inner:
                xy_np = np.vstack((xy_np, np.rot90(i)))
        xy = xy_np[1:,:]
    else:
        pass

    # save coors in one contiguous array
    vers = np.asarray([0,0])
    # save channel widths in one contiguous array
    widths = np.asarray([0])
    # and in a list of arrays, with an array per Linestring
    widths_per_line = []
    # loop over all linestrings
    for ls in mls:
        coor = np.rot90(ls.xy)

        if multiprocessing_flag:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count()-2)
            w = pool.starmap(channelWidthForOnePoint, [(xy, coor[i,:]) for i in range(np.shape(coor)[0])])
            pool.close()
            w = np.asarray(w)

        # otherwise, calculate the widths at nodes one by one
        else:
            w = np.zeros(np.shape(coor)[0])
            for i in range(np.shape(coor)[0]):
                w[i] = channelWidthForOnePoint(xy, coor[i,:])

        # append to lists and arrays
        widths_per_line.append(w)
        vers = np.vstack((vers, coor))
        widths = np.concatenate((widths, w))

    # remove the upper line ([0,0])
    vers = vers[1:,:]
    widths = widths[1:]

    # create a numpy array with three columns: lat, lon and channel width
    coor_and_widths = np.empty([np.shape(vers)[0], 3])
    coor_and_widths[:,0:2] = vers
    coor_and_widths[:,2] = widths

    # if the stream orders are given, improve the channel width at the confluences
    if type(stream_orders) == np.ndarray:
        # update the channel_widths based on the stream orders (fix bubbles at confluences)
        widths_updated = removeBubbles(mls, stream_orders, segment_connections, node_coordinates, widths_per_line)
        # flatten list of numpy arrays
        widths_flat = np.asarray([item for sublist in widths_updated for item in sublist])
        # Boolean array at which nodes the channel width is updated
        nonupdated_widths = (widths_flat - widths == 0)
        # update the return array
        coor_and_widths[:,2] = widths_flat
        #coor_and_widths = coor_and_widths[coor_and_widths[:,2] > 0,:]
    return coor_and_widths

def interpChannelWidth(mpol, widths, res = 1, save_as_geotiff = False, geotiff_epsg = 4326, save_as_background_field_parameters = False, interp_method = 'nearest', rasterized_channels = False):
    """
    Function to interpolate the channel widths to a raster

    author: Ignace Pelckmans (University of Antwerp, Belgium)

    Args:
        mpol: (Required) Shapely Multi(Polygon) to rasterize
        widths: (Required) Numpy Array with dimensions n x 3 with the channel width (third column) at the given coordinates (first two columns) (unit depends on the coordinate system, in case of UTM it is meter)
        res: (Optional, defaults to 1) float/integer representing the cell size
        save_as_geotiff: (Optional, defaults to False) False/directory string to store a geotiff file
        save_as_background_field_parameters: (Optional, defaults to False) False/directory string to store a pickled file with all information to create a gmsh structured background field
        interp_method: (Optinal, defaults to 'nearest') String of 'nearest' or 'linear' or cubic' with the interpolation method
        rasterized_channels: (Optional, defaults to False) False/String of directory path to the .txt file storing a 0/1 raster of the channels

    Returns:
        Numpy array with dimensions n x m with values zero outside the channels and the channel width in the channels
        Tuple of size two with coordinates of left top corner of the array
    """
    n_skeleton_vertices = widths.shape[0]
    # assign channel widths to vertices of the channel edges
    if interp_method == 'linear' or False:
     if type(mpol) == Polygon: mpol = [mpol]
     for pol in mpol:
         # read shapely object into a coordinate array
         xy = pol2Numpy(pol)
         # loop over all vertices
         for i in range(len(xy)):
             printProgressBar(i, len(xy))
             # get coordinate pair of the point of interest
             p = xy[i,:]
             # look for closest skeleton vertices
             _, ind = getNeighbor(p, widths[:n_skeleton_vertices,:2], return_index = True)
             # add the point of interest with the newly generated channel width to the array of widths
             w = widths[ind,2]
             new_row = np.array([p[0], p[1], w])
             widths = np.vstack((widths, new_row))

    if rasterized_channels:
        # check if the directory exist, if not create a npy file and pickle with corner coordinates, otherwise, load it
        if os.path.isfile(rasterized_channels):
            arr = np.loadtxt(rasterized_channels)
            with open(rasterized_channels[-3]+'pckl', 'rb') as input:
                xmin, ymin, TL = pickle.load(input)

        else:
            # rasterize the original channel polygonsfrom scipy.interpolate import griddata
            arr, xmin, ymin, _, _, TL, res = rasterize(mpol, res = res, return_minmaxs = True)
            np.savetxt(rasterized_channels, arr)
            with open(rasterized_channels[-3]+'pckl', 'wb') as output:
                pickle.dump([xmin, ymin, TL], output, pickle.HIGHEST_PROTOCOL)

    # transform coordinates of the nodes for which the channel width is known to array indices
    J = np.round((widths[:,0] - xmin - res/2)/res)
    I = np.round(arr.shape[0] - (widths[:,1] - ymin - res/2)/res)

    # meshgrid over the entire rasterized polygon array
    x = np.arange(arr.shape[1])
    y =  np.arange(arr.shape[0])
    X, Y = np.meshgrid(x,y)


    # interpolate over the entire array
    Ti = griddata((J, I), widths[:,2], (X, Y), method=interp_method)

    # store the results of the interpolation in a 2d Numpy array
    width_arr = np.zeros(np.shape(arr))
    for n in range(len(X)):
        i, j = Y[n], X[n]
        width_arr[i,j] = Ti[n]

    # all cells outside the river channels should stay zero
    width_arr = width_arr * arr

    # if indicated, save as a geotiff
    if save_as_geotiff:
        arr2Geotiff(width_arr, save_as_geotiff, TL, res, geotiff_epsg)

    if save_as_background_field_parameters:
        TLx, TLy = TL
        zeropoint = [TLx, TLy - arr.shape[0]*res]
        XY = [width_arr, zeropoint, res, res]
        with open(save_as_background_field_parameters, 'wb') as output:
            pickle.dump(XY, output, pickle.HIGHEST_PROTOCOL)

    return width_arr, TL
