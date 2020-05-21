""" Channel Width

This module calculates the channel width at nodes located on a Centerline
!!! it assumes that a node is located in the middle of a channel as the channel width is calculated as twice the closest distance to the channel boundaries !!!

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""

import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, Polygon, \
                             MultiPolygon


import warnings
warnings.filterwarnings("ignore")

#############################################################################
# Flags

multiprocessing_flag = True
#############################################################################


# function to find the closest vertex of a POI
def getNeighbor(p, pol, kernel_fract = 0.01):
    """
    Function to find the closest vertex of polgyon to a POI.
    The closest neighbor is found by taking all vertices of a polygon which are located
    within a distance r from the POI. The distance r is increased step by step (by the kernel_fract)
    untill a point is found. Therefore, the kernel_fract does not affect the result but it is meant
    to optimize the computational time of the procedure.

    Args:
        POI: (Required) 1x2 Numpy Array, tuple or list with x- and y-coordinates of the poin of interest (POI)
        pol: (Required) nx2 numpy array with x- and y-coordinates of the polygon's vertices
        kernel_fract: (Optional) stepsize of the kernel size (% of the total width/height of the polygon). The kernel stepsize is the size to increase the searching circle around POI.
                                 It does not affect the final result is meant to optimize the computational time of the procedure.

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

    return neighbor

def channelWidthForOnePoint(xy, p):
    """
    Function to calculate the channel width of a single point in a polygon representing channels

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

def channelWidthNodes(xy, nodes, multiprocessing_flag = False):
    """
    Function to calculate the channel width at given nodes

    Args:
        xy: (Required) shapely (Multi)Polygon or Numpy array of dimensions n x 2 storing the vertices coordinates representing the channel
        nodes: (Required) Numpy array of dimensions n x 2 storing the node coordinates
        multiprocessing_flag: (Optional) False/number flag to indicate whether to use multiprocessing. #cpu's is the total number - 2

    Returns:
        Numpy Array with dimensions n x 1 with the channel width at the nodes (unit depends on the coordinate system, in case of UTM it is meter)
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


    # in the case of multiprocessing, start a pool where each member calculates the width at a node
    if multiprocessing_flag:
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count()-2)
        widths = pool.starmap(channelWidthForOnePoint, [(xy, nodes[i,:]) for i in range(np.shape(nodes)[0])])
        pool.close()
        widths = np.asarray(widths)
    # otherwise, calculate the widths at nodes one by one
    else:
        widths = np.zeros(np.shape(nodes)[0])
        for i in range(np.shape(nodes)[0]):
            widths[i] = channelWidthForOnePoint(xy, nodes[i,:])

    return widths
