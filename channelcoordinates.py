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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def channelCoordsForOnePoint(xy, p, return_projected_point = False, print_info = False):
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
    neigh_dist = ((xy[:,0]-p[0])**2 + (xy[:,1]-p[1])**2)**0.5
    neigh_loc = np.argmin(neigh_dist)
    neigh = xy[neigh_loc]

    # get left and right neigbour of the closest vertex
    # if the neighbor is the last the point of the coordinate list, its right neigbor is the first point of the coordinate list
    p_c = xy[neigh_loc]
    p_l = xy[neigh_loc-1]
    if neigh_loc == np.shape(xy)[0]-1:
        p_r = xy[0]
    else: p_r = xy[neigh_loc+1]

    # calculate equations of straights between p_c & p_r and p_c & p_r: y = a*x + b
    a_l = (p_c[1] - p_l[1])/(p_c[0]-p_l[0])
    b_l = p_c[1] - a_l*p_c[0]

    if print_info:
        print('# --------------------------------------------------------------------------------------------------------- #\n left, central and right vertex:')

        print(p_l)
        print(p_c)
        print(p_r)
        print('--------------------------------------')

    a_r = (p_c[1] - p_r[1])/(p_c[0]-p_r[0])
    b_r = p_c[1] - a_r*p_c[0]

    # calculate distance between POI and the two straights
    if a_l != float('inf') and a_l != float('-inf'):
        d_l = abs(a_l*p[0] - p[1] + b_l)/(a_l**2 + 1)**0.5
    else:
        d_l = abs(p[0] - p_l[0])
    if a_r != float('inf') and a_r != float('-inf'):
        d_r = abs(a_r*p[0] - p[1] + b_r)/(a_r**2 + 1)**0.5
    else:
        d_r = abs(p[0] - p_r[0])

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
    if a_l != float('inf') and a_l != float('-inf'):
        x_l = (p[1] + (a_l**-1)*p[0]-b_l)/(a_l + 1/a_l)
        y_l = a_l*x_l+b_l
    else:
        x_l = p_l[0]
        y_l = p[1]
    if min(p_l[0], p_c[0]) < x_l < max(p_l[0], p_c[0]) or \
            min(p_l[1], p_c[1]) < y_l < max(p_l[1], p_c[1]):
        pass
    else: d_l = float('inf')


    # check if intersection of straigt between POI and right segment and right segment straight is located within the bounds of the right segment
    if a_r != float('inf') and a_r != float('-inf'):
        x_r = (p[1] + (a_r**-1)*p[0]-b_r)/(a_r + 1/a_r)
        y_r = a_r*x_r+b_r
    else:
        x_r = p_r[0]
        y_r = p[1]
    if min(p_r[0], p_c[0])-.01 < x_r < max(p_r[0], p_c[0]) or \
            min(p_r[1], p_c[1]) < y_r < max(p_r[1], p_c[1]):
        pass
    else:
        d_r = float('inf')

    if print_info:
        print('left:')
        print('a_l: ', a_l)
        print('x_l: ', x_l)
        print('y_l: ', y_l)
        print('------------')
        print('right:')
        print('a_r: ', a_r)
        print('x_r: ', x_r)
        print('y_r: ', y_r)
        print('------------')

    # like in example 2
    if d_l == d_r == float('inf'):
        process = 'example 2'
        pc = ((p_c[0]-p[0])**2+(p_c[1]-p[1])**2)**0.5
        x_proj, y_proj = p_c
        xy_new = xy.copy()
        ac_i = neigh_loc
    # like in example 1 or three (if the intersection point is not located within segment (either left or right) bounds, that distance becomes infinite)
    else:
        pc = np.min([d_l, d_r])
        if d_l < d_r:
            process = 'example 1'
            x_proj, y_proj = x_l,y_l
            xy_new = np.insert(xy, neigh_loc, np.array([x_proj, y_proj]), axis = 0)
            ac_i = neigh_loc
        else:
            process = 'example 3'
            x_proj, y_proj = x_r,y_r
            xy_new = np.insert(xy, neigh_loc+1, np.array([x_proj, y_proj]), axis = 0)
            ac_i = neigh_loc+1
    xy_dist = np.cumsum(((xy_new[1:,0]-xy_new[:-1,0])**2 + (xy_new[1:,1]-xy_new[:-1,1])**2)**0.5)
    xy_dist = np.insert(xy_dist,0,0)
    ac = xy_dist[ac_i]
    if print_info: process

    # projection vector
    v_proj = np.array([x_proj - p[0],y_proj - p[1]])
    if process in ['example 1','example 3']:
        v_cl = np.array([xy[ac_i,0] - x_proj,xy[ac_i,1] - y_proj])
    else:
        v_cl = np.array([xy[ac_i+1,0] - x_proj,xy[ac_i+1,1] - y_proj])
    v_angle = np.cross(v_proj,v_cl)
    if v_angle > 0: pc *= -1

    if print_info:
        print('projection vector: ', v_proj)
        print('centerline vector: ', v_cl)
        print('angle between vectors', v_angle)

    if return_projected_point:
        return pc, ac, [x_proj, y_proj]
    else:
        return pc, ac

def getChannelCoords(xy, pois, return_projected_points = False, print_info = False):
    """
    Function to calculate channel coordinates for given cartesian coordinates

    Args:
        xy: (Required) Numpy array with dimensions n x 2 representing the x- and y-coordinates of the centerlines vertices
        pois: (Required) Numpy array with dimensions m x 2 representing the x- and y-coordinates of the pois

    Returns:
        Numpy Array with dimensions m x 2 with the channel coordinates of the pois
    """

    PPs = []
    CCs = []
    for poi in pois:
        vars = channelCoordsForOnePoint(xy, poi, return_projected_point = return_projected_points, print_info = print_info)
        pc = vars[0]
        ac = vars[1]
        CCs.append(np.array([pc,ac]))
        if return_projected_points:
            PPs.append(vars[-1])

    CCs = np.asarray(CCs)

    if not return_projected_points:
        return CCs
    else:
        return CCs, PPs
