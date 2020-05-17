""" Skeleton

This module allows to compute the skeleton of a tidal channel network quasi-automatically

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, MultiPolygon
from shapely.ops import nearest_points

from centerline.geometry import Centerline



################################################################################
# raw skeleton #################################################################
################################################################################

def raw_skeleton(mpol):

  """ Compute the raw skeleton of a tidal channel network based on channel edges

  Required parameters:
  mpol (MultiPolygon): structure of multiple polygons describing channel edges

  Returns:
  list of MultiLineStrings (one for each Polygon of the input MultiPolygon) describing the raw skeleton

  """

  # compute and return list of raw skeletons (one for each polygon)
  skls = []
  for pol in mpol.geoms:
    skls.append(Centerline(pol))
  return skls



################################################################################
# clean skeleton ###############################################################
################################################################################

def clean_skeleton(skls, mpol, ratio = 1):

  """ Clean raw skeleton by merging segments between two confluence points into single LineStrings (so-called sections) and removing tip sections whose ratio (section length / distance between confluence point and channel edges) are shorter than a threshold value

  Required parameters:
  skls (list of MultiLineStrings): raw skeleton
  mpol (MultiPolygon): structure of multiple Polygons describing channel edges

  Optional parameters:
  ratio (float, default = 1): threshold ratio for tip sections (section length / distance between confluence point and channel edges): tip sections with smaller ratio are disregarded

  Returns:
  Numpy array of shape (n, 2): coordinates of the tip and confluence points (so-called nodes)
  Numpy array of shape (m, 2): section connectivity table (i-th raw gives node indices of the i-th section)
  MultiLineString describing the clean skeleton

  """


  ##############
  # initialize #
  ##############

  # list of skeleton point coordinates (coords)
  # list of skeleton section point indices (sps)
  coords = []
  sps = []
  for skl in skls:
    for ls in skl.geoms:
      sp = []
      for coord in ls.coords:
        if coord not in coords:
          coords.append(coord)
          sp.append(len(coords) - 1)
        else:
          sp.append(coords.index(coord))
      sps.append(sp)

  # convert list into array
  coords = np.array(coords)

  # array of skeleton nodes
  nodes = np.array(range(coords.shape[0]))

  # array of skeleton section node indices
  sns = np.array(sps)

  # number of connections
  # 1: skeleton tip point (up- or downstream)
  # 2: between two skeleton sections to merge
  # 3: confluence point
  ncon = np.zeros(nodes.shape[0], dtype = int)
  for i in range(ncon.shape[0]):
    ncon[i] = np.sum(sns == i)

  # channel edges: convert multi-polygons into list of linear rings
  lrs = []
  for pol in mpol.geoms:
    lrs.append(pol.exterior)
    for interior in pol.interiors:
      lrs.append(interior)


  #########
  # clean #
  #########

  # initialize go boolean (determines whether while loop goes on)
  go = True

  while go:

    # initialize go boolean for this iteration
    go = False


    ##################
    # merge sections #
    ##################

    # initialize new lists
    sps_new = []
    sns_new = []

    # for each tip and confluence point
    for n0 in nodes[(ncon == 1) + (ncon == 3)]:

      # index of corresponding section(s)
      for i in range(ncon[n0]):
        tmp = np.argwhere(sns == n0)[i]
        s = tmp[0]

        # index of other skeleton section node
        n1 = sns[s, np.mod(tmp[1] + 1, 2)]

        # initialize list of skeleton point indices of new section
        if n0 == sns[s][0]:
          sp_new = sps[s]
        else:
          sp_new = sps[s][-1::-1]

        # compute remaining part of new section
        while ncon[n1] == 2:
          tmp = np.argwhere(sns == n1)
          for j in range(2):
            s = tmp[j][0]
            n1 = sns[s, np.mod(tmp[j][1] + 1, 2)]
            if n1 not in sp_new:
              if n1 == sns[s][1]:
                sp_new += sps[s][1:]
              else:
                sp_new += sps[s][-2::-1]
              break

        # update new lists
        if [n1, n0] not in sns_new:
          sps_new.append(sp_new)
          sns_new.append([n0, n1])

    # update lists and arrays
    sps = sps_new.copy()
    sns = np.array(sns_new)

    # number of connections
    ncon = np.zeros(nodes.shape[0], dtype = int)
    for i in range(ncon.shape[0]):
      ncon[i] = np.sum(sns == i)


    #########################
    # remove small sections #
    #########################

    # initialize new lists
    sps_new = []
    sns_new = []

    # for each section
    for i in range(len(sps)):

      # tip sections
      if ncon[sns[i, 0]] == 1 or ncon[sns[i, 1]] == 1:

        # section length
        coords_loc = []
        for j in sps[i]:
          coords_loc.append((coords[j, 0], coords[j, 1]))
        l = LineString(coords_loc).length

        # maximum distance to closest channel edge
        p0 = Point((coords[sns[i, 0], 0], coords[sns[i, 0], 1]))
        p1 = Point((coords[sns[i, 1], 0], coords[sns[i, 1], 1]))
        d0 = np.inf
        d1 = np.inf
        for lr in lrs:
          d0 = np.minimum(d0, p0.distance(lr))
          d1 = np.minimum(d1, p1.distance(lr))
        d = np.maximum(d0, d1)

        # only keep tip sections longer than distance to channel edge
        if l / d > ratio:
          sps_new.append(sps[i])
          sns_new.append(sns[i])

        # if sections have been removed, the loop must goes on (merging, etc)
        else:
          go = True

      # keep other sections
      else:
        sps_new.append(sps[i])
        sns_new.append(sns[i])

    # update lists and arrays
    sps = sps_new.copy()
    sns = np.array(sns_new)

    # number of connections
    ncon = np.zeros(nodes.shape[0], dtype = int)
    for i in range(ncon.shape[0]):
      ncon[i] = np.sum(sns == i)


  ##########################
  # clean lists and arrays #
  ##########################

  # initialize new lists
  coords_new = []
  sps_new = []
  sns_new = []

  # update new lists
  for sp in sps:
    for p in sp:
      coord = list(coords[p, :])

      # first node
      if p == sp[0]:
        if coord in coords_new:
          n = coords_new.index(coord)
          sps_new.append([n])
          sns_new.append([n])
        else:
          coords_new.append(coord)
          sps_new.append([len(coords_new) - 1])
          sns_new.append([len(coords_new) - 1])

      # middle points
      elif p != sp[-1]:
        coords_new.append(coord)
        sps_new[-1].append(len(coords_new) - 1)

      # last node
      else:
        if coord in coords_new:
          n = coords_new.index(coord)
          sps_new[-1].append(n)
          sns_new[-1].append(n)
        else:
          coords_new.append(coord)
          sps_new[-1].append(len(coords_new) - 1)
          sns_new[-1].append(len(coords_new) - 1)

  # update lists and arrays
  coords = np.array(coords_new)
  sps = sps_new.copy()
  sns = np.array(sns_new)


  ##########
  # output #
  ##########

  # section line strings as a multi-line string
  lss = []
  for sp in sps:
    tmp = []
    for p in sp:
      tmp.append((coords[p, 0], coords[p, 1]))
    lss.append(LineString(tmp))
  mls = MultiLineString(lss)

  # section nodes
  nodes = list(np.unique(sns))
  sections = np.zeros(sns.shape, dtype = int)
  for i in range(sns.shape[0]):
    sections[i, :] = [nodes.index(sns[i, 0]), nodes.index(sns[i, 1])]

  # node coordinates
  coords = coords[nodes, :]

  # channel edges
  pols = []
  for pol in mpol.geoms:
    for ls in mls.geoms:
      if pol.contains(ls):
        pols.append(pol)
        break
  mpol = MultiPolygon(pols)

  return coords, sections, mls, mpol



################################################################################
# final skeleton ###############################################################
################################################################################

def final_skeleton(coords, sections, mls, mpol, dns, ratio = 1, dx = 1):

  """ Compute final skeleton by computing downstream distances, orienting sections from down- to upstream, connecting unconnected mini-skeletons if the ratio (mini-skeleton upstream distance / distance to main skeleton) is higher than a threshold value, and redefining the skeleton with a regular distance between points

  Attention: the indices of the downstream tip points must be given by the user (only non-automated part of the process)

  Required parameters:
  coords (Numpy array of shape (n, 2)): coordinates of the tip and confluence points (so-called nodes)
  sections (Numpy array of shape (m, 2)): section connectivity table (i-th raw gives node indices of the i-th section)
  mls (MultiLineString): structure of multiple LineStrings describing the clean skeleton
  mpol (MultiPolygon): structure of multiple Polygons describing channel edges
  dns (list of integers): indices of the downstream nodes

  Optional parameters:
  ratio (float, default = 1): threshold ratio for unconnected mini-skeletons (mini-skeleton upstream distance / distance to main skeleton): mini-skeleton with smaller ratio are disregarded
  dx (float, default = 1): distance (m) between two points of the final skeleton

  Returns:
  Numpy array of shape (n, 2): coordinates of the confluence points (so-called nodes)
  Numpy array of shape (n): downstream distances of each node
  Numpy array of shape (m, 2): section connectivity table (i-th raw gives node indices of the i-th section)
  MultiLineString describing the final skeleton
  Numpy array of shape (p, 2): coordinates of the skeleton points
  Numpy array of shape (p): downstream distances of each skeleton point
  Numpy array of shape (p): section index of each skeleton point

  """


  ##########################################################################
  # compute array of downstream distances, change section orientations to  #
  # follow down- to upstream direction, split sections at equal downstream #
  # distances                                                              #
  ##########################################################################

  coords, dist, sections, mls \
    = downstream_distance(coords, sections, mls, mpol, dns)


  ########################################################
  # connect unconnected sections if they are long enough #
  ########################################################

  # initialize list of reconnected nodes
  reconnected_nodes = []

  # convert multi-line string into list of line strings
  lss = list(mls.geoms)

  # as long as there are nodes not connected and not disregarded
  while np.sum(np.isinf(dist)) > 0:

    # main skeleton
    lss0 = []
    for s in range(len(lss)):
      n0 = sections[s, 0]
      if np.isfinite(dist[n0]):
        lss0.append(lss[s])
    mls0 = MultiLineString(lss0)

    # distance-to-skeleton array
    dist_skl = np.zeros(dist.shape) + np.inf
    for n in range(len(dist_skl)):
      if np.isinf(dist[n]):
        dist_skl[n] = Point(coords[n, 0], coords[n, 1]).distance(mls0)

    # look for closest mini-skeleton long enough to be added to main skeleton
    while True:

      # closest unconnected node
      n_ext = np.argmin(dist_skl)

      # stop loop if all unconnected nodes have been connected or disregarded
      if np.sum(np.isfinite(dist_skl)) == 0:
        break

      # corresponding mini-skeleton and upstream distance
      coords_ext, dist_ext, sections_ext, mls_ext = \
        downstream_distance(coords, sections, mls, mpol, [n_ext], \
                            with_distance_to_downstream_edge = False)
      upstream_ext = np.max(dist_ext[np.isfinite(dist_ext)])

      ####################################################################
      # to be corrected in future versions:                              #
      #   coords_ext, sections_ext and mls_ext should probably be used   #
      #   to update coords, sections and mls                             #
      ####################################################################

      # add mini-skeleton if ratio between upstream distance and distance to
      # main skeleton higher than threshold
      if upstream_ext / dist_skl[n_ext] < ratio:
        dist[n_ext] = np.nan
        dist_skl[n_ext] = np.inf

      else:

        # nearest section
        p_ext = Point(coords[n_ext, 0], coords[n_ext, 1])
        for i in range(len(lss)):
          if p_ext.distance(lss[i]) == dist_skl[n_ext]:
            s = i
            break

        # nearest point on the main skeleton
        p_skl = nearest_points(p_ext, lss[s])[1]

        # downstream distance of nearest point
        dist_p_skl = dist[sections[s, 0]] + lss[s].project(p_skl)

        # update downstream distance array
        dist[np.isfinite(dist_ext)] = dist_p_skl + dist_skl[n_ext] \
                                    + dist_ext[np.isfinite(dist_ext)]

        # update list of reconnected nodes
        for n in range(coords.shape[0]):
          if np.isfinite(dist_ext[n]):
            reconnected_nodes.append(n)

        break


  #########
  # clean #
  #########

  # initialize new lists
  coords_new = []
  dist_new = []
  sections_new = []
  lss_new = []
  p_coords = []
  p_dist = []
  p_sections = []

  # for each old section
  for s in range(len(lss)):

    # old node indices and corresponding downstream distances
    n0 = sections[s, 0]
    n1 = sections[s, 1]
    dist0 = dist[n0]
    dist1 = dist[n1]

    ########################################################################
    # to be corrected in future versions:                                  #
    #   for some reasons, some sections are still reversed                 #
    #   (apparently from added disconnected mini-skeletons)                #
    ########################################################################

    # hack to handle the problem mentioned in the frame above
    if dist0 > dist1:
      n0 = sections[s, 1]
      n1 = sections[s, 0]
      dist0 = dist[n0]
      dist1 = dist[n1]
      ls_coords = list(lss[s].coords)
      ls_coords.reverse()
      lss[s] = LineString(ls_coords)

    # only keep main skeleton
    if np.isfinite(dist0) and np.isfinite(dist1):

      # first node
      coord0 = [coords[n0, 0], coords[n0, 1]]
      if coord0 in coords_new:
        sections_new.append([coords_new.index(coord0)])
      else:
        coords_new.append(coord0)
        dist_new.append(dist0)
        sections_new.append([len(coords_new) - 1])

      # last node
      coord1 = [coords[n1, 0], coords[n1, 1]]
      if coord1 in coords_new:
        sections_new[-1].append(coords_new.index(coord1))
      else:
        coords_new.append(coord1)
        dist_new.append(dist1)
        sections_new[-1].append(len(coords_new) - 1)


      #######################
      # regular line string #
      #######################

      # downstream distances at extreme regular points
      p_dist_loc_0 = np.ceil(dist0 / dx).astype(int) * dx
      p_dist_loc_1 = np.floor(dist1 / dx).astype(int) * dx

      # initialize list of line string points
      points = []

      # regular points and downstream distances
      for p_dist_loc in np.arange(p_dist_loc_0, p_dist_loc_1 + dx, dx):
        point = lss[s].interpolate(p_dist_loc - p_dist_loc_0)
        p_coords.append([point.x, point.y])
        p_dist.append(p_dist_loc)
        p_sections.append(len(sections_new) - 1)
        points.append(point)

      # line string
      if len(points) > 1:
        lss_new.append(LineString(points))
      else:
        print(dist1 - dist0)
        print('\n warning in final_skeleton:')
        print('number of line strings in mls is smaller than number of sections because some sections are too small compared to dx \n')

  # update arrays and multi-line string
  coords = np.array(coords_new)
  dist = np.array(dist_new)
  sections = np.array(sections_new)
  mls = MultiLineString(lss_new)
  p_coords = np.array(p_coords)
  p_dist = np.array(p_dist)
  p_sections = np.array(p_sections)

  return coords, dist, sections, mls, p_coords, p_dist, p_sections



################################################################################
# downstream distance ##########################################################
################################################################################

def downstream_distance(coords, sections, mls, mpol, dns, \
                        with_distance_to_downstream_edge = True):

  # initialize
  coords_new = coords.copy()
  sections_new = sections.copy()

  # convert multi-polygons into list of linear rings
  lrs = []
  for pol in mpol.geoms:
    lrs.append(pol.exterior)
    for interior in pol.interiors:
      lrs.append(interior)

  # convert multi-line string into list of line strings
  lss = list(mls.geoms)

  # initialize buffer array with downstream segments
  buf = []
  for dn in dns:
    tmp = np.argwhere(sections_new == dn)[0]
    buf.append(tmp[0])
  buf = np.array(buf)

  # array of section lengths
  lens = np.zeros(sections_new.shape[0])
  for i in range(len(lens)):
    lens[i] = lss[i].length

  # initialize downstream distance array
  dist = np.zeros(coords_new.shape[0]) + np.inf
  for i in range(len(buf)):
    s = buf[i]
    n0 = dns[i]
    if n0 == sections_new[s, 0]:
      n1 = sections_new[s, 1]
    else:
      n1 = sections_new[s, 0]

    # downstream distance of tip point is the distance to channel edges
    if with_distance_to_downstream_edge:
      p0 = Point(coords_new[n0, 0], coords_new[n0, 1])
      for lr in lrs:
        dist[n0] = np.minimum(dist[n0], p0.distance(lr))
    else:
      dist[n0] = 0

    # add section length to obtain downstream distance of other section node
    dist[n1] = dist[n0] + lens[s]

  # as long as the buffer array is not empty
  while len(buf) > 0:

    # initialize list of connected sections
    con = []

    # for each section in the buffer array
    for s in buf:
      n0 = sections_new[s, 0]
      n1 = sections_new[s, 1]

      # look for connected sections
      for n in [n0, n1]:
        ind = np.where(sections_new == n)
        for nc in ind[0]:
          if nc != n:
            con.append(nc)

    # convert list into array
    con = np.array(con)

    # reset buffer array as a list
    buf = []

    # for each section in the connected array
    for s in con:
      n0 = sections_new[s, 0]
      n1 = sections_new[s, 1]

      # update downstream distances
      dist0 = dist[n1] + lens[s]
      dist1 = dist[n0] + lens[s]
      if dist0 < dist[n0]:
        dist[n0] = dist0
        buf.append(s)
      if dist1 < dist[n1]:
        dist[n1] = dist1
        buf.append(s)

    # update section and line string orientation
    for s in range(sections_new.shape[0]):
      n0 = sections_new[s, 0]
      n1 = sections_new[s, 1]
      dist0 = dist[n0]
      dist1 = dist[n1]
      if dist1 < dist0:
        sections_new[s, 0] = n1
        sections_new[s, 1] = n0
        ls_coords = list(lss[s].coords)
        ls_coords.reverse()
        lss[s] = LineString(ls_coords)

    # convert list into array
    buf = np.array(buf)


  #########################################################
  # split skeleton sections at equal downstream distances #
  #########################################################

  # initialize list of section indices to delete
  trash = []

  # convert arrays into lists
  coords_new = coords_new.tolist()
  dist = dist.tolist()
  sections_new = sections_new.tolist()

  # for each section
  for s in range(len(sections_new)):
    n0 = sections_new[s][0]
    n1 = sections_new[s][1]

    # if section length higher than the difference of downstream distances
    if lens[s] - np.abs(dist[n1] - dist[n0]) > 1e-6:

      # equal downstream distance
      dist_eq = .5 * (lens[s] + dist[n0] + dist[n1])

      # equal downstream distance point
      p_eq = lss[s].interpolate(dist_eq - dist[n0])

      # cut line string at equal downstream distance point
      for i in range(len(lss[s].coords)):
        pi = Point(lss[s].coords[i])
        if lss[s].project(pi) == dist_eq - dist[n0]:
          coords_new.append([p_eq.x, p_eq.y])
          dist.append(dist_eq)
          sections_new.append([n0, len(coords_new) - 1])
          sections_new.append([n1, len(coords_new) - 1])
          lss.append(LineString(lss[s].coords[:i + 1]))
          lss.append(LineString(lss[s].coords[i:]))
          trash.append(s)
          break
        if lss[s].project(pi) > dist_eq - dist[n0]:
          coords_new.append([p_eq.x, p_eq.y])
          dist.append(dist_eq)
          sections_new.append([n0, len(coords_new) - 1])
          sections_new.append([n1, len(coords_new) - 1])
          lss.append(LineString(lss[s].coords[:i] + [(p_eq.x, p_eq.y)]))
          lss.append(LineString([(p_eq.x, p_eq.y)] + lss[s].coords[i:]))
          trash.append(s)
          break

  # delete split sections and line strings
  trash.reverse()
  for s in trash:
    del sections_new[s]
    del lss[s]

  # convert list into arrays
  coords_new = np.array(coords_new)
  dist = np.array(dist)
  sections_new = np.array(sections_new)

  # convert list of section line strings into multi-line string
  mls_new = MultiLineString(lss)

  return coords_new, dist, sections_new, mls_new