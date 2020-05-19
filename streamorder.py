""" Stream Order

This module the ordering of a channel network following an extended version of the Strahler method

Author: Ignace Pelckmans
       (University of Antwerp, Belgium)

"""

import numpy as np

def findPaths(conn, downstream_nodes = 0):
    """
    Function to find loops in a connectivity table

    author: Ignace Pelckmans, University of Antwerp, Belgium

    Args:
        conn: (Required) Numpy array of shape (m, 2) representing section connectivity table (i-th row gives node indices of the i-th section)
        downstream_nodes: (Optional, defaults to 0) list or int of node id('s) of the downstream nodes

    Returns:
        A list of al possible paths from the downstream point
    """

    # total number of segments
    n_segs = np.shape(conn)[0]
    # total number of nodes
    n_nodes = np.max(conn)+1
    # total number of loops to be found
    n_loops = n_segs - n_nodes + 1
    # sort the connectivity table
    conn_sorted = np.sort(conn, axis = 1)
    # downstram node can be both a single number or a list
    if type(downstream_nodes) != list: downstream_nodes = [downstream_nodes]

    # initialize a list to store all paths
    paths = [[dn] for dn in downstream_nodes]

    # initialize variables for while loop:
    # make sure the while loop starts
    new_paths = [1]

    # keep generating new paths as long as there are new ones which meet all criteria
    while len(new_paths)>0:
        # initialize list for new paths
        new_paths = []
        # build further on each path
        for p in paths:
            # check for each node if it can be part of a path
            for n in range(n_nodes):
                # a new possible path
                new_path = np.concatenate((p, np.asarray([n])))
                # a path should meet the following conditions:
                # it does not simply returns to the previous point
                # therefore, it won't catch single loops
                # it can only travel in between connected nodes per step
                # it can only pass by a point twice (to avoid endless looping)
                # the path is not already listed
                if np.unique(new_path[-3:]).shape == new_path[-3:].shape and\
                np.sum(np.all((conn_sorted == np.sort(new_path[-2:])), axis = 1)) > 0 and\
                max(np.bincount(new_path)) < 3 and \
                not list(new_path) in [list(item) for item in paths] and\
                not list(new_path) in [list(item) for item in new_paths]:
                    new_paths.append(new_path)

        # append the new paths to the existing list
        paths += new_paths

    return paths


def streamOrder(conn, downstream_nodes = 0):
    """
    Function to calculate the stream order of each segment of a stream network using the Strahler method, expanded to segment which form loops

    author: Ignace Pelckmans, University of Antwerp, Belgium

    Args:
        conn: (Required) Numpy array of shape (m, 2): section connectivity table (i-th row gives node indices of the i-th section)
        downstream_nodes: (Optional, defaults to 0) list or int of node id('s) of the downstream nodes

    Returns:
        Numpy array of shape(m, 1): section orders
    """

    # downstream node can be both a single number or a list
    if type(downstream_nodes) != list: downstream_nodes = [downstream_nodes]
    # in the case of multiple downstream nodes, there will be splits instead of confluences (water will split into two channels when flowing downstream)
    # therefore the Strahler algoritm will get stuck
    # the extended Strahler algoritm can break this if it detects one or multiple loops
    # if the ordering algoritm reaches a node which is part of a loop, it does not need two
    # ordered segments to order the third one, instead it only need one ordered and will assign
    # the two remaining segments the same order
    # therefore we create fictional segments between downstream nodes to simulate such a situation
    # in case of an even number of downstream nodes we create a single loop between each pair of downstream point
    # in case of an un even number of downstream nodes (dn_n), only dn_n - 1 downstreams have to be connected

    if len(downstream_nodes) > 1:
       dn = np.asarray(downstream_nodes).copy()
       if len(dn)%2 != 0: dn = dn[:-1]
       fict_conn = np.repeat(np.asarray([[dn[i],dn[i+1]] for i in range(0,len(dn),2)]), 2, axis = 0)
       fict_segs = np.shape(fict_conn)[0]
       conn = np.vstack((conn, fict_conn))
    # total number of segments
    n_segs = np.shape(conn)[0]
    # total number of nodes
    n_nodes = np.max(conn)+1
    # numpy array to store the order of the downstream node
    nodes = np.zeros(n_nodes)
    # numpy array to store the order of the segments
    segs = np.zeros(n_segs)
    # connectivity list of lists which stores the connected segments per node
    conn_nodes = [np.where(conn == n)[0] for n in range(0, n_nodes)]
    # total number of loops to be found
    n_loops = n_segs - n_nodes + 1


    ############################################
    ### check validity of connectivity table ###
    ############################################

    # loop over the node id's
    for n in range(0,n_nodes):
            count_node = np.count_nonzero(conn == n)
            # if the the first order segments form a single loop, the nodes will occur twice in the connectivity table
            if count_node > 3:
                print('ERROR! Connectivity table is invalid!')
                exit()


    ############################
    ### first order segments ###
    ############################

    # loop over the node id's
    for n in range(0,n_nodes):
        if not n in downstream_nodes:
            # a node has a downstream order 1 segment as it occurs only once or twice in the connectivity table
            count_node = np.count_nonzero(conn == n)
            # if the the first order segments form a single loop, the nodes will occur twice in the connectivity table
            if count_node < 3:
                segs[np.where(conn == n)[0]] = 1
                nodes[n] = 1


    ##################
    ### find loops ###
    ##################

    # store all nodes within a loop

    loop_nodes = []

    # --------------------------------------- #
    # find loops of 2 nodes ( = single loops) #
    # --------------------------------------- #

    # sort the connectivity table so that segs with the same connected nodes, appear the same
    conn_sorted = np.sort(conn, axis = 1)
    # loop over all segments
    for i in range(conn_sorted.shape[0]):
        # if two segments have the same connected nodes, they form a single loop
        if np.sum(np.all(conn_sorted == conn_sorted[i,:], axis = 1)) == 2:
            for n in conn_sorted[i,:]:
                loop_nodes.append(n)

    # --------------- #
    # find long loops #
    # --------------- #

    # this is only necessary if not all loops to be found are single loops
    if len(loop_nodes)/4 != n_loops:
        # initialize a list of al non-looping paths
        nonloop_paths = []
        # list all possible flow paths (heavy computation, might take time depending on the number of nodes)
        paths = findPaths(conn, downstream_nodes = downstream_nodes)
        # loop over all paths
        for p in paths:

            # a loop exist if a node is visited twice
            p_count = np.bincount(p)
            if max(p_count) == 2:
                # list which nodes occur twice in the path
                doubles = np.where(p_count == 2)
                # if the number after the first time a node occurs in a path
                # is the same number as the number before it's second occurence
                # it is not part of the loop
                # for example, the path: [0,1,2,3,1,0]
                #
                # 4 .          . 5
                #    \         /
                #     \       /
                #    2 .- - -. 3
                #       \   /
                #        \ /
                #         . 1
                #         |
                #         . 0
                #
                # to go back to 0, the path has to pass by 1 again and so node 0
                # is not part of the loop
                #
                for d in doubles[0]:
                    d_1, d_2 = np.where(p == d)[0]
                    if p[d_1+1] != p[d_2-1]:
                        loop_nodes.append(d)
                    else:
                        break
            else:
                nonloop_paths.append(p)
    else:
        nonloop_paths = findPaths(conn, downstream_nodes = downstream_nodes)

    # remove duplicates
    # now loop_nodes is an array with indices of nodes which are part of the loop
    loop_nodes = np.sort(np.unique(np.asarray(loop_nodes)))

    # ------------------ #
    # find loop segments #
    # ------------------ #

    # initialize an array to store ones for the segs which are part of the loop and zeros for segments which are not
    loop_segs = np.zeros(np.shape(segs))
    # loop over all segments
    for s in range(n_segs):
        # the two connected nodes of a segment
        n1, n2 = conn[s,:]
        # both should be a loop node to make the segment part of the loop
        if np.any(loop_nodes == n1) and np.any(loop_nodes == n2):
            loop_segs[s] = 1

    ############################
    ### Assign Stream Orders ###
    ############################


    # assign orders untill all segments are ordered but the fictional segments added to create a loop between the downstream nodes
    while np.min(segs[0:-fict_segs]) == 0:
        # list to store the newly ordered segments
        new = []
        # list to store the newly assigned orders
        new_orders = []
        # loop over all nodes
        for i in range(0, n_nodes):
            # a node shouldn't have a downstream segment with order 1
            # the downstream node has no donwtream segment so it should not be ordered
            # we can only order a segment if it is connected to a node which is connected to two already ordered segments
            if nodes[i] == 0 and\
            i not in downstream_nodes and \
            np.count_nonzero(segs[conn_nodes[i]] == 0) == 1:
                # loop over all segments connected to the node
                for s in conn_nodes[i]:
                    # if that segment is unordered or part of a loop, we can order it
                    # reordering is only allowed for loop segments as they are artifically ordered by the breaker algoritm
                    # however, later on this might result in a situation where a more appropriate order can be assigned and
                    # this is allowed. Otherwide, reordering is forbidden.
                    if segs[s] == 0 or loop_segs[s] == 1:
                        new.append(s)
                        # define the new order
                        # if the two segments in this node have the same order, the newly ordered segment will
                        # be assigned one higher order (for example, two 1-streams arrive at a node, than the third stream
                        # will have order 2) if that node is not part of a loop:
                        #
                        #    4.     .5                3.
                        #      \   /                   |
                        #       \ /                   2.
                        #    2.  .3                   / \
                        #      \ |                    \ /
                        #       \|                    1.
                        #       1.                     |
                        #        |                    0.
                        #       0.
                        #
                        # Left: segments 4-3 and 3-5, both first-order segments, confluence in 3 and therefore, the segment 3-1 will have order 2.
                        # segment 2-1 (first-order) will confluence with 3-1 (second-order) at node 1 and so, segment 1-0 will adopt the highest order
                        # which is order 2 (from segment 3-1).
                        #
                        # Right: both streams in the single loop (between 2 and 1) will be assigned order 1 (by the breaker algoritm, tbc)
                        # however, at node 1, they will not confluence into a stream with order 2 but remain order 1
                        if np.count_nonzero(segs[conn_nodes[i]] == max(segs[conn_nodes[i]])) == 2:
                            if not i in loop_nodes:
                                new_orders.append(max(segs[conn_nodes[i]])+1)
                            elif i in loop_nodes and loop_segs[s] == 1:
                                new_orders.append(max(segs[conn_nodes[i]])+1)
                            else:
                                new_orders.append(max(segs[conn_nodes[i]]))
                        else: new_orders.append(max(segs[conn_nodes[i]]))

        # assign the new orders
        segs[new] = new_orders

        # in case of a loop, the upper algoritm would 'get stuck' as it reaches a loop.
        # it will never have a node which have to incoming segments and thus it it will keep running forever
        # therefore we look for an unblocking node which which is part of a lopo
        # and which have one connected segment which is already ordered. That node we call the 'breaker'
        if len(new_orders) == 0 and np.min(segs) == 0:
            # look for the best node to unblock the ordering
            # that is the node which still has two connected segments which are unordered
            # and which is located the furthest away from the downstream nodes

            # canditate nodes (nodes which are part of a loop and have one ordered connected segment)
            breaker_candidates = [n for n in range(n_nodes) if np.count_nonzero(segs[conn_nodes[n]] == 0) == 2 and n in loop_nodes]

            # the breaker will be the one from the candidates which has the longest path to a downstream point
            breaker_candidates_shortestpathslengths = []
            for b in breaker_candidates:
                candidate_pathslength = []
                # search for the shortest path to each candidate from any downstream point
                for p in nonloop_paths:
                    if p[-1] == b: candidate_pathslength.append(len(p))
                    else: candidate_pathslength.append(float('inf'))
                # make a list with the length of each shortest path between a breaker candidate and a downstream node
                breaker_candidates_shortestpathslengths.append(len(nonloop_paths[np.argmin(candidate_pathslength)]))
            breaker = breaker_candidates[np.argmax(breaker_candidates_shortestpathslengths)]

            nodes[breaker] = max(segs[conn_nodes[breaker]])
            # all unordered segment (which logically, are part of the loop) will get the order of the incoming segment
            for s in conn_nodes[breaker]:
                if segs[s] == 0:
                    new.append(s)
                    new_orders.append(max(segs[conn_nodes[breaker]]))

        # assign new orders
        segs[new] = new_orders

    print(loop_nodes)
    return segs[0:-fict_segs]