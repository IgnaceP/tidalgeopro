""" UPL: Unchanneled Path Length

This module allows to calculate the unchanneled path length in a watershed with a channel network

Author: Olivier Gourgue
       (University of Antwerp, Belgium & Boston University, MA, United States)

"""


import numpy as np
from scipy import spatial



################################################################################
# compute upl ##################################################################
################################################################################

def compute_upl(x, y, creek):

  """ Compute the unchanneled path length from a boolean creek field

  Required parameters:
  x, y (NumPy arrays of shape (n)) grid node coordinates
  creek (NumPy array of shape (n, m) and type logical): True if creek, False otherwise (m is number of time steps)

  Returns:
  NumPy array of shape (n, m): unchanneled path length

  """

  # initialize
  upl = np.zeros(creek.shape)

  # case of one time step
  if creek.ndim == 1:
    creek = creek.reshape((creek.shape[0], 1))
    upl = upl.reshape((upl.shape[0], 1))

  # for each time step
  for i in range(creek.shape[1]):

    # creek nodes
    creek_ind = np.flatnonzero(creek[:, i])
    creek_xy = np.array([x[creek_ind], y[creek_ind]]).T

    # non-creek nodes
    non_creek_ind = np.flatnonzero(creek[:, i] == 0)
    non_creek_xy = np.array([x[non_creek_ind], y[non_creek_ind]]).T

    # unchanneled path length
    tree = spatial.KDTree(creek_xy)
    non_creek_upl, ind = tree.query(non_creek_xy)
    upl[non_creek_ind, i] = non_creek_upl

  if upl.shape[1] == 1:
    upl = upl.reshape((upl.shape[0]))

  # return
  return upl