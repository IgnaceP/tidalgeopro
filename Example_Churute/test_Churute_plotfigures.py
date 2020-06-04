"""
Script to plot the figures of the Churute subregion test

author: Ignace Pelckmans
            (University of Antwerp, Belgium)

"""
#--------------------#
#-- import modules --#
#--------------------#

import numpy as np
import matplotlib.pyplot as plt

#-------------------------#
#-- plot clean skeleton --#
#-------------------------#

def plotFinalSkeleton(mpol, mls, coords, p_dist, p_coords):
      plt.figure()
      for pol in mpol.geoms:
        xp, yp = pol.exterior.xy
        plt.plot(xp, yp, linewidth = .5, color = 'k')
        for j in range(len(pol.interiors)):
          xp, yp = pol.interiors[j].xy
          plt.plot(xp, yp, linewidth = .5, color = 'k')
      for ls in mls.geoms:
        xp, yp = ls.xy
        plt.plot(xp, yp, linewidth = 1)
      for coord in coords:
        xp, yp = list(coord)
        plt.plot(xp, yp, 'k.', markersize = 1)
      for i in range(len(p_dist)):
        if np.mod(p_dist[i], 100) == 0:
          xp, yp = list(p_coords[i, :])
          plt.text(xp, yp, p_dist[i], fontsize = 2)
      plt.axis('scaled')
      plt.xticks([])
      plt.yticks([])
      plt.savefig('./figures/final_skeleton.jpg', bbox_inches = 'tight')
      plt.close()

#------------------------#
#-- plot Stream Orders --#
#------------------------#

def plotStreamOrders(mpol, mls, coords, orders):
    f = plt.figure(figsize = (50,50))
    for pol in mpol.geoms:
      xp, yp = pol.exterior.xy
      plt.plot(xp, yp, linewidth = .5, color = 'k')
      for j in range(len(pol.interiors)):
        xp, yp = pol.interiors[j].xy
        plt.plot(xp, yp, linewidth = .5, color = 'k')
    colors = ['silver','green', 'red', 'yellow', 'blue', 'purple', 'pink', 'bluegreen', 'silver']
    t = 1
    for ls in mls.geoms:
      xp, yp = ls.xy
      plt.plot(xp, yp, linewidth = 3, color = colors[int(orders[t-1])])
      t += 1
    for c in coords:
        x,y = c
        plt.scatter(x, y, 40, marker = "o")
    plt.axis('scaled')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./figures/stream_orders.jpg', bbox_inches = 'tight')
    plt.close()
