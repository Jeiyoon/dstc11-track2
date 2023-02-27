# PyVot Python Variational Optimal Transportation
# Author: Liang Mi <icemiliang@gmail.com>
# Date: April 28th 2020
# Licence: MIT


import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vot_numpy import VOT
import utils

np.random.seed(19)

# ------------ Generate data ------------- #
N0, N1, N2 = 1000, 200, 200

mean1, cov1 = [0., -0.2], [[0.05, 0], [0, 0.05]]
x11, x12 = np.random.multivariate_normal(mean1, cov1, N0).T
x1 = np.stack((x11, x12), axis=1).clip(-0.99, 0.99)

mean2, cov2 = [0.5, 0.5], [[0.01, 0], [0, 0.01]]
x21, x22 = np.random.multivariate_normal(mean2, cov2, N1).T
x2 = np.stack((x21, x22), axis=1).clip(-0.99, 0.99)

mean3, cov3 = [-0.5, 0.5], [[0.01, 0], [0, 0.01]]
x31, x32 = np.random.multivariate_normal(mean3, cov3, N2).T
x3 = np.stack((x31, x32), axis=1).clip(-0.99, 0.99)


K = 3
mean, cov = [0.0, 0.0], [[0.02, 0], [0, 0.02]]
y1, y2 = np.random.multivariate_normal(mean, cov, K).T

# y: ndarray(3, 2), where 3 is K
y = np.stack((y1, y2), axis=1).clip(-0.99, 0.99)
# x: ndarray(1400, 2)
x = np.concatenate((x1, x2, x3), axis=0)


# xmin, xmax, ymin, ymax = -.7, .8, -.65, .8
plt.close()


# ---------------kmeans--------------- #

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, init=y).fit(x)

# label: ndarray(1400, )
label = kmeans.predict(x)
# ndarray(3, 2)
test = kmeans.cluster_centers_
# Note that "y" is re-used
y = kmeans.cluster_centers_

# ndarray(3, 4)
color_map = np.array([[237, 125, 49, 255],
                      [112, 173, 71, 255],
                      [91, 155, 213, 255]]) / 255

# ---------------VWB---------------

for reg in [0.5, 2, 1e9]:

    # y_copy: ndarray(3, 2)
    y_copy = y.copy()
    # x_copy: ndarray(1400, 2)
    x_copy = x.copy()

    vot = VOT(y_copy, [x_copy], verbose=False)
    vot.cluster(lr=0.5, max_iter_h=1000, max_iter_y=1, beta=0.5, reg=reg)

    # idx: list(1) / ndarray(1400,) / cluster # (e.g., 0, 1, 2)
    idx = vot.idx
    fig = plt.figure(figsize=(4, 4))
    # ce: ndarray(1400, 4)
    ce = color_map[idx[0]]
    # vot.y: ndarray(3, 2)
    # vot.x: list(1) / ndarray(1400, 2)
    # utils.scatter_otsamples(vot.y, vot.x[0], size_p=30, marker_p='o', color_x=ce,
    #                         xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, facecolor_p='none')
    utils.scatter_otsamples(vot.y, vot.x[0], size_p=30, marker_p='o', color_x=ce,
                            facecolor_p='none')
    # plt.axis('off')
    plt.show()
    # plt.savefig(str(int(reg)) + ".svg", bbox_inches='tight')
    # plt.savefig(str(reg) + ".png", dpi=300, bbox_inches='tight')


plt.figure(figsize=(4, 4))

ce = color_map[label]
utils.scatter_otsamples(y, x, size_p=30, marker_p='o', color_x=ce,
                        facecolor_p='none')
# plt.axis('off')
plt.show()
# plt.savefig("0.svg", bbox_inches='tight')
# plt.savefig("0.png", dpi=300, bbox_inches='tight')
# plt.close()