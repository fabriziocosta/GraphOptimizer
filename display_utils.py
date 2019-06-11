#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import matplotlib.pyplot as plt


def display_estimate(
        estimator, graphs=None, targets=None, selected_graphs=None,
        target_graph=None,
        grid_n_points=500, size=9,
        cmap='gist_heat', cmap2='rainbow'):  # 'bone'
    """display_estimate."""
    def make_mesh(x_min, x_max, y_min, y_max, grid_n_points):
        x_width = x_max - x_min
        y_width = y_max - y_min
        width = max(x_width, y_width)
        x_cent = (x_max - x_min) / 2 + x_min
        y_cent = (y_max - y_min) / 2 + y_min
        x_min, x_max = x_cent - width, x_cent + width
        y_min, y_max = y_cent - width, y_cent + width
        xx, yy = np.meshgrid(np.linspace(
            x_min, x_max, grid_n_points),
            np.linspace(y_min, y_max, grid_n_points))
        return xx, yy

    data_x = estimator.transform(graphs)
    sdata_x = estimator.transform(selected_graphs)
    if target_graph is not None:
        target_data_x = estimator.transform([target_graph])
    all_data = np.vstack([data_x, sdata_x])

    # make grid
    minx, maxx = min(all_data[:, 0]), max(all_data[:, 0])
    miny, maxy = min(all_data[:, 1]), max(all_data[:, 1])
    dx = (maxx - minx) / 7.0
    dy = (maxy - miny) / 7.0
    x_min, x_max = minx - dx, maxx + dx
    y_min, y_max = miny - dy, maxy + dy
    xx, yy = make_mesh(x_min, x_max, y_min, y_max, grid_n_points)
    test_x = (np.vstack((xx.ravel(), yy.ravel())).T)
    z = estimator.exp_imp_estimator.predict(test_x)
    eis = z.reshape(xx.shape)
    ei_levels = [np.percentile(eis, i)
                 for i in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]]
    mu, var = estimator.exp_imp_estimator.predict_mean_and_variance(test_x)
    mus = mu.reshape(xx.shape)
    mu_levels = [np.percentile(mus, i)
                 for i in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]]

    args = dict(interpolation='nearest', extent=(
        xx.min(), xx.max(), yy.min(), yy.max()), origin='lower')
    args2 = dict(colors='k', linewidths=.5)

    xs, ys = data_x.T
    sxs, sys = sdata_x.T
    if target_graph is not None:
        txs, tys = target_data_x.T

    plt.figure(figsize=(2 * size, size))
    plt.subplot(121)
    plt.imshow(mus, cmap=cmap, **args)
    plt.contour(xx, yy, mus, levels=mu_levels, **args2)
    plt.scatter(xs, ys, s=100, edgecolors='grey',
                c=targets, cmap=cmap, zorder=2)
    plt.scatter(sxs, sys, s=60, c='g',
                marker='P', edgecolor='w', linewidth=.5, zorder=3)
    if target_graph is not None:
        plt.scatter(txs, tys, s=60, c='w',
                    marker='X', edgecolor='k', linewidth=.5, zorder=3)

    plt.title('Mean')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.subplot(122)
    plt.imshow(eis, cmap=cmap2, **args)
    plt.contour(xx, yy, eis, levels=ei_levels, **args2)
    plt.scatter(xs, ys, s=100, edgecolors='grey',
                c=targets, cmap=cmap, zorder=2)
    plt.scatter(sxs, sys, s=60, c='g',
                marker='P', edgecolor='w', linewidth=.5, zorder=3)
    if target_graph is not None:
        plt.scatter(txs, tys, s=60, c='w',
                    marker='X', edgecolor='k', linewidth=.5, zorder=3)

    plt.title('Expected Improvement')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()
