# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)


import numpy as np


def plot_result(x, y, indices=None, plot=None, plot_kwargs_cb=None,
                ls=('-', '--', ':', '-.'),
                c=('k', 'r', 'g', 'b', 'c', 'm', 'y'),
                m=('o', 'v', '8', 's', 'p', 'x', '+', 'd', 's'),
                m_lim=-1, lines=None, interpolate=None, interp_from_deriv=None,
                names=None, post_processor=None):
    """
    Parameters
    ----------
    x: array_like:
        Values of the independent variable.
    y: array_like:
        Values of the independent variable. This must hold
        ``y.shape[0] == len(x)``, plot_results will draw
        ``y.shape[1]`` lines. If ``interpolate != None``
        y is expected two be three dimensional, otherwise two dimensional.
    indices: iterable of integers
        what indices to plot
    plot: callback (default: None)
        If None, use ``matplotlib.pyplot.plot``
    plot_kwargs_cb: callback(int) -> dict
        kwargs for plot for each index (0:len(y)-1)
    ls: iterable
        linestyles to cycle through (only used if plot and plot_kwargs_cb
        are both None)
    c: iterable
        colors to cycle through (only used if plot and plot_kwargs_cb
        are both None)
    m: iterable
        markers to cycle through (only used if plot and plot_kwargs_cb
        are both None and m_lim > 0)
    m_lim: int (default: 0)
        limit number of points to use markers instead of lines
    lines: None
        default: draw between markers unless we are interpolating as well.
    interpolate: (default: None)
        to use for interpolation using 3rd dimension of y as input.
    interp_from_deriv: callback (default: None)
        when None: ``scipy.interpolate.BPoly.from_derivatives``
    post_processor: callback (default: None)
    """
    if indices is None:
        indices = range(y.shape[-1])
    if plot is None:
        from matplotlib.pyplot import plot
    if plot_kwargs_cb is None:
        def plot_kwargs_cb(idx, lines=False, markers=False, labels=None):
            kwargs = {'c': c[idx % len(c)]}

            if lines:
                kwargs['ls'] = ls[idx % len(ls)]
                if isinstance(lines, float):
                    kwargs['alpha'] = lines
            else:
                kwargs['ls'] = 'None'

            if markers:
                kwargs['marker'] = m[idx % len(m)]

            if labels:
                kwargs['label'] = labels[idx]
            return kwargs
    else:
        plot_kwargs_cb = plot_kwargs_cb or (lambda idx: {})

    def post_process(x, y):
        if post_processor is None:
            return x, y
        else:
            return post_processor(x, y)

    x_post, y_post = post_process(x, y[:, 0, :] if interpolate and
                                  y.ndim == 3 else y)
    if lines is None:
        lines = interpolate in (None, False)
    markers = len(x) < m_lim
    for idx in indices:
        plot(x_post, y_post[:, idx], **plot_kwargs_cb(
            idx, lines=lines, labels=names))
        if markers:
            plot(x_post, y_post[:, idx], **plot_kwargs_cb(
                idx, lines=False, markers=markers, labels=names))

    if interpolate:
        if interpolate is True:
            interpolate = 20

        if isinstance(interpolate, int):
            if interpolate > 0:
                x_plot = np.concatenate(
                    [np.linspace(a, b, interpolate)
                     for a, b in zip(x[:-1], x[1:])])
            elif interpolate < 0:
                x_plot = np.concatenate([
                    np.logspace(np.log10(a), np.log10(b),
                                -interpolate) for a, b
                    in zip(x[:-1], x[1:])])
        else:
            x_plot = interpolate

        if interp_from_deriv is None:
            import scipy.interpolate
            interp_from_deriv = scipy.interpolate.BPoly.from_derivatives

        y2 = np.empty((x_plot.size, y.shape[-1]))
        for idx in range(y.shape[-1]):
            interp_cb = interp_from_deriv(x, y[..., idx])
            y2[:, idx] = interp_cb(x_plot)

        x_post2, y_post2 = post_process(x_plot, y2)
        for idx in range(y.shape[-1]):
            plot(x_post2, y_post2[:, idx], **plot_kwargs_cb(
                idx, lines=True, markers=False))
        return x_post2, y_post2
    return x_post, y_post


def plot_phase_plane(x, y, indices=None, post_processor=None, plot=None,
                     names=None, **kwargs):
    if indices is None:
        indices = (0, 1)
    if len(indices) != 2:
        raise ValueError('Only two phase variables supported at the moment')
    if plot is None:
        import matplotlib.pyplot as plt
        plot = plt.plot
        if names is not None:
            plt.xlabel(names[indices[0]])
            plt.ylabel(names[indices[1]])

    if post_processor is not None:
        x, y = post_processor(x, y)

    plot(y[:, indices[0]], y[:, indices[1]], **kwargs)
