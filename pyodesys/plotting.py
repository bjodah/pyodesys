# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)


import numpy as np


def plot_result(x, y, params=(), indices=None, plot=None, plot_kwargs_cb=None,
                ls=('-', '--', ':', '-.'),
                c=('k', 'r', 'g', 'b', 'c', 'm', 'y'),
                m=('o', 'v', '8', 's', 'p', 'x', '+', 'd', 's'),
                m_lim=-1, lines=None, interpolate=None, interp_from_deriv=None,
                names=None, post_processors=(), xlabel=None, ylabel=None,
                xscale=None, yscale=None, latex_attr='latex_name'):
    """
    Plot the depepndent variables vs. the independent variable

    Parameters
    ----------
    x : array_like
        Values of the independent variable.
    y : array_like
        Values of the independent variable. This must hold
        ``y.shape[0] == len(x)``, plot_results will draw
        ``y.shape[1]`` lines. If ``interpolate != None``
        y is expected two be three dimensional, otherwise two dimensional.
    params : array_like
        Parameters used.
    indices : iterable of integers
        What indices to plot (default: None => all).
    plot : callback (default: None)
        If None, use ``matplotlib.pyplot.plot``.
    plot_kwargs_cb : callback(int) -> dict
        Keyword arguments for plot for each index (0:len(y)-1).
    ls : iterable
        Linestyles to cycle through (only used if plot and plot_kwargs_cb
        are both None).
    c : iterable
        Colors to cycle through (only used if plot and plot_kwargs_cb
        are both None).
    m : iterable
        Markers to cycle through (only used if plot and plot_kwargs_cb
        are both None and m_lim > 0).
    m_lim : int (default: -1)
        Upper limit (exclusive, number of points) for using markers instead of
        lines.
    lines : None
        default: draw between markers unless we are interpolating as well.
    interpolate : bool or int (default: None)
        Density-multiplier for grid of independent variable when interpolating
        if True => 20. negative integer signifies log-spaced grid.
    interp_from_deriv : callback (default: None)
        When ``None``: ``scipy.interpolate.BPoly.from_derivatives``
    post_processors : iterable of callback (default: tuple())

    """
    import matplotlib.pyplot as plt

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
                if latex_attr:
                    try:
                        kwargs['label'] = '$' + getattr(labels[idx],
                                                        latex_attr) + '$'
                    except (AttributeError, TypeError):
                        pass
            return kwargs
    else:
        plot_kwargs_cb = plot_kwargs_cb or (lambda idx: {})

    def post_process(x, y, p):
        for post_processor in post_processors:
            x, y, p = post_processor(x, y, p)
        return x, y, p

    if interpolate is None:
        interpolate = y.ndim == 3 and y.shape[1] > 1

    x_post, y_post, params_post = post_process(x, y[:, 0, :] if interpolate and
                                               y.ndim == 3 else y, params)
    if indices is None:
        indices = range(y_post.shape[-1])  # e.g. PartiallySolvedSys
    if lines is None:
        lines = interpolate in (None, False)
    markers = len(x) < m_lim
    for idx in indices:
        plot(x_post, y_post[:, idx], **plot_kwargs_cb(
            idx, lines=lines, labels=names))
        if markers:
            plot(x_post, y_post[:, idx], **plot_kwargs_cb(
                idx, lines=False, markers=markers, labels=names))

    if xlabel is None:
        try:
            plt.xlabel(x_post.dimensionality.latex)
        except AttributeError:
            pass
    else:
        plt.xlabel(xlabel)

    if ylabel is None:
        try:
            plt.ylabel(y_post.dimensionality.latex)
        except AttributeError:
            pass
    else:
        plt.ylabel(ylabel)

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

        x_post2, y_post2, params2 = post_process(x_plot, y2, params)
        for idx in indices:
            plot(x_post2, y_post2[:, idx], **plot_kwargs_cb(
                idx, lines=True, markers=False))
        return x_post2, y_post2

    if xscale is not None:
        plt.gca().set_xscale(xscale)
    if yscale is not None:
        plt.gca().set_yscale(yscale)
    return x_post, y_post


def plot_phase_plane(x, y, params=(), indices=None, post_processors=(),
                     plot=None, names=None, **kwargs):
    """ Plot the phase portrait of two dependent variables

    Parameters
    ----------
    x: array_like
        Values of the independent variable
    y: array_like
        Values of the dependent variables
    params: array_like
        parameters
    indices: pair of integers (default: None)
        what dependent variable to plot for (None => (0, 1))
    post_processors: iterable of callbles
        see :class:OdeSystem
    plot: callable (default: None)
        Uses matplotlib.pyplot.plot if None
    names: iterable of strings
        labels for x and y axis
    \*\*kwargs:
        keyword arguemtns passed to ``plot()``

    """
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

    for post_processor in post_processors:
        x, y, params = post_processor(x, y, params)

    plot(y[:, indices[0]], y[:, indices[1]], **kwargs)
