# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)

from math import log

import numpy as np


def _set_scale(cb, argstr):
    if argstr.count(';') == 0:
        cb(argstr)
    else:
        arg, kw = argstr.split(';')
        cb(arg, **eval('dict(%s)' % kw))


def plot_result(x, y, indices=None, plot_kwargs_cb=None, ax=None,
                ls=('-', '--', ':', '-.'),
                c=('k', 'r', 'g', 'b', 'c', 'm', 'y'),
                m=('o', 'v', '8', 's', 'p', 'x', '+', 'd', 's'),
                m_lim=-1, lines=None, interpolate=None, interp_from_deriv=None,
                names=None, latex_names=None, xlabel=None, ylabel=None,
                xscale=None, yscale=None, legend=False, yerr=None):
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
    indices : iterable of integers
        What indices to plot (default: None => all).
    plot : callback (default: None)
        If None, use ``matplotlib.pyplot.plot``.
    plot_kwargs_cb : callback(int) -> dict
        Keyword arguments for plot for each index (0:len(y)-1).
    ax : Axes
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

    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.subplot(1, 1, 1)
    if plot_kwargs_cb is None:
        def plot_kwargs_cb(idx, lines=False, markers=False, labels=None):

            kw = {'c': c[idx % len(c)]}

            if lines:
                kw['ls'] = ls[idx % len(ls)]
                if isinstance(lines, float):
                    kw['alpha'] = lines
            else:
                kw['ls'] = 'None'

            if markers:
                kw['marker'] = m[idx % len(m)]

            if labels:
                kw['label'] = labels[idx]
            return kw
    else:
        plot_kwargs_cb = plot_kwargs_cb or (lambda idx: {})

    if interpolate is None:
        interpolate = y.ndim == 3 and y.shape[1] > 1

    if interpolate and y.ndim == 3:
        _y = y[:, 0, :]
    else:
        _y = y

    if indices is None:
        indices = range(_y.shape[-1])  # e.g. PartiallySolvedSys
    if lines is None:
        lines = interpolate in (None, False)
    markers = len(x) < m_lim

    if yerr is not None:
        for idx in indices:
            clr = plot_kwargs_cb(idx)['c']
            ax.fill_between(x, _y[:, idx] - yerr[:, idx], _y[:, idx] + yerr[:, idx], facecolor=clr, alpha=.3)

    if isinstance(yscale, str) and 'linthreshy' in yscale:
        arg, kw = yscale.split(';')
        thresh = eval('dict(%s)' % kw)['linthreshy']
        ax.axhline(thresh, linewidth=.5, linestyle='--', color='k', alpha=.5)
        ax.axhline(-thresh, linewidth=.5, linestyle='--', color='k', alpha=.5)

    labels = names if latex_names is None else ['$%s$' % ln.strip('$') for ln in latex_names]

    for idx in indices:
        ax.plot(x, _y[:, idx], **plot_kwargs_cb(
            idx, lines=lines, labels=labels))
        if markers:
            ax.plot(x, _y[:, idx], **plot_kwargs_cb(
                idx, lines=False, markers=markers, labels=labels))

    if xlabel is None:
        try:
            plt.xlabel(x.dimensionality.latex)
        except AttributeError:
            pass
    else:
        plt.xlabel(xlabel)

    if ylabel is None:
        try:
            plt.ylabel(_y.dimensionality.latex)
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

        y2 = np.empty((x_plot.size, _y.shape[-1]))
        for idx in range(_y.shape[-1]):
            interp_cb = interp_from_deriv(x, y[..., idx])
            y2[:, idx] = interp_cb(x_plot)

        for idx in indices:
            ax.plot(x_plot, y2[:, idx], **plot_kwargs_cb(
                idx, lines=True, markers=False))
        return x_plot, y2

    if xscale is not None:
        _set_scale(ax.set_xscale, xscale)
    if yscale is not None:
        _set_scale(ax.set_yscale, yscale)

    if legend is True:
        ax.legend()
    elif legend in (None, False):
        pass
    else:
        ax.legend(**legend)
    return ax


def plot_phase_plane(x, y, indices=None, plot=None, names=None, **kwargs):
    """ Plot the phase portrait of two dependent variables

    Parameters
    ----------
    x: array_like
        Values of the independent variable.
    y: array_like
        Values of the dependent variables.
    indices: pair of integers (default: None)
        What dependent variable to plot for (None => (0, 1)).
    plot: callable (default: None)
        Uses ``matplotlib.pyplot.plot`` if ``None``
    names: iterable of strings
        Labels for x and y axis.
    \*\*kwargs:
        Keyword arguemtns passed to ``plot()``.

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

    plot(y[:, indices[0]], y[:, indices[1]], **kwargs)


def right_hand_ylabels(ax, labels):
    ax2 = ax.twinx()
    ylim = ax.get_ylim()
    yspan = ylim[1]-ylim[0]
    ax2.set_ylim(ylim)
    yticks = [ylim[0] + (idx + 0.5)*yspan/len(labels) for idx in range(len(labels))]
    ax2.tick_params(length=0)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(labels)


def info_vlines(ax, xout, info, vline_colors=('maroon', 'purple'),
                vline_keys=('steps', 'rhs_xvals', 'jac_xvals'),
                post_proc=None, alpha=None, fpes=None, every=None):
    """ Plot vertical lines in the background

    Parameters
    ----------
    ax : axes
    xout : array_like
    info : dict
    vline_colors : iterable of str
    vline_keys : iterable of str
        Choose from ``'steps', 'rhs_xvals', 'jac_xvals',
        'fe_underflow', 'fe_overflow', 'fe_invalid', 'fe_divbyzero'``.
    vline_post_proc : callable
    alpha : float

    """

    nvk = len(vline_keys)
    for idx, key in enumerate(vline_keys):
        if key == 'steps':
            vlines = xout
        elif key.startswith('fe_'):
            if fpes is None:
                raise ValueError("Need fpes when vline_keys contain fe_*")
            vlines = xout[info['fpes'] & fpes[key.upper()] > 0]
        else:
            vlines = info[key] if post_proc is None else post_proc(info[key])

        if alpha is None:
            alpha = 0.01 + 1/log(len(vlines)+3)

        if every is None:
            ln_np1 = log(len(vlines)+1)
            every = min(round((ln_np1 - 4)/log(2)), 1)

        ax.vlines(vlines[::every], idx/nvk + 0.002, (idx+1)/nvk - 0.002,
                  colors=vline_colors[idx % len(vline_colors)],
                  alpha=alpha, transform=ax.get_xaxis_transform())
    right_hand_ylabels(ax, [k[3] if k.startswith('fe_') else k[0] for k in vline_keys])
