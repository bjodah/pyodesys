# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import warnings
from math import exp

import numpy as np


def fit_factory(discard=1):
    def fit(x, y):
        p = np.polyfit(x, y, 1)
        v = np.polyval(p, x)
        e = np.abs(y - v)
        drop_idxs = np.argsort(e)[-discard]
        return np.polyfit(np.delete(x, drop_idxs),
                          np.delete(y, drop_idxs), 1)
    return fit


def integrate_tolerance_series(odesys, atols, rtols, x, y0, params=(),
                               fit=lambda x, y: np.polyfit(x, y, 1), val=np.polyval, **kwargs):
    """
    Parameters
    ----------
    odesys : :class:`ODESys`
    atols : array_like
         Positive, monotonically increasing 1D array.
    rtols : array_like
         Positive, monotonically increasing 1D array.
    x : array_like
        Passed on to ``odesys.integrate`` for first set of tolerances.
        (subsequent calls will use xout from first integration).
    y0 : array_like
        Passed on to ``odesys.integrate``.
    params : array_like
        Passed on to ``odesys.integrate``.
    fit : callable
    val : callable
    \\*\\*kwargs:
        Passed on to ``odesys.integrate``.

    Returns
    -------
    result0 : Result
    results : list of Result instances
    extra : dict
        errest : 2D array of error estimates for result0.yout

    """
    if atols is None:
        atols = rtols
    if rtols is None:
        rtols = atols
    atols, rtols = map(np.asarray, (atols, rtols))
    if atols.ndim != 1:
        raise NotImplementedError("Assuming 1-dimensional array")
    if atols.shape != rtols.shape:
        raise ValueError("atols & rtols need to be of same length")
    if 'atol' in kwargs or 'rtol' in kwargs:
        raise ValueError("Neither atol nor rtol are allowed in kwargs")
    if not np.all(atols > 0) or not np.all(rtols > 0):
        raise ValueError("atols & rtols need to > 0")
    if not np.all(np.diff(atols) > 0) or not np.all(np.diff(rtols) > 0):
        raise ValueError("atols & rtols need to obey strict positive monotonicity")
    if atols.size < 4:
        raise ValueError("Pointless doing linear interpolation on less than 3 points")
    if atols.size < 6:
        warnings.warn("Statistics will be (very) shaky when doing linear "
                      "interpolation on less than 5 points.")
    ntols = atols.size
    result0 = odesys.integrate(x, y0, params, atol=atols[0], rtol=rtols[0], **kwargs)
    results = [odesys.integrate(result0.xout, y0, params, atol=atols[i], rtol=rtols[i], **kwargs)
               for i in range(1, ntols)]
    errest = []
    for ix, vx in enumerate(result0.xout):
        diffs = np.array([result0.yout[ix, :] - r.yout[ix, :] for r in results])
        tols = np.array([atol + rtol*np.abs(r.yout[ix, :]) for r, atol, rtol in
                         zip([result0] + results, atols, rtols)])
        ln_tols = np.log(tols).astype(np.float64)
        ln_absd = np.log(np.abs(diffs)).astype(np.float64)
        yerrs = []
        for iy in range(result0.yout.shape[-1]):
            if np.all(diffs[:, iy] == 0):
                yerrs.append(0)
            else:
                p = fit(ln_tols[1:, iy], ln_absd[:, iy])
                yerrs.append(exp(val(p, ln_tols[0, iy])))
        errest.append(yerrs)
    return result0, results, {'errest': np.array(errest)}
