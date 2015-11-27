# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)


import numpy as np
import inspect


def stack_1d_on_left(x, y):
    """ Stack a 1D array on the left side of a 2D array

    Parameters
    ----------
    x: 1D array
    y: 2D array
        Requirement: ``shape[0] == x.size``
    """
    return np.hstack((np.asarray(x).reshape(len(x), 1),
                      np.asarray(y)))


def banded_jacobian(y, x, ml, mu):
    """ Calculates a banded version of the jacobian

    Compatible with the format requested by
    :func:`scipy.integrate.ode` (for SciPy >= v0.15).

    Parameters
    ----------
    y: array_like of expressions
    x: array_like of symbols
    ml: int
        number of lower bands
    mu: int
        number of upper bands

    Returns
    -------
    2D array of shape ``(1+ml+mu, len(y))``
    """
    ny = len(y)
    nx = len(x)
    packed = np.zeros((mu+ml+1, nx), dtype=object)

    def set(ri, ci, val):
        packed[ri-ci+mu, ci] = val

    for ri in range(ny):
        for ci in range(max(0, ri-ml), min(nx, ri+mu+1)):
            set(ri, ci, y[ri].diff(x[ci]))
    return packed


def check_transforms(fw, bw, symbs):
    """ Verify validity of a pair of forward and backward transformations

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    symbs: iterable of symbols
        the variables that are transformed
    """
    for f, b, y in zip(fw, bw, symbs):
        if f.subs(y, b) - y != 0:
            raise ValueError('Incorrect (did you set real=True?) fw: %s'
                             % str(f))
        if b.subs(y, f) - y != 0:
            raise ValueError('Incorrect (did you set real=True?) bw: %s'
                             % str(b))


def transform_exprs_dep(fw, bw, dep_exprs, check=True):
    """ Transform y[:] in dydx

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    dep_exprs: iterable of (symbol, expression) pairs
        pairs of (dependent variable, derivative expressions),
        i.e. (y, dydx) pairs
    check: bool (default: True)
        whether to verification of the analytic correctness should
        be performed

    Returns
    -------
    List of transformed expressions for dydx
    """
    if len(fw) != len(dep_exprs) or \
       len(fw) != len(bw):
        raise ValueError("Incompatible lengths")
    dep, exprs = zip(*dep_exprs)
    if check:
        check_transforms(fw, bw, dep)
    bw_subs = list(zip(dep, bw))
    return [(e*f.diff(y)).subs(bw_subs) for f, y, e in zip(fw, dep, exprs)]


def transform_exprs_indep(fw, bw, dep_exprs, indep, check=True):
    """ Transform x in dydx

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    dep_exprs: iterable of (symbol, expression) pairs
        pairs of (dependent variable, derivative expressions)
    check: bool (default: True)
        whether to verification of the analytic correctness should
        be performed

    Returns
    -------
    List of transformed expressions for dydx
    """
    if check:
        if fw.subs(indep, bw) - indep != 0:
            fmtstr = 'Incorrect (did you set real=True?) fw: %s'
            raise ValueError(fmtstr % str(fw))
        if bw.subs(indep, fw) - indep != 0:
            fmtstr = 'Incorrect (did you set real=True?) bw: %s'
            raise ValueError(fmtstr % str(bw))
    dep, exprs = zip(*dep_exprs)
    return [(e/fw.diff(indep)).subs(indep, bw) for e in exprs]


def ensure_3args(func):
    """ Conditionally wrap function to ensure 3 input arguments

    Parameters
    ----------
    func: callable
        with two or three positional arguments

    Returns
    -------
    callable which possibly ignores a third positional argument
    """
    nargs = len(inspect.getargspec(func)[0])
    if nargs == 2:
        return lambda x, y, _ignored: func(x, y)
    elif nargs == 3:
        return func
    else:
        raise NotImplementedError
