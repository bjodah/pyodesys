# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)


import numpy as np
import inspect


def stack_1d_on_left(x, y):
    return np.hstack((np.asarray(x).reshape(len(x), 1),
                      np.asarray(y)))


def banded_jacobian(y, x, ml, mu):
    """
    Calculates a banded version of the jacobian

    Compatible with the format requested by
    scipy.integrate.ode
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
    for f, b, y in zip(fw, bw, symbs):
        if f.subs(y, b) - y != 0:
            raise ValueError('Incorrect (did you set real=True?) fw: %s'
                             % str(f))
        if b.subs(y, f) - y != 0:
            raise ValueError('Incorrect (did you set real=True?) bw: %s'
                             % str(b))


def transform_exprs_dep(fw, bw, dep_exprs, check=True):
    if len(fw) != len(dep_exprs) or \
       len(fw) != len(bw):
        raise ValueError("Incompatible lengths")
    dep, exprs = zip(*dep_exprs)
    if check:
        check_transforms(fw, bw, dep)
    bw_subs = list(zip(dep, bw))
    return [(e*f.diff(y)).subs(bw_subs) for f, y, e in zip(fw, dep, exprs)]


def transform_exprs_indep(fw, bw, dep_exprs, indep, check=True):
    if check:
        if fw.subs(indep, bw) - indep != 0:
                raise ValueError('Incorrect (did you set real=True?) fw: %s'
                                 % str(fw))
        if bw.subs(indep, fw) - indep != 0:
            raise ValueError('Incorrect (did you set real=True?) bw: %s'
                             % str(bw))
    dep, exprs = zip(*dep_exprs)
    return [(e/fw.diff(indep)).subs(indep, bw) for e in exprs]


def ensure_3args(func):
    nargs = len(inspect.getargspec(func)[0])
    if nargs == 2:
        return lambda x, y, params: func(x, y)
    elif nargs == 3:
        return func
    else:
        raise NotImplementedError
