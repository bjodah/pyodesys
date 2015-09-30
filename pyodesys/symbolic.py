# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import sympy as sp

from .core import OdeSys
from .util import banded_jacobian, stack_1d_on_left


def _lambdify(*args, **kwargs):
    if 'modules' not in kwargs:
        kwargs['modules'] = [{'ImmutableMatrix': np.array}, 'numpy']
    return sp.lambdify(*args, **kwargs)


def _Symbol(name):
    return sp.Symbol(name, real=True)


def _symarray(key, n):
    # see https://github.com/sympy/sympy/pull/9939
    # when merged: return sp.symarray(key, n, real=True)
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = Symbol('%s_%s' % (prefix, '_'.join(map(str, index))),
                            real=True)
    return arr


class SymbolicSys(OdeSys):
    """
    Parameters
    ----------
    dep_exprs: iterable of (symbol, expression)-pairs
    indep: symbol
        independent variable (default: None => autonomous system)
    jac: ImmutableMatrix or bool (default: True)
        If True:
            calculate jacobian from exprs
        If False:
            do not compute jacobian (use explicit steppers)
        If ImmutableMatrix:
            user provided expressions for the jacobian

    Notes
    -----
    Works for a moderate number of unknowns, sympy.lambdify has
    an upper limit on number of arguments.
    """

    def __init__(self, dep_exprs, indep=None, jac=True,
                 lband=None, uband=None, lambdify=None):
        self.dep, self.exprs = zip(*dep_exprs)
        self.indep = indep
        self._jac = jac
        if (lband, uband) != (None, None):
            if not lband >= 0 or not uband >= 0:
                raise ValueError("bands needs to be > 0 if provided")
        self.lband = lband
        self.uband = uband
        self.lambdify = lambdify or _lambdify

    @classmethod
    def from_callback(cls, cb, n, *args, **kwargs):
        x = _Symbol('x')
        y = _symarray('y', n)
        exprs = cb(x, y)
        return cls(zip(y, exprs), x, *args, **kwargs)

    @property
    def ny(self):
        return len(self.exprs)

    def args(self, x=None, y=None):
        if x is None:
            x = self.indep
        if y is None:
            y = self.dep
        args = tuple(y)
        if self.indep is not None:
            args = (x,) + args
        return args

    def get_jac(self):
        if self._jac is True:
            if self.lband is None:
                f = sp.Matrix(1, self.ny, lambda _, q: self.exprs[q])
                return f.jacobian(self.dep)
            else:
                # Banded
                return sp.ImmutableMatrix(banded_jacobian(
                    self.exprs, self.dep, self.lband, self.uband))
        elif self._jac is False:
            return False
        else:
            return self._jac

    def get_f_lambda(self):
        return self.lambdify(self.args(), self.exprs)

    def get_jac_lambda(self):
        return self.lambdify(self.args(), self.get_jac())

    def dfdx(self):
        if self.indep is None:
            return [0]*self.ny
        else:
            return [expr.diff(self.indep) for expr in self.exprs]

    def get_dfdx_lambda(self):
        return self.lambdify(self.args(), self.dfdx())

    def get_f_ty_callback(self):
        f_lambda = self.get_f_lambda()
        return lambda x, y: np.asarray(f_lambda(*self.args(x, y)))

    def get_j_ty_callback(self):
        j_lambda = self.get_jac_lambda()
        return lambda x, y: np.asarray(j_lambda(*self.args(x, y)))

    def get_dfdx_callback(self):
        dfdx_lambda = self.get_dfdx_lambda()
        return lambda x, y: np.asarray(dfdx_lambda(*self.args(x, y)))

    # Not working yet:
    def integrate_mpmath(self, xout, y0):
        """ Not working at the moment, need to fix
        (low priority - taylor series is a poor method)"""
        try:
            len(xout)
        except TypeError:
            xout = (0, xout)

        from mpmath import odefun
        cb = odefun(lambda x, y: [e.subs(
            ([(self.indep, x)] if self.indep is not None else []) +
            list(zip(self.dep, y))
        ) for e in self.exprs], xout[0], y0)
        yout = []
        for x in xout:
            yout.append(cb(x))
        return stack_1d_on_left(xout, yout)


def transform_exprs_dep(fw, bw, dep_exprs, check=True):
    if len(fw) != len(dep_exprs) or \
       len(fw) != len(bw):
        raise ValueError("Incompatible lengths")
    dep, exprs = zip(*dep_exprs)
    if check:
        for f, b, y in zip(fw, bw, dep):
            if f.subs(x, b) - x != 0:
                raise ValueError('Incorrect (did you set real=True?) fw: %s'
                                 % str(f))
            if b.subs(x, f) - x != 0:
                raise ValueError('Incorrect (did you set real=True?) bw: %s'
                                 % str(b))
    bw_subs = zip(dep, bw)
    return [(e*f.diff(y)).subs(bw_subs) for f, y, e in zip(fw, dep, exprs)]


def transform_exprs_indep(fw, bw, dep_exprs, indep, check=True):
    if check:
        if fw.subs(indep, bw) - indep != 0:
                raise ValueError('Incorrect (did you set real=True?) fw: %s'
                                 % str(fw))
        if bw.subs(x, fw) - indep != 0:
            raise ValueError('Incorrect (did you set real=True?) bw: %s'
                             % str(bw))
    dep, exprs = zip(*dep_exprs)
    return [(e/fw.diff(indep)).subs(indep, bw) for e in exprs]


def num_dep_tranformer_factory(fw, bw, dep, lambdify=None):
    lambdify = lambdify or _lambdify
    return lambdify(dep, fw), lambdify(dep, bw)
