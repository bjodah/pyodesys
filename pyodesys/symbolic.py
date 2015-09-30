# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import sympy as sp

from .core import OdeSys
from .util import (
    banded_jacobian, stack_1d_on_left, transform_exprs_dep,
    transform_exprs_indep
)


def _lambdify(*args, **kwargs):
    if 'modules' not in kwargs:
        kwargs['modules'] = [{'ImmutableMatrix': np.array}, 'numpy']
    return sp.lambdify(*args, **kwargs)


def _num_transformer_factory(fw, bw, dep, lambdify=None):
    lambdify = lambdify or _lambdify
    return lambdify(dep, fw), lambdify(dep, bw)


def _Symbol(name):
    return sp.Symbol(name, real=True)


def _symarray(prefix, shape, Symbol=None):
    # see https://github.com/sympy/sympy/pull/9939
    # when merged: return sp.symarray(key, n, real=True)
    arr = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        arr[index] = (Symbol or _Symbol)('%s_%s' % (
            prefix, '_'.join(map(str, index))))
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

    def dfdx(self):
        if self.indep is None:
            return [0]*self.ny
        else:
            return [expr.diff(self.indep) for expr in self.exprs]

    def get_f_ty_callback(self):
        f_lambda = self.lambdify(self.args(), self.exprs)
        return lambda x, y: np.asarray(f_lambda(*self.args(x, y)))

    def get_j_ty_callback(self):
        j_lambda = self.lambdify(self.args(), self.get_jac())
        return lambda x, y: np.asarray(j_lambda(*self.args(x, y)))

    def get_dfdx_callback(self):
        dfdx_lambda = self.lambdify(self.args(), self.dfdx())
        return lambda x, y: np.asarray(dfdx_lambda(*self.args(x, y)))

    # Not working yet:
    def integrate_mpmath(self, xout, y0):
        """ Not working at the moment, need to fix
        (low priority - taylor series is a poor method)"""
        xout, y0 = self.pre_process(xout, y0)
        from mpmath import odefun
        cb = odefun(lambda x, y: [e.subs(
            ([(self.indep, x)] if self.indep is not None else []) +
            list(zip(self.dep, y))
        ) for e in self.exprs], xout[0], y0)
        yout = []
        for x in xout:
            yout.append(cb(x))
        return self.post_process(stack_1d_on_left(xout, yout))


class TransformedSys(SymbolicSys):

    def __init__(self, dep_exprs, indep=None,
                 dep_transf=None, indep_transf=None, **kwargs):
        dep, exprs = zip(*dep_exprs)
        if dep_transf is not None:
            self.dep_fw, self.dep_bw = zip(*dep_transf)
            exprs = transform_exprs_dep(self.dep_fw, self.dep_bw,
                                        zip(dep, exprs))
        else:
            self.dep_fw, self.dep_bw = None, None

        if indep_transf is not None:
            self.indep_fw, self.indep_bw = indep_transf
            exprs = transform_exprs_indep(self.indep_fw, self.indep_bw,
                                          zip(dep, exprs), indep)
        else:
            self.indep_fw, self.indep_bw = None, None
        super(TransformedSys, self).__init__(zip(dep, exprs), indep, **kwargs)

        self.f_dep, self.b_dep = _num_transformer_factory(
            self.dep_fw, self.dep_bw, dep)
        self.f_indep, self.b_indep = _num_transformer_factory(
            self.indep_fw, self.indep_bw, indep)
        self._post_processor = self.back_transform_out
        self._pre_processor = self.forward_transform_xy

    @classmethod
    def from_callback(cls, cb, n, dep_transf_cbs=None, indep_transf_cbs=None,
                      **kwargs):
        x = _Symbol('x')
        y = _symarray('y', n)
        exprs = cb(x, y)
        if dep_transf_cbs is not None:
            try:
                dep_transf = [(dep_transf_cbs[idx][0](yi),
                               dep_transf_cbs[idx][1](yi))
                              for idx, yi in enumerate(y)]
            except TypeError:
                dep_transf = zip(map(dep_transf_cbs[0], y),
                                 map(dep_transf_cbs[1], y))
        else:
            dep_transf = None

        if indep_transf_cbs is not None:
            indep_transf = indep_transf_cbs[0](x), indep_transf_cbs[1](x)
        else:
            indep_transf = None

        return cls(zip(y, exprs), x, dep_transf, indep_transf, **kwargs)

    def back_transform_out(self, out):
        return stack_1d_on_left(self.b_indep(out[:, 0]),
                                np.array(self.b_dep(*out[:, 1:].T)).T)

    def forward_transform_xy(self, x, y):
        return self.f_indep(x), self.f_dep(*y)
