# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import chain

import numpy as np
import sympy as sp

from .core import OdeSys
from .util import (
    banded_jacobian, stack_1d_on_left, transform_exprs_dep,
    transform_exprs_indep
)


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

    def __init__(self, dep_exprs, indep=None, params=(), jac=True,
                 lband=None, uband=None, lambdify=None, lambdify_unpack=True):
        self.dep, self.exprs = zip(*dep_exprs)
        self.indep = indep
        self.params = params
        self._jac = jac
        if (lband, uband) != (None, None):
            if not lband >= 0 or not uband >= 0:
                raise ValueError("bands needs to be > 0 if provided")
        self.lband = lband
        self.uband = uband
        if lambdify is not None:
            self.lambdify = lambdify
        self.lambdify_unpack = lambdify_unpack
        self.f_cb = self.get_f_ty_callback()
        self.j_cb = self.get_j_ty_callback()
        self.dfdx_cb = self.get_dfdx_callback()

    @staticmethod
    def Symbol(name):
        return sp.Symbol(name, real=True)

    @classmethod
    def symarray(cls, prefix, shape, Symbol=None):
        # see https://github.com/sympy/sympy/pull/9939
        # when released: return sp.symarray(key, n, real=True)
        arr = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            arr[index] = (Symbol or cls.Symbol)('%s_%s' % (
                prefix, '_'.join(map(str, index))))
        return arr

    @staticmethod
    def lambdify(*args, **kwargs):
        if 'modules' not in kwargs:
            kwargs['modules'] = [{'ImmutableMatrix': np.array}, 'numpy']
        return sp.lambdify(*args, **kwargs)

    @classmethod
    def num_transformer_factory(cls, fw, bw, dep, lambdify=None):
        lambdify = lambdify or cls.lambdify
        return lambdify(dep, fw), lambdify(dep, bw)

    @classmethod
    def from_callback(cls, cb, n, nparams=-1, *args, **kwargs):
        x = cls.Symbol('x')
        y = cls.symarray('y', n)
        if nparams == -1:
            p = ()
            exprs = cb(x, y)
        else:
            p = cls.symarray('p', nparams)
            exprs = cb(x, y, p)
        return cls(zip(y, exprs), x, p, *args, **kwargs)

    @property
    def ny(self):
        return len(self.exprs)

    def args(self, x=None, y=None, params=()):
        if x is None:
            x = self.indep
        if y is None:
            y = self.dep
        args = tuple(y)
        if self.indep is not None:
            args = (x,) + args + tuple(params)
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
        cb = self.lambdify(list(chain(self.args(), self.params)), self.exprs)

        def f(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self.args(x, y, params)))
            else:
                return np.asarray(cb(self.args(x, y, params)))
        return f

    def get_j_ty_callback(self):
        cb = self.lambdify(list(chain(self.args(), self.params)), self.get_jac())

        def j(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self.args(x, y, params)))
            else:
                return np.asarray(cb(self.args(x, y, params)))
        return j

    def get_dfdx_callback(self):
        cb = self.lambdify(list(chain(self.args(), self.params)), self.dfdx())

        def dfdx(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self.args(x, y, params)))
            else:
                return np.asarray(cb(self.args(x, y, params)))
        return dfdx

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

    def __init__(self, dep_exprs, indep=None, dep_transf=None,
                 indep_transf=None, params=(), **kwargs):
        dep, exprs = zip(*dep_exprs)
        if dep_transf is not None:
            self.dep_fw, self.dep_bw = zip(*dep_transf)
            exprs = transform_exprs_dep(self.dep_fw, self.dep_bw,
                                        list(zip(dep, exprs)))
        else:
            self.dep_fw, self.dep_bw = None, None

        if indep_transf is not None:
            self.indep_fw, self.indep_bw = indep_transf
            exprs = transform_exprs_indep(self.indep_fw, self.indep_bw,
                                          list(zip(dep, exprs)), indep)
        else:
            self.indep_fw, self.indep_bw = None, None
        super(TransformedSys, self).__init__(zip(dep, exprs), indep, params,
                                             **kwargs)

        self.f_dep, self.b_dep = self.num_transformer_factory(
            self.dep_fw, self.dep_bw, dep)
        self.f_indep, self.b_indep = self.num_transformer_factory(
            self.indep_fw, self.indep_bw, indep)
        self._post_processor = self.back_transform_out
        self._pre_processor = self.forward_transform_xy

    @classmethod
    def from_callback(cls, cb, n, nparams=-1, dep_transf_cbs=None,
                      indep_transf_cbs=None, **kwargs):
        x = cls.Symbol('x')
        y = cls.symarray('y', n)
        if nparams == -1:
            p = ()
            exprs = cb(x, y)
        else:
            p = cls.symarray('p', nparams)
            exprs = cb(x, y, p)
        if dep_transf_cbs is not None:
            try:
                dep_transf = [(dep_transf_cbs[idx][0](yi),
                               dep_transf_cbs[idx][1](yi))
                              for idx, yi in enumerate(y)]
            except TypeError:
                dep_transf = list(zip(list(map(dep_transf_cbs[0], y)),
                                      list(map(dep_transf_cbs[1], y))))
        else:
            dep_transf = None

        if indep_transf_cbs is not None:
            indep_transf = indep_transf_cbs[0](x), indep_transf_cbs[1](x)
        else:
            indep_transf = None

        return cls(list(zip(y, exprs)), x, dep_transf, indep_transf, p, **kwargs)

    def back_transform_out(self, out):
        return stack_1d_on_left(self.b_indep(out[:, 0]),
                                np.array(self.b_dep(*out[:, 1:].T)).T)

    def forward_transform_xy(self, x, y):
        return self.f_indep(x), self.f_dep(*y)
