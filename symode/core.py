# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

from .util import banded_jacobian


def stack_1d_on_left(x, y):
    return np.hstack((np.asarray(x).reshape(len(x), 1),
                      np.asarray(y)))


class OdeSystem(object):
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

    # Possible future abstractions:
    # scaling, (variable transformations, then including scaling)

    def __init__(self, dep_exprs, indep=None, jac=True,
                 lband=None, uband=None):
        self.dep, self.exprs = zip(*dep_exprs)
        self.indep = indep
        self._jac = jac
        if (lband, uband) != (None, None):
            if not lband >= 0 or not uband >= 0:
                raise ValueError("bands needs to be > 0 if provided")
        self.lband = lband
        self.uband = uband

    @classmethod
    def from_callback(cls, cb, n, *args, **kwargs):
        x = sp.Symbol('x')
        y = sp.symarray('y', n)
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
                return banded_jacobian(self.exprs, self.dep,
                                       self.lband, self.uband)
        elif self._jac is False:
            return False
        else:
            return self._jac

    def get_f_lambda(self):
        return sp.lambdify(self.args(), self.exprs)

    def get_jac_lambda(self):
        return sp.lambdify(self.args(), self.get_jac(),
                           modules={'ImmutableMatrix': np.array})

    def dfdx(self):
        if self.indep is None:
            return [0]*self.ny
        else:
            return [expr.diff(self.indep) for expr in self.exprs]

    def get_dfdx_lambda(self):
        return sp.lambdify(self.args(), self.dfdx())

    def get_f_ty_callback(self):
        f_lambda = self.get_f_lambda()
        return lambda x, y: np.asarray(f_lambda(*self.args(x, y)))

    def get_j_ty_callback(self):
        j_lambda = self.get_jac_lambda()
        return lambda x, y: np.asarray(j_lambda(*self.args(x, y)))

    def get_dfdx_callback(self):
        dfdx_lambda = self.get_dfdx_lambda()
        return lambda x, y: np.asarray(dfdx_lambda(*self.args(x, y)))

    def integrate(self, solver, *args, **kwargs):
        if solver == 'scipy':
            return self.integrate_scipy(*args, **kwargs)
        elif solver == 'gsl':
            return self.integrate_gsl(*args, **kwargs)
        elif solver == 'odeint':
            return self.integrate_odeint(*args, **kwargs)
        elif solver == 'cvode':
            return self.integrate_cvode(*args, **kwargs)
        else:
            raise NotImplementedError("Unkown solver %s" % solver)

    def integrate_mpmath(self, xout, y0):
        try:
            len(xout)
        except TypeError:
            xout = (0, xout)

        from mpmath import odefun
        cb = odefun(lambda x, y: [e.subs(
            [(self.indep, x)]+list(zip(self.dep, y))
        ) for e in self.exprs], xout[0], y0)
        yout = []
        for x in xout:
            yout.append(cb(x))
        return stack_1d_on_left(xout, yout)

    def integrate_scipy(self, xout, y0, name='lsoda', atol=1e-8,
                        rtol=1e-8, with_jacobian=None, **kwargs):
        """
        Parameters
        ----------
        xout: array_like or pair (start and final time) or float
            if array_like:
            length-2 iterable
                values of independent variable to integrate to
            if a pair (length two):
                initial and final time
            if a float:
                make it a pair: (0, xout)
        y0: array_like

        Returns
        -------
        2-dimensional array (first column indep., rest dep.)
        """
        if with_jacobian is None:
            if name == 'lsoda':  # lsoda might call jacobian
                with_jacobian = True
            elif name in ('dop853', 'dopri5'):
                with_jacobian = False  # explicit steppers
            elif name == 'vode':
                with_jacobian = kwargs.get('method', 'adams') == 'bdf'
        try:
            len(xout)
        except TypeError:
            xout = (0, xout)
        from scipy.integrate import ode
        f = self.get_f_ty_callback()
        if with_jacobian:
            j = self.get_j_ty_callback()
        else:
            j = None
        r = ode(f, jac=j)
        if 'lband' in kwargs or 'uband' in kwargs:
            raise ValueError("lband and uband set locally (set at"
                             " initialization of OdeSystem instead)")
        if self.lband is not None:
            kwargs['lband'], kwargs['uband'] = self.lband, self.uband
        r.set_integrator(name, atol=atol, rtol=rtol, **kwargs)
        r.set_initial_value(y0, xout[0])
        if len(xout) == 2:
            yout = [y0]
            tstep = [xout[0]]
            while r.t < xout[1]:
                r.integrate(xout[1], step=True)
                if not r.successful:
                    raise RuntimeError("failed")
                tstep.append(r.t)
                yout.append(r.y)
            out = stack_1d_on_left(tstep, yout)
        else:
            out = np.empty((len(xout), 1 + self.ny))
            for idx, t in enumerate(xout):
                print(t)
                r.integrate(t)
                if not r.successful:
                    raise RuntimeError("failed")
                out[idx, 0] = t
                out[idx, 1:] = r.y
        return out

    def _integrate(self, adaptive, predefined, xout, y0, with_jacobian,
                   atol=1e-8, rtol=1e-8, first_step=1e-16, **kwargs):

        try:
            nx = len(xout)
        except TypeError:
            xout = (0, xout)
            nx = 2
        new_kwargs = dict(dx0=first_step, atol=atol,
                          rtol=rtol)
        new_kwargs.update(kwargs)
        f = self.get_f_ty_callback()

        def _f(x, y, fout):
            fout[:] = f(x, y)

        if with_jacobian:
            j = self.get_j_ty_callback()
            dfdx = self.get_dfdx_callback()

            def _j(x, y, jout, dfdx_out=None, fy=None):
                jout[:, :] = j(x, y)
                if dfdx_out is not None:
                    dfdx_out[:] = dfdx(x, y)
        else:
            _j = None

        if nx == 2:
            xsteps, yout = adaptive(_f, _j, y0, *xout, **new_kwargs)
            return stack_1d_on_left(xsteps, yout)
        else:
            yout = predefined(_f, _j, y0, xout, **new_kwargs)
            return stack_1d_on_left(xout, yout)

    def integrate_gsl(self, *args, **kwargs):
        import pygslodeiv2
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'bsimp') in pygslodeiv2.requires_jac
        return self._integrate(pygslodeiv2.integrate_adaptive,
                               pygslodeiv2.integrate_predefined,
                               *args, **kwargs)

    def integrate_odeint(self, *args, **kwargs):
        import pyodeint
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'rosenbrock4') in pyodeint.requires_jac
        return self._integrate(pyodeint.integrate_adaptive,
                               pyodeint.integrate_predefined,
                               *args, **kwargs)

    def integrate_cvode(self, *args, **kwargs):
        import pycvodes
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'bdf') in pycvodes.requires_jac
        if 'lband' in kwargs or 'uband' in kwargs:
            raise ValueError("lband and uband set locally (set at"
                             " initialization of OdeSystem instead)")
        if self.lband is not None:
            kwargs['lband'], kwargs['uband'] = self.lband, self.uband
        return self._integrate(pycvodes.integrate_adaptive,
                               pycvodes.integrate_predefined,
                               *args, **kwargs)
