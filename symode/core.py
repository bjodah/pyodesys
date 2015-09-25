# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp


def _fuse(x, y):
    return np.hstack((np.asarray(x).reshape(len(x), 1),
                      np.asarray(y)))


class OdeSystem(object):
    """
    Parameters
    ----------
    dep_exprs: iterable of (symbol, expression)-pairs
    indep: symbol
        independent variable (default: None => autonomous system)

    Notes
    -----
    Works for a moderate number of unknowns, sympy.lambdify has
    an upper limit on number of arguments.
    """

    # Possible future abstractions:
    # scaling, (variable transformations, then including scaling)

    @classmethod
    def from_callback(cls, cb, n):
        x = sp.Symbol('x')
        y = sp.symarray('y', n)
        exprs = cb(x, y)
        return cls(zip(y, exprs), x)

    def __init__(self, dep_exprs, indep=None):
        self.dep, self.exprs = zip(*dep_exprs)
        self.indep = indep

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

    def jac(self):
        f = sp.Matrix(1, self.ny, lambda _, q: self.exprs[q])
        return f.jacobian(self.dep)

    def get_f_lambda(self):
        return sp.lambdify(self.args(), self.exprs)

    def get_jac_lambda(self):
        return sp.lambdify(self.args(), self.jac(),
                           modules={'ImmutableMatrix': np.array})

    def dfdx(self):
        if self.indep is None:
            return [0]*self.ny
        else:
            return [expr.diff(self.indep) for expr in self.exprs]

    def get_dfdx_lambda(self):
        return sp.lambdify(self.args(), self.dfdx())

    def get_fj_ty_callbacks(self):
        f_lambda = self.get_f_lambda()
        j_lambda = self.get_jac_lambda()
        f = lambda x, y: np.asarray(f_lambda(*self.args(x, y)))
        j = lambda x, y: np.asarray(j_lambda(*self.args(x, y)))
        return f, j

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
        else:
            raise NotImplementedError("Unkown solver %s" % solver)

    def integrate_scipy(self, xout, y0, name='lsoda',
                        atol=1e-8, rtol=1e-8,
                        **kwargs):
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
        try:
            len(xout)
        except TypeError:
            xout = (0, xout)
        from scipy.integrate import ode
        f, j = self.get_fj_ty_callbacks()
        r = ode(f, jac=j)
        r.set_integrator(name, atol=atol, rtol=rtol, **kwargs)
        r.set_initial_value(y0, xout[0])
        if len(xout) == 2:
            yout = [y0]
            tstep = [xout[0]]
            while r.t < xout[1]:
                r.integrate(xout[1], step=True)
                if not r.successful:
                    raise Excpetion("failed")
                tstep.append(r.t)
                yout.append(r.y)
            out = _fuse(tstep, yout)
        else:
            out = np.empty((len(xout), 1 + self.ny))
            for idx, t in enumerate(xout):
                print(t)
                r.integrate(t)
                if not r.successful:
                    raise Excpetion("failed")
                out[idx, 0] = t
                out[idx, 1:] = r.y
        return out

    def _integrate(self, adaptive, predefined, xout, y0,
                   atol=1e-8, rtol=1e-8, first_step=1e-16, **kwargs):
        try:
            len(xout)
        except TypeError:
            xout = (0, xout)
        new_kwargs = dict(dx0=first_step, atol=atol,
                          rtol=rtol)
        new_kwargs.update(kwargs)
        f, j = self.get_fj_ty_callbacks()
        dfdx = self.get_dfdx_callback()

        def _f(x, y, fout):
            fout[:] = f(x, y)

        def _j(x, y, jout, dfdx_out):
            jout[:, :] = j(x, y)
            dfdx_out[:] = dfdx(x, y)

        if len(xout) == 2:
            xsteps, yout = adaptive(_f, _j, y0, xout[0], xout[1], **new_kwargs)
            return _fuse(xsteps, yout)
        else:
            yout = predefined(_f, _j, y0, xout, **new_kwargs)
            return _fuse(xout, yout)

    def integrate_gsl(self, *args, **kwargs):
        import pygslodeiv2
        return self._integrate(pygslodeiv2.integrate_adaptive,
                               pygslodeiv2.integrate_predefined,
                               *args, **kwargs)

    def integrate_odeint(self, *args, **kwargs):
        import pyodeint
        return self._integrate(pyodeint.integrate_adaptive,
                               pyodeint.integrate_predefined,
                               *args, **kwargs)
