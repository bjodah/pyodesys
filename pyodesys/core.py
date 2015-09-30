# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from .util import stack_1d_on_left


class OdeSys(object):
    """
    Object representing odesystem. Provides unified interface to:

    - scipy.integarte.ode
    - pygslodeiv2
    - pyodeint
    - pycvodes

    Adaptive (dense output) or predefined set of data points to report
    at is chosen automatically.

    Parameters
    ----------
    f: callback
        first derivatives of dependent variables (y) with respect to
        dependent variable (x). Signature f(x, y)
    jac: callback
        Jacobian matrix (dfdy). optional for explicit methods,
        required for implicit methods
    lband: int or None (default)
        If jacobian is banded: number of sub-diagonals
    uband: int or None (default)
        If jacobian is banded: number of super-diagonals

    Notes
    -----
    banded jacobians are supported by "scipy" and "cvode" solvers
    """

    # Possible future abstractions:
    # scaling, (variable transformations, then including scaling)

    _pre_processor = None
    _post_processor = None

    def __init__(self, f, jac=None, dfdx=None, lband=None, uband=None,
                 **kwargs):
        self.get_f_ty_callback = lambda: f
        self.get_j_ty_callback = lambda: jac
        self.get_dfdx_callback = lambda: dfdx
        self.lband = lband
        self.uband = uband
        self._y0 = None
        self.kwargs = kwargs  # default kwargs for integration

    def pre_process(self, xout, y0):
        # Should be used by all methods matching "integrate_*"
        try:
            nx = len(xout)
            if nx == 1:
                xout = (0, xout[0])
        except TypeError:
            xout = (0, xout)

        if self._pre_processor is None:
            return xout, y0
        else:
            return self._pre_processor(xout, y0)

    def post_process(self, out):
        # Should be used by all methods matching "integrate_*"
        if self._post_processor is None:
            return out
        else:
            return self._post_processor(out)

    def integrate(self, solver, *args, **kwargs):
        """
        Integrate with ``solver``. Convenience method.
        """
        return getattr(self, 'integrate_'+solver)(*args, **kwargs)

    def integrate_scipy(self, xout, y0, atol=1e-8,
                        rtol=1e-8, with_jacobian=None,
                        name='lsoda', **kwargs):
        """
        Use scipy.integrate.ode

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
            Initial values at xout[0] for the dependent variables.
        atol: float
            Absolute tolerance
        rtol: float
            Relative tolerance
        with_jacboian: bool or None (default)
            Whether to use the jacobian, when None the choice is
            done automatically (only used when required). This matters
            when jacobian derived (at high computational cost).
        name: str (default: 'lsoda')
            what solver wrapped in scipy.integrate.ode to use.
        \*\*kwargs:
            keyword arguments passed onto set_integrator(...)

        Returns
        -------
        2-dimensional array (first column indep., rest dep.)
        """
        xout, y0 = self.pre_process(xout, y0)
        nx = len(xout)
        if with_jacobian is None:
            if name == 'lsoda':  # lsoda might call jacobian
                with_jacobian = True
            elif name in ('dop853', 'dopri5'):
                with_jacobian = False  # explicit steppers
            elif name == 'vode':
                with_jacobian = kwargs.get('method', 'adams') == 'bdf'
        from scipy.integrate import ode
        f = self.get_f_ty_callback()
        if with_jacobian:
            j = self.get_j_ty_callback()
        else:
            j = None
        r = ode(f, jac=j)
        if 'lband' in kwargs or 'uband' in kwargs:
            raise ValueError("lband and uband set locally (set at"
                             " initialization instead)")
        if self.lband is not None:
            kwargs['lband'], kwargs['uband'] = self.lband, self.uband
        r.set_integrator(name, atol=atol, rtol=rtol, **kwargs)
        r.set_initial_value(y0, xout[0])
        if nx == 2:
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
            out = np.empty((nx, 1 + self.ny))
            out[0, 0] = xout[0]
            out[0, 1:] = y0
            for idx in range(1, nx):
                r.integrate(xout[idx])
                if not r.successful:
                    raise RuntimeError("failed")
                out[idx, 0] = xout[idx]
                out[idx, 1:] = r.y
        return self.post_process(out)

    def _integrate(self, adaptive, predefined, xout, y0, with_jacobian,
                   atol=1e-8, rtol=1e-8, first_step=1e-16, **kwargs):
        xout, y0 = self.pre_process(xout, y0)
        nx = len(xout)
        new_kwargs = dict(dx0=first_step, atol=atol,
                          rtol=rtol, check_indexing=False)
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
            out = stack_1d_on_left(xsteps, yout)
        else:
            yout = predefined(_f, _j, y0, xout, **new_kwargs)
            out = stack_1d_on_left(xout, yout)
        return self.post_process(out)

    def integrate_gsl(self, *args, **kwargs):
        """ Use GNU Scientific Library to integrate ODE system. """
        import pygslodeiv2
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'bsimp') in pygslodeiv2.requires_jac
        return self._integrate(pygslodeiv2.integrate_adaptive,
                               pygslodeiv2.integrate_predefined,
                               *args, **kwargs)

    def integrate_odeint(self, *args, **kwargs):
        """ Use Boost.Numeric.Odeint to integrate the ODE system. """
        import pyodeint
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'rosenbrock4') in pyodeint.requires_jac
        return self._integrate(pyodeint.integrate_adaptive,
                               pyodeint.integrate_predefined,
                               *args, **kwargs)

    def integrate_cvode(self, *args, **kwargs):
        """ Use CVode (from CVodes in Sundials) to
        integrate the ODE system. """
        import pycvodes
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'bdf') in pycvodes.requires_jac
        if 'lband' in kwargs or 'uband' in kwargs:
            raise ValueError("lband and uband set locally (set at"
                             " initialization instead)")
        if self.lband is not None:
            kwargs['lband'], kwargs['uband'] = self.lband, self.uband
        return self._integrate(pycvodes.integrate_adaptive,
                               pycvodes.integrate_predefined,
                               *args, **kwargs)
