# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from .util import ensure_3args
from .plotting import plot_result, plot_phase_plane


class OdeSys(object):
    """
    Object representing odesystem. Provides unified interface to:

    - scipy.integarte.ode
    - pygslodeiv2
    - pyodeint
    - pycvodes

    The numerical integration can be performed either in an :meth:`adaptive`
    or :meth:`predefined` mode. Where locations to report the solution is
    chosen by the stepper or the user respectively. For convenience in user
    code one may use :meth:`integrate` which automatically chooses between
    the two based on the length of ``xout`` provided by the user.

    Parameters
    ----------
    f: callback
        first derivatives of dependent variables (y) with respect to
        dependent variable (x). Signature f(x, y)
    jac: callback
        Jacobian matrix (dfdy). optional for explicit methods,
        required for implicit methods
    dfdx: callback
        pass
    band: tuple of 2 ints or None (default: None)
        If jacobian is banded: number of sub- and super-diagonals
    names: iterable of str (default: None)
        names of variables, used for plotting

    Attributes
    ----------
    f_cb: callback for evaluating the vector of derivatives
    j_cb: callback for evaluating the Jacobian matrix of f
    names: iterable of str objects
    internal_xout: before post-processing
    internal_yout: before post-processing

    Notes
    -----
    banded jacobians are supported by "scipy" and "cvode" solvers
    """

    # Possible future abstractions:
    # scaling, (variable transformations, then including scaling)

    _pre_processor = None
    _post_processor = None

    def __init__(self, f, jac=None, dfdx=None, roots=None, nroots=None,
                 band=None, names=None):
        self.f_cb = ensure_3args(f)
        self.j_cb = ensure_3args(jac) if jac is not None else None
        self.dfdx_cb = dfdx
        self.roots_cb = roots
        self.nroots = nroots
        if band is not None:
            if not band[0] >= 0 or not band[1] >= 0:
                raise ValueError("bands needs to be > 0 if provided")
        self.band = band
        self.names = names

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

    def post_process(self, xout, yout):
        # Should be used by all methods matching "integrate_*"
        self.internal_xout = np.asarray(xout, dtype=np.float64).copy()
        self.internal_yout = np.asarray(yout, dtype=np.float64).copy()
        if self._post_processor is None:
            return xout, yout
        else:
            return self._post_processor(xout, yout)

    def adaptive(self, solver, y0, x0, xend, params=(), **kwargs):
        """
        Parameters
        ----------
        solver: str
            see :meth:`integrate`
        y0: array_like
            see :meth:`integrate`
        x0: float
            initial value of the independent variable
        xend: float
            final value of the independent variable
        params: array_like
            see :meth:`integrate`
        \*\*kwargs:
            see :py:meth:`integrate`

        Returns
        -------
        Same as :meth:`integrate`
        """
        return self.integrate(solver, (x0, xend), y0, params=params, **kwargs)

    def predefined(self, solver, y0, xout, params=(), **kwargs):
        """
        Parameters
        ----------
        solver: str
            see :meth:`integrate`
        y0: array_like
            see :meth:`integrate`
        xout: array_like
        params: array_like
            see :meth:`integrate`
        \*\*kwargs:
            see :meth:`integrate`

        Returns
        -------
        Length 2 tuple: (yout, info)
            see :meth:`integrate`
        """
        xout, yout, info = self.integrate(solver, xout, y0, params=params,
                                          force_predefined=True, **kwargs)
        return yout, info

    def integrate(self, solver, xout, y0, params=(),
                  **kwargs):
        """
        Integrate using ``solver``.

        Parameters
        ----------
        solve: str
            Name of solver, one of: 'scipy', 'gsl', 'odeint', 'cvode'.
            See respective method for more information.
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
        params: array_like (default: tuple())
        atol: float
            Absolute tolerance
        rtol: float
            Relative tolerance
        with_jacobian: bool or None (default)
            Whether to use the jacobian. When ``None`` the choice is
            done automatically (only used when required). This matters
            when jacobian is derived at runtime (high computational cost).
        force_predefined: bool (default: False)
            override behaviour of len(xout) == 2 => adaptive
        \*\*kwargs:
            Additional keyword arguments passed to ``integrate_$(solver)``.

        Returns
        -------
        Length 3 tuple: (xout, yout, info)
        xout: array of values of the independent variable
        yout: array of the dependent variable(s) for the different values of x
        info: dict ('nrhs' and 'njac' guaranteed to be there)
        """
        return getattr(self, 'integrate_'+solver)(xout, y0, params, **kwargs)

    def integrate_scipy(self, xout, y0, params=None, atol=1e-8, rtol=1e-8,
                        first_step=None, with_jacobian=None,
                        force_predefined=False, name='lsoda', **kwargs):
        """
        Use scipy.integrate.ode

        Parameters
        ----------
        \*args:
            see :method:`integrate`
        name: str (default: 'lsoda')
            what solver wrapped in scipy.integrate.ode to use.
        \*\*kwargs:
            keyword arguments passed onto set_integrator(...)

        Returns
        -------
        Pair (length 2-tuple):
            2-dimensional array (first column indep., rest dep.), infodict
        """
        ny = len(y0)
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

        def rhs(t, y, p=()):
            rhs.ncall += 1
            return self.f_cb(t, y, p)
        rhs.ncall = 0

        def jac(t, y, p=()):
            jac.ncall += 1
            return self.j_cb(t, y, p)
        jac.ncall = 0

        r = ode(rhs, jac=jac if with_jacobian else None)
        if 'lband' in kwargs or 'uband' in kwargs or 'band' in kwargs:
            raise ValueError("lband and uband set locally (set `band` at"
                             " initialization instead)")
        if self.band is not None:
            kwargs['lband'], kwargs['uband'] = self.band
        r.set_integrator(name, atol=atol, rtol=rtol, **kwargs)
        if params is not None:
            r.set_f_params(params)
            r.set_jac_params(params)
        r.set_initial_value(y0, xout[0])
        if nx == 2 and not force_predefined:
            ysteps = [y0]
            xsteps = [xout[0]]
            while r.t < xout[1]:
                r.integrate(xout[1], step=True)  # vode itask 2 (may overshoot)
                if not r.successful():
                    raise RuntimeError("failed")
                xsteps.append(r.t)
                ysteps.append(r.y)
            yout = np.array(ysteps)
            xout = np.array(xsteps)
        else:
            yout = np.empty((nx, ny))
            yout[0, :] = y0
            for idx in range(1, nx):
                r.integrate(xout[idx])
                if not r.successful():
                    raise RuntimeError("failed")
                yout[idx, :] = r.y
        info = {'success': r.successful(), 'nrhs': rhs.ncall,
                'njac': jac.ncall}
        return self.post_process(xout, yout) + (info,)

    def _integrate(self, adaptive, predefined, xout, y0, params=(),
                   atol=1e-8, rtol=1e-8, first_step=None, with_jacobian=None,
                   force_predefined=False, **kwargs):
        xout, y0 = self.pre_process(xout, y0)
        if first_step is None:
            first_step = 1e-14 + xout[0]*1e-14  # arbitrary, often works
        nx = len(xout)
        new_kwargs = dict(dx0=first_step, atol=atol,
                          rtol=rtol, check_indexing=False)
        new_kwargs.update(kwargs)

        def _f(x, y, fout):
            if len(params) > 0:
                fout[:] = self.f_cb(x, y, params)
            else:
                fout[:] = self.f_cb(x, y)

        if with_jacobian is None:
            raise ValueError("Need to pass with_jacobian")
        elif with_jacobian is True:
            def _j(x, y, jout, dfdx_out=None, fy=None):
                if len(params) > 0:
                    jout[:, :] = self.j_cb(x, y, params)
                else:
                    jout[:, :] = self.j_cb(x, y)
                if dfdx_out is not None:
                    if len(params) > 0:
                        dfdx_out[:] = self.dfdx_cb(x, y, params)
                    else:
                        dfdx_out[:] = self.dfdx_cb(x, y)
        else:
            _j = None

        if self.roots_cb is not None:
            def _roots(x, y, out):
                if len(params) > 0:
                    out[:] = self.roots_cb(x, y, params)
                else:
                    out[:] = self.roots_cb(x, y)
            if 'roots' in new_kwargs:
                raise ValueError("cannot override roots")
            else:
                new_kwargs['roots'] = _roots
                if 'nroots' in new_kwargs:
                    raise ValueError("cannot override nroots")
                new_kwargs['nroots'] = self.nroots
        if nx == 2 and not force_predefined:
            xout, yout, info = adaptive(_f, _j, y0, *xout, **new_kwargs)
        else:
            yout, info = predefined(_f, _j, y0, xout, **new_kwargs)
        return self.post_process(xout, yout) + (info,)

    def integrate_gsl(self, *args, **kwargs):
        """
        Use GNU Scientific Library to integrate ODE system.

        Parameters
        ----------
        \*args:
            see :meth:`integrate`
        method: str (default: 'bsimp')
            what stepper to use, see ``gslodeiv2.steppers``
        \*\*kwargs:
            keyword arguments passed onto
            gslodeiv2.integrate_adaptive/gslodeiv2.integrate_predefined

        Returns
        -------
        Pair (length 2-tuple):
            2-dimensional array (first column indep., rest dep.), infodict
        """
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
        if 'lband' in kwargs or 'uband' in kwargs or 'band' in kwargs:
            raise ValueError("lband and uband set locally (set at"
                             " initialization instead)")
        if self.band is not None:
            kwargs['lband'], kwargs['uband'] = self.band
        return self._integrate(pycvodes.integrate_adaptive,
                               pycvodes.integrate_predefined,
                               *args, **kwargs)

    def _plot(self, cb, **kwargs):
        kwargs = kwargs.copy()
        if 'x' in kwargs or 'y' in kwargs:
            raise ValueError("x and y from internal_xout and internal_yout")

        if 'post_processor' in kwargs:
            raise ValueError("post_processor taken from self")
        else:
            kwargs['post_processor'] = self._post_processor

        if 'names' not in kwargs:
            kwargs['names'] = getattr(self, 'names', None)

        return cb(self.internal_xout, self.internal_yout, **kwargs)

    def plot_result(self, **kwargs):
        return self._plot(plot_result, **kwargs)

    def plot_phase_plane(self, indices=None, **kwargs):
        return self._plot(plot_phase_plane, indices=indices, **kwargs)

    def stiffness(self, xy=None, params=()):
        """
        Calculate sittness ratio, i.e. the ratio between the largest and
        smallest absolute eigenvalue of the jacobian matrix
        """
        from scipy.linalg import svd

        if xy is None:
            x, y = self.internal_xout, self.internal_yout
        else:
            x, y = self.pre_process(*xy)

        singular_values = []
        for xval, yvals in zip(x, y):
            J = self.j_cb(xval, yvals, params)
            if self.band is None:
                singular_values.append(svd(J, compute_uv=False))
            else:
                raise NotImplementedError

        return (np.abs(singular_values).max(axis=-1) /
                np.abs(singular_values).min(axis=-1))
