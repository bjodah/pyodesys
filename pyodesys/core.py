# -*- coding: utf-8 -*-
"""
Core functionality for ODESys.

Note that it is possible to use new custom ODE integrators with pyodesys by
providing a module with two functions named ``integrate_adaptive`` and
``integrate_predefined``. See the ``pyodesys.integrators`` module for examples.
"""

from __future__ import absolute_import, division, print_function


from collections import defaultdict
import os
import warnings

import numpy as np

from .util import _ensure_4args, _default
from .plotting import plot_result, plot_phase_plane
from .results import Result


class RecoverableError(Exception):
    pass


class ODESys(object):
    """ Object representing an ODE system.

    ``ODESys`` provides unified interface to:

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
    f : callback
        first derivatives of dependent variables (y) with respect to
        dependent variable (x). Signature is any of:
            - ``rhs(x, y[:]) -> f[:]``
            - ``rhs(x, y[:], p[:]) -> f[:]``
            - ``rhs(x, y[:], p[:], backend=math) -> f[:]``
    jac : callback
        Jacobian matrix (dfdy). Required for implicit methods.
    dfdx : callback
        Signature ``dfdx(x, y[:], p[:]) -> out[:]`` (used by e.g. GSL)
    first_step_cb : callback
        Signature ``step1st(x, y[:], p[:]) -> dx0`` (pass first_step==0 to use).
        This is available for ``cvode``, ``odeint`` & ``gsl``, but not for ``scipy``.
    roots : callback
        Signature ``roots(x, y[:], p[:]=(), backend=math) -> discr[:]``.
    nroots : int
        Length of return vector from ``roots``.
    band : tuple of 2 integers or None (default: None)
        If jacobian is banded: number of sub- and super-diagonals
    names : iterable of strings (default : None)
        Names of variables, used for referencing dependent variables by name
        and for labels in plots.
    param_names : iterable of strings (default: None)
        Names of the parameters, used for referencing parameters by name.
    dep_by_name : bool
        When ``True`` :meth:`integrate` expects a dictionary as input for y0.
    par_by_name : bool
        When ``True`` :meth:`integrate` expects a dictionary as input for params.
    latex_names : iterable of strings (default : None)
        Names of variables in LaTeX format (e.g. for labels in plots).
    latex_param_names : iterable of strings (default : None)
        Names of parameters in LaTeX format (e.g. for labels in plots).
    pre_processors : iterable of callables (optional)
        signature: f(x1[:], y1[:], params1[:]) -> x2[:], y2[:], params2[:].
        When modifying: insert at beginning.
    post_processors : iterable of callables (optional)
        signature: f(x2[:], y2[:, :], params2[:]) -> x1[:], y1[:, :],
        params1[:]
        When modifying: insert at end.
    append_iv :  bool
        If ``True`` params[:] passed to :attr:`f_cb`, :attr:`jac_cb` will contain
        initial values of y.
    autonomous_interface : bool (optional)
        If given, sets the :attr:`autonomous` to indicate whether
        the system appears autonomous or not upon call to :meth:`integrate`.

    Attributes
    ----------
    f_cb : callback
        For evaluating the vector of derivatives.
    j_cb : callback or None
        For evaluating the Jacobian matrix of f.
    dfdx_cb : callback or None
        For evaluating the second order derivatives.
    first_step_cb : callback or None
        For calculating the first step based on x0, y0 & p.
    roots_cb : callback
    nroots : int
    names : iterable of strings
    param_names : iterable of strings
    description : str
    dep_by_name : bool
    par_by_name : bool
    latex_names : iterable of str
    latex_param_names : iterable of str
    pre_processors : iterable of callbacks
    post_processors : iterable of callbacks
    append_iv : bool
    autonomous_interface : bool or None
        Indicates whether the system appears autonomous upon call to
        :meth:`integrate`. ``None`` indicates that it is unknown.

    Examples
    --------
    >>> odesys = ODESys(lambda x, y, p: p[0]*x + p[1]*y[0]*y[0])
    >>> yout, info = odesys.predefined([1], [0, .2, .5], [2, 1])
    >>> print(info['success'])
    True


    Notes
    -----
    Banded jacobians are supported by "scipy" and "cvode" integrators.

    """

    def __init__(self, f, jac=None, dfdx=None, first_step_cb=None, roots_cb=None, nroots=None,
                 band=None, names=None, param_names=None, description=None, dep_by_name=False,
                 par_by_name=False, latex_names=None, latex_param_names=None, pre_processors=None,
                 post_processors=None, append_iv=False, **kwargs):
        self.f_cb = _ensure_4args(f)
        self.j_cb = _ensure_4args(jac) if jac is not None else None
        self.dfdx_cb = dfdx
        self.first_step_cb = first_step_cb
        self.roots_cb = roots_cb
        self.nroots = nroots or 0
        if band is not None:
            if not band[0] >= 0 or not band[1] >= 0:
                raise ValueError("bands needs to be > 0 if provided")
        self.band = band
        self.names = names
        self.param_names = param_names
        self.description = description
        self.dep_by_name = dep_by_name
        self.par_by_name = par_by_name
        self.latex_names = latex_names
        self.latex_param_names = latex_param_names
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []
        self.append_iv = append_iv
        autonomous_interface = kwargs.pop('autonomous_interface', None)
        if hasattr(self, 'autonomous_interface'):
            if autonomous_interface is not None and autonomous_interface != self.autonomous_interface:
                raise ValueError("Got conflicting autonomous_interface infomation.")
        else:
            if (autonomous_interface is None and getattr(self, 'autonomous_exprs', False) and
               len(self.post_processors) == 0 and len(self.pre_processors) == 0):
                self.autonomous_interface = True
            else:
                self.autonomous_interface = autonomous_interface

        if self.autonomous_interface not in (True, False, None):
            raise ValueError("autonomous_interface needs to be a boolean value or None.")

        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs: %s" % str(kwargs))

    @staticmethod
    def _array_from_dict(d, keys):
        vals = [d[k] for k in keys]
        lens = [len(v) for v in vals if hasattr(v, '__len__')]
        if len(lens) == 0:
            return np.array(vals).T
        else:
            if not all(l == lens[0] for l in lens):
                raise ValueError("Mixed lenghts in dictionary.")
            out = np.empty((lens[0], len(vals)))
            for idx, v in enumerate(vals):
                out[:, idx] = v
            return out

    def pre_process(self, xout, y0, params=()):
        """ Transforms input to internal values, used internally. """
        if self.dep_by_name and isinstance(y0, dict):
            y0 = self._array_from_dict(y0, self.names)
        if self.par_by_name and isinstance(params, dict):
            params = self._array_from_dict(params, self.param_names)

        try:
            nx = len(xout)
            if nx == 1:
                xout = (0*xout[0], xout[0])
        except TypeError:
            xout = (0*xout, xout)

        for pre_processor in self.pre_processors:
            xout, y0, params = pre_processor(xout, y0, params)
        return np.atleast_1d(xout), np.atleast_1d(y0), np.atleast_1d(params)

    def post_process(self, xout, yout, params):
        """ Transforms internal values to output, used internally. """
        for post_processor in self.post_processors:
            xout, yout, params = post_processor(xout, yout, params)
        return xout, yout, params

    def adaptive(self, y0, x0, xend, params=(), **kwargs):
        """ Integrate with integrator chosen output.

        Parameters
        ----------
        integrator : str
            See :meth:`integrate`.
        y0 : array_like
            See :meth:`integrate`.
        x0 : float
            Initial value of the independent variable.
        xend : float
            Final value of the independent variable.
        params : array_like
            See :meth:`integrate`.
        \*\*kwargs :
            See :meth:`integrate`.

        Returns
        -------
        Same as :meth:`integrate`
        """
        return self.integrate((x0, xend), y0,
                              params=params, **kwargs)

    def predefined(self, y0, xout, params=(), **kwargs):
        """ Integrate with user chosen output.

        Parameters
        ----------
        integrator : str
            See :meth:`integrate`.
        y0 : array_like
            See :meth:`integrate`.
        xout : array_like
        params : array_like
            See :meth:`integrate`.
        \*\*kwargs:
            See :meth:`integrate`

        Returns
        -------
        Length 2 tuple : (yout, info)
            See :meth:`integrate`.
        """
        xout, yout, info = self.integrate(xout, y0, params=params,
                                          force_predefined=True, **kwargs)
        return yout, info

    def integrate(self, x, y0, params=(), **kwargs):
        """
        Integrate the system of ordinary differential equations.

        Parameters
        ----------
        x : array_like or pair (start and final time) or float
            if float:
                make it a pair: (0, x)
            if pair or length-2 array:
                initial and final value of the independent variable
            if array_like:
                values of independent variable report at
        y0 : array_like
            Initial values at x[0] for the dependent variables.
        params : array_like (default: tuple())
            Value of parameters passed to user-supplied callbacks.
        integrator : str or None
            Name of integrator, one of:
                - 'scipy': :meth:`_integrate_scipy`
                - 'gsl': :meth:`_integrate_gsl`
                - 'odeint': :meth:`_integrate_odeint`
                - 'cvode':  :meth:`_integrate_cvode`

            See respective method for more information.
            If ``None``: ``os.environ.get('PYODESYS_INTEGRATOR', 'scipy')``
        atol : float
            Absolute tolerance
        rtol : float
            Relative tolerance
        with_jacobian : bool or None (default)
            Whether to use the jacobian. When ``None`` the choice is
            done automatically (only used when required). This matters
            when jacobian is derived at runtime (high computational cost).
        force_predefined : bool (default: False)
            override behaviour of ``len(x) == 2`` => :meth:`adaptive`
        \*\*kwargs:
            Additional keyword arguments for ``_integrate_$(integrator)``.

        Returns
        -------
        Length 3 tuple: (x, yout, info)
            x : array of values of the independent variable
            yout : array of the dependent variable(s) for the different values of x
            info : dict ('nfev' is guaranteed to be a key)
        """
        intern_x, intern_y0, intern_p = self.pre_process(x, y0, params)
        intern_x = intern_x.squeeze()
        intern_y0 = np.atleast_1d(intern_y0)
        if self.append_iv:
            intern_p = np.concatenate((intern_p, intern_y0), axis=-1)
        if hasattr(self, 'ny'):
            if intern_y0.shape[-1] != self.ny:
                raise ValueError("Incorrect shape of intern_y0")
        integrator = kwargs.pop('integrator', None)
        if integrator is None:
            integrator = os.environ.get('PYODESYS_INTEGRATOR', 'scipy')

        if intern_y0.ndim == 1 and intern_p.ndim == 2:
            # repeat y based on p
            intern_y0 = np.tile(intern_y0, (intern_p.shape[0], 1))
        elif intern_y0.ndim == 2 and intern_p.ndim == 1:
            # repeat p based on p
            intern_p = np.tile(intern_p, (intern_y0.shape[0], 1))

        if intern_x.ndim == 1 and intern_y0.ndim == 2:
            # repeat x based on y
            intern_x = np.tile(intern_x, (intern_y0.shape[0], 1))

        ndims = (intern_x.ndim, intern_y0.ndim, intern_p.ndim)
        if ndims == (1, 1, 1):
            twodim = False
        elif ndims == (2, 2, 2):
            twodim = True
            if not intern_x.shape[0] == intern_y0.shape[0] == intern_p.shape[0]:
                raise ValueError("Inconsistent shape[0] in x, y, p: (%d, %d, %d)" % (
                    intern_x.shape[0], intern_y0.shape[0], intern_p.shape[0]))
        else:
            raise ValueError("Mixed number of dimensions")

        args = tuple(map(np.atleast_2d, (intern_x, intern_y0, intern_p)))

        if isinstance(integrator, str):
            nfo = getattr(self, '_integrate_' + integrator)(*args, **kwargs)
        else:
            kwargs['with_jacobian'] = getattr(integrator, 'with_jacobian', None)
            nfo = self._integrate(integrator.integrate_adaptive,
                                  integrator.integrate_predefined,
                                  *args, **kwargs)
        if twodim:
            if nfo[0]['mode'] == 'predefined':
                _xout = np.array([d['internal_xout'] for d in nfo])
                _yout = np.array([d['internal_yout'] for d in nfo])
                _params = np.array([d['internal_params'] for d in nfo])
            else:
                _xout = [d['internal_xout'] for d in nfo]
                _yout = [d['internal_yout'] for d in nfo]
                _params = [d['internal_params'] for d in nfo]
        else:
            _xout = nfo[0]['internal_xout']
            _yout = nfo[0]['internal_yout']
            _params = intern_p  # nfo[0]['internal_params']
            self._internal = _xout.copy(), _yout.copy(), _params.copy()
            nfo = nfo[0]
        return Result(*(self.post_process(_xout, _yout, _params) + (nfo, self)))

    def _integrate_scipy(self, intern_xout, intern_y0, intern_p,
                         atol=1e-8, rtol=1e-8, first_step=None, with_jacobian=None,
                         force_predefined=False, name=None, **kwargs):
        """ Do not use directly (use ``integrate('scipy', ...)``).

        Uses `scipy.integrate.ode <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_

        Parameters
        ----------
        \*args :
            See :meth:`integrate`.
        name : str (default: 'lsoda'/'dopri5' when jacobian is available/not)
            What integrator wrapped in scipy.integrate.ode to use.
        \*\*kwargs :
            Keyword arguments passed onto `set_integrator(...) <
        http://docs.scipy.org/doc/scipy/reference/generated/
        scipy.integrate.ode.set_integrator.html#scipy.integrate.ode.set_integrator>`_

        Returns
        -------
        See :meth:`integrate`.
        """
        from scipy.integrate import ode
        ny = intern_y0.shape[-1]
        nx = intern_xout.shape[-1]
        results = []
        for _xout, _y0, _p in zip(intern_xout, intern_y0, intern_p):
            if name is None:
                if self.j_cb is None:
                    name = 'dopri5'
                else:
                    name = 'lsoda'
            if with_jacobian is None:
                if name == 'lsoda':  # lsoda might call jacobian
                    with_jacobian = True
                elif name in ('dop853', 'dopri5'):
                    with_jacobian = False  # explicit steppers
                elif name == 'vode':
                    with_jacobian = kwargs.get('method', 'adams') == 'bdf'

            def rhs(t, y, p=()):
                rhs.ncall += 1
                return self.f_cb(t, y, p)
            rhs.ncall = 0

            if self.j_cb is not None:
                def jac(t, y, p=()):
                    jac.ncall += 1
                    return self.j_cb(t, y, p)
                jac.ncall = 0

            r = ode(rhs, jac=jac if with_jacobian else None)
            if 'lband' in kwargs or 'uband' in kwargs or 'band' in kwargs:
                raise ValueError("lband and uband set locally (set `band` at initialization instead)")
            if self.band is not None:
                kwargs['lband'], kwargs['uband'] = self.band
            r.set_integrator(name, atol=atol, rtol=rtol, **kwargs)
            if len(_p) > 0:
                r.set_f_params(_p)
                r.set_jac_params(_p)
            r.set_initial_value(_y0, _xout[0])
            if nx == 2 and not force_predefined:
                mode = 'adaptive'
                if name in ('vode', 'lsoda'):
                    warnings.warn("'adaptive' mode with SciPy's integrator (vode/lsoda) may overshoot (itask=2)")
                    # vode itask 2 (may overshoot)
                    ysteps = [_y0]
                    xsteps = [_xout[0]]
                    while r.t < _xout[1]:
                        r.integrate(_xout[1], step=True)
                        if not r.successful():
                            raise RuntimeError("failed")
                        xsteps.append(r.t)
                        ysteps.append(r.y)
                else:
                    xsteps, ysteps = [], []

                    def solout(x, y):
                        xsteps.append(x)
                        ysteps.append(y)
                    r.set_solout(solout)
                    r.integrate(_xout[1])
                    if not r.successful():
                        raise RuntimeError("failed")
                _yout = np.array(ysteps)
                _xout = np.array(xsteps)

            else:  # predefined
                mode = 'predefined'
                _yout = np.empty((nx, ny))
                _yout[0, :] = _y0
                for idx in range(1, nx):
                    r.integrate(_xout[idx])
                    if not r.successful():
                        raise RuntimeError("failed")
                    _yout[idx, :] = r.y
            info = {
                'internal_xout': _xout,
                'internal_yout': _yout,
                'internal_params': intern_p,
                'success': r.successful(),
                'nfev': rhs.ncall,
                'name': name,
                'mode': mode
            }
            if self.j_cb is not None:
                info['njev'] = jac.ncall
            results.append(info)
        return results

    def _integrate(self, adaptive, predefined, intern_xout, intern_y0, intern_p,
                   atol=1e-8, rtol=1e-8, first_step=0.0, with_jacobian=None,
                   force_predefined=False, **kwargs):
        nx = intern_xout.shape[-1]
        results = []
        for _xout, _y0, _p in zip(intern_xout, intern_y0, intern_p):
            new_kwargs = dict(dx0=first_step, atol=atol,
                              rtol=rtol, check_indexing=False)
            new_kwargs.update(kwargs)

            def _f(x, y, fout):
                try:
                    if len(_p) > 0:
                        fout[:] = np.asarray(self.f_cb(x, y, _p))
                    else:
                        fout[:] = np.asarray(self.f_cb(x, y))
                except RecoverableError:
                    return 1  # recoverable error

            if with_jacobian is None:
                raise ValueError("Need to pass with_jacobian")
            elif with_jacobian is True:
                def _j(x, y, jout, dfdx_out=None, fy=None):
                    if len(_p) > 0:
                        jout[:, :] = np.asarray(self.j_cb(x, y, _p))
                    else:
                        jout[:, :] = np.asarray(self.j_cb(x, y))
                    if dfdx_out is not None:
                        if len(_p) > 0:
                            dfdx_out[:] = np.asarray(self.dfdx_cb(x, y, _p))
                        else:
                            dfdx_out[:] = np.asarray(self.dfdx_cb(x, y))
            else:
                _j = None

            if self.first_step_cb is not None:
                def _first_step(x, y):
                    if len(_p) > 0:
                        return self.first_step_cb(x, y, _p)
                    else:
                        return self.first_step_cb(x, y)
                if 'dx0cb' in new_kwargs:
                    raise ValueError("cannot override dx0cb")
                else:
                    new_kwargs['dx0cb'] = _first_step

            if self.roots_cb is not None:
                def _roots(x, y, out):
                    if len(_p) > 0:
                        out[:] = np.asarray(self.roots_cb(x, y, _p))
                    else:
                        out[:] = np.asarray(self.roots_cb(x, y))
                if 'roots' in new_kwargs:
                    raise ValueError("cannot override roots")
                else:
                    new_kwargs['roots'] = _roots
                    if 'nroots' in new_kwargs:
                        raise ValueError("cannot override nroots")
                    new_kwargs['nroots'] = self.nroots
            if nx == 2 and not force_predefined:
                _xout, yout, info = adaptive(_f, _j, _y0, *_xout, **new_kwargs)
                info['mode'] = 'adaptive'
            else:
                yout, info = predefined(_f, _j, _y0, _xout, **new_kwargs)
                info['mode'] = 'predefined'

            info['internal_xout'] = _xout
            info['internal_yout'] = yout
            info['internal_params'] = _p
            results.append(info)
        return results

    def _integrate_gsl(self, *args, **kwargs):
        """ Do not use directly (use ``integrate(..., integrator='gsl')``).

        Uses `GNU Scientific Library <http://www.gnu.org/software/gsl/>`_
        (via `pygslodeiv2 <https://pypi.python.org/pypi/pygslodeiv2>`_)
        to integrate the ODE system.

        Parameters
        ----------
        \*args :
            see :meth:`integrate`
        method : str (default: 'bsimp')
            what stepper to use, see :py:attr:`gslodeiv2.steppers`
        \*\*kwargs :
            keyword arguments passed onto
            :py:func:`gslodeiv2.integrate_adaptive`/:py:func:`gslodeiv2.integrate_predefined`

        Returns
        -------
        See :meth:`integrate`
        """
        import pygslodeiv2  # Python interface GSL's "odeiv2" integrators
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'bsimp') in pygslodeiv2.requires_jac
        return self._integrate(pygslodeiv2.integrate_adaptive,
                               pygslodeiv2.integrate_predefined,
                               *args, **kwargs)

    def _integrate_odeint(self, *args, **kwargs):
        """ Do not use directly (use ``integrate(..., integrator='odeint')``).

        Uses `Boost.Numeric.Odeint <http://www.odeint.com>`_
        (via `pyodeint <https://pypi.python.org/pypi/pyodeint>`_) to integrate
        the ODE system.
        """
        import pyodeint  # Python interface to boost's odeint integrators
        kwargs['with_jacobian'] = kwargs.get(
            'method', 'rosenbrock4') in pyodeint.requires_jac
        return self._integrate(pyodeint.integrate_adaptive,
                               pyodeint.integrate_predefined,
                               *args, **kwargs)

    def _integrate_cvode(self, *args, **kwargs):
        """ Do not use directly (use ``integrate(..., integrator='cvode')``).

        Uses CVode from CVodes in
        `SUNDIALS <https://computation.llnl.gov/casc/sundials/>`_
        (via `pycvodes <https://pypi.python.org/pypi/pycvodes>`_)
        to integrate the ODE system. """
        import pycvodes  # Python interface to SUNDIALS's cvodes integrators
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

    def _plot(self, cb, internal_xout=None, internal_yout=None,
              internal_params=None, **kwargs):
        kwargs = kwargs.copy()
        if 'x' in kwargs or 'y' in kwargs or 'params' in kwargs:
            raise ValueError("x and y from internal_xout and internal_yout")

        if 'post_processors' not in kwargs:
            kwargs['post_processors'] = self.post_processors

        if 'names' not in kwargs:
            kwargs['names'] = getattr(self, 'names', None)
        else:
            if 'indices' not in kwargs and getattr(self, 'names', None) is not None:
                kwargs['indices'] = [self.names.index(n) for n in kwargs['names']]
                kwargs['names'] = self.names
        _internal = getattr(self, '_internal', [None]*3)
        return cb(_default(internal_xout, _internal[0]),
                  _default(internal_yout, _internal[1]),
                  _default(internal_params, _internal[2]), **kwargs)

    def plot_result(self, **kwargs):
        """ Plots the integrated dependent variables from last integration.

        This method will be deprecated. Please use :meth:`Result.plot`.
        See :func:`pyodesys.plotting.plot_result`
        """
        return self._plot(plot_result, **kwargs)

    def plot_phase_plane(self, indices=None, **kwargs):
        """ Plots a phase portrait from last integration.

        This method will be deprecated. Please use :meth:`Result.plot_phase_plane`.
        See :func:`pyodesys.plotting.plot_phase_plane`
        """
        return self._plot(plot_phase_plane, indices=indices, **kwargs)

    def _jac_eigenvals_svd(self, xval, yvals, intern_p):
        from scipy.linalg import svd
        J = self.j_cb(xval, yvals, intern_p)
        return svd(J, compute_uv=False)

    def stiffness(self, xyp=None, eigenvals_cb=None):
        """ [DEPRECATED] Use :meth:`Result.stiffness`, stiffness ration

        Running stiffness ratio from last integration.
        Calculate sittness ratio, i.e. the ratio between the largest and
        smallest absolute eigenvalue of the jacobian matrix. The user may
        supply their own routine for calculating the eigenvalues, or they
        will be calculated from the SVD (singular value decomposition).
        Note that calculating the SVD for any but the smallest Jacobians may
        prove to be prohibitively expensive.

        Parameters
        ----------
        xyp : length 3 tuple (default: None)
            internal_xout, internal_yout, internal_params, taken
            from last integration if not specified.
        eigenvals_cb : callback (optional)
            Signature (x, y, p) (internal variables), when not provided an
            internal routine will use ``self.j_cb`` and ``scipy.linalg.svd``.

        """
        if eigenvals_cb is None:
            if self.band is not None:
                raise NotImplementedError
            eigenvals_cb = self._jac_eigenvals_svd

        if xyp is None:
            x, y, intern_p = self._internal
        else:
            x, y, intern_p = self.pre_process(*xyp)

        singular_values = []
        for xval, yvals in zip(x, y):
            singular_values.append(eigenvals_cb(xval, yvals, intern_p))

        return (np.abs(singular_values).max(axis=-1) /
                np.abs(singular_values).min(axis=-1))


def _new_x(xout, x, guaranteed_autonomous):
    if guaranteed_autonomous:
        return 0, abs(x[-1] - xout[-1])  # rounding
    else:
        return xout[-1], x[-1]


def integrate_chained(odes, kw, x, y0, params=(), **kwargs):
    """ Auto-switching between formulations of ODE system.

    In case one has a formulation of a system of ODEs which is preferential in
    the beginning of the intergration this function allows the user to run the
    integration with this system where it takes a user-specified maximum number
    of steps before switching to another formulation (unless final value of the
    independent variables has been reached). Number of systems used i returned
    as ``nsys`` in info dict.

    Parameters
    ----------
    odes : iterable of :class:`OdeSy` instances
    kw : dict mapping kwarg to iterables of same legnth as ``odes``
    x : array_like
    y0 : array_like
    params : array_like
    \*\*kwargs:
        See :meth:`ODESys.integrate`

    Notes
    -----
    Plays particularly well with :class:`symbolic.TransformedSys`.

    """
    x_arr = np.asarray(x)
    if x_arr.shape[-1] > 2:
        raise NotImplementedError("Only adaptive support return_on_error for now")
    multimode = False if x_arr.ndim < 2 else x_arr.shape[0]
    nfo_keys = ('nfev', 'njev', 'time_cpu', 'time_wall')

    next_autonomous = getattr(odes[0], 'autonomous_interface', False) == True  # noqa (np.True_)
    if multimode:
        tot_x = [np.array([0] if next_autonomous else [x[_][0]]) for _ in range(multimode)]
        tot_y = [np.asarray([y0[_]]) for _ in range(multimode)]
        tot_nfo = [defaultdict(int) for _ in range(multimode)]
        glob_x = [_[0] for _ in x] if next_autonomous else [0.0]*multimode
    else:
        tot_x, tot_y, tot_nfo = np.array([0 if next_autonomous else x[0]]), np.asarray([y0]), defaultdict(int)
        glob_x = x[0] if next_autonomous else 0.0

    for oi in range(len(odes)):
        if oi < len(odes) - 1:
            next_autonomous = getattr(odes[oi+1], 'autonomous_interface', False) == True  # noqa (np.True_)
        _int_kw = kwargs.copy()
        for k, v in kw.items():
            _int_kw[k] = v[oi]
        xout, yout, nfo = odes[oi].integrate(x, y0, params, **_int_kw)

        if multimode:
            for idx in range(multimode):
                tot_x[idx] = np.concatenate((tot_x[idx], xout[idx][1:] + glob_x[idx]))
                tot_y[idx] = np.concatenate((tot_y[idx], yout[idx][1:, :]))
                for k in nfo_keys:
                    if k in nfo[idx]:
                        tot_nfo[idx][k] += nfo[idx][k]
                tot_nfo[idx]['success'] = nfo[idx]['success']
        else:
            tot_x = np.concatenate((tot_x, xout[1:] + glob_x))
            tot_y = np.concatenate((tot_y, yout[1:, :]))
            for k in nfo_keys:
                if k in nfo:
                    tot_nfo[k] += nfo[k]
            tot_nfo['success'] = nfo['success']

        if multimode:
            if all([d['success'] for d in nfo]):
                break
        else:
            if nfo['success']:
                break
        if oi < len(odes) - 1:
            if multimode:
                _x, y0 = [], []
                for idx in range(multimode):
                    _x.append(_new_x(xout[idx], x[idx], next_autonomous))
                    y0.append(yout[idx][-1, :])
                    if next_autonomous:
                        glob_x[idx] += xout[idx][-1]
                x = _x
            else:
                x = _new_x(xout, x, next_autonomous)
                y0 = yout[-1, :]
                if next_autonomous:
                    glob_x += xout[-1]
    if multimode:  # don't return defaultdict
        tot_nfo = [dict(nsys=oi+1, **_nfo) for _nfo in tot_nfo]
    else:
        tot_nfo = dict(nsys=oi+1, **tot_nfo)
    return tot_x, tot_y, tot_nfo


class OdeSys(ODESys):
    """ DEPRECATED, use ODESys instead. """
    pass
