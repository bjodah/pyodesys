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
    roots_cb : callback
        Signature ``roots_cb(x, y[:], p[:]=(), backend=math) -> discr[:]``.
    nroots : int
        Length of return vector from ``roots_cb``.
    band : tuple of 2 integers or None (default: None)
        If jacobian is banded: number of sub- and super-diagonals
    names : iterable of strings (default : None)
        Names of variables, used for referencing dependent variables by name
        and for labels in plots.
    param_names : iterable of strings (default: None)
        Names of the parameters, used for referencing parameters by name.
    indep_name : str
        Name of the independent variable
    dep_by_name : bool
        When ``True`` :meth:`integrate` expects a dictionary as input for y0.
    par_by_name : bool
        When ``True`` :meth:`integrate` expects a dictionary as input for params.
    latex_names : iterable of strings (default : None)
        Names of variables in LaTeX format (e.g. for labels in plots).
    latex_param_names : iterable of strings (default : None)
        Names of parameters in LaTeX format (e.g. for labels in plots).
    latex_indep_name : str
        LaTeX formatted name of independent variable.
    taken_names : iterable of str
        Names of dependent variables which are calculated in pre_processors
    pre_processors : iterable of callables (optional)
        signature: f(x1[:], y1[:], params1[:]) -> x2[:], y2[:], params2[:].
        When modifying: insert at beginning.
    post_processors : iterable of callables (optional)
        signature: f(x2[:], y2[:, :], params2[:]) -> x1[:], y1[:, :],
        params1[:]
        When modifying: insert at end.
    append_iv :  bool
        See :attr:`append_iv`.
    autonomous_interface : bool (optional)
        If given, sets the :attr:`autonomous_interface` to indicate whether
        the system appears autonomous or not upon call to :meth:`integrate`.
    autonomous_exprs : bool
        Describes whether the independent variable appears in the rhs expressions.
        If set to ``True`` the underlying solver is allowed to shift the
        independent variable during integration.

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
    names : tuple of strings
    param_names : tuple of strings
    description : str
    dep_by_name : bool
    par_by_name : bool
    latex_names : tuple of str
    latex_param_names : tuple of str
    pre_processors : iterable of callbacks
    post_processors : iterable of callbacks
    append_iv : bool
        If ``True`` params[:] passed to :attr:`f_cb`, :attr:`jac_cb` will contain
        initial values of y. Note that this happens after pre processors have been
        applied.
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
                 band=None, names=(), param_names=(), indep_name=None, description=None, dep_by_name=False,
                 par_by_name=False, latex_names=(), latex_param_names=(), latex_indep_name=None,
                 taken_names=None, pre_processors=None, post_processors=None, append_iv=False,
                 autonomous_interface=None, to_arrays_callbacks=None, autonomous_exprs=None,
                 _indep_autonomous_key=None, **kwargs):
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
        self.names = tuple(names or ())
        self.param_names = tuple(param_names or ())
        self.indep_name = indep_name
        self.description = description
        self.dep_by_name = dep_by_name
        self.par_by_name = par_by_name
        self.latex_names = tuple(latex_names or ())
        self.latex_param_names = tuple(latex_param_names or ())
        self.latex_indep_name = latex_indep_name
        self.taken_names = tuple(taken_names or ())
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []
        self.append_iv = append_iv
        self.autonomous_exprs = autonomous_exprs
        if hasattr(self, 'autonomous_interface'):
            if autonomous_interface is not None and autonomous_interface != self.autonomous_interface:
                raise ValueError("Got conflicting autonomous_interface infomation.")
        else:
            if (autonomous_interface is None and self.autonomous_exprs and
               len(self.post_processors) == 0 and len(self.pre_processors) == 0):
                self.autonomous_interface = True
            else:
                self.autonomous_interface = autonomous_interface

        if self.autonomous_interface not in (True, False, None):
            raise ValueError("autonomous_interface needs to be a boolean value or None.")
        self._indep_autonomous_key = _indep_autonomous_key
        self.to_arrays_callbacks = to_arrays_callbacks
        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs: %s" % str(kwargs))

    @staticmethod
    def _array_from_dict(d, keys):
        vals = [d[k] for k in keys]
        lens = [len(v) for v in vals if hasattr(v, '__len__') and getattr(v, 'ndim', 1) > 0]
        if len(lens) == 0:
            return vals, True
        else:
            if not all(l == lens[0] for l in lens):
                raise ValueError("Mixed lenghts in dictionary.")
            out = np.empty((lens[0], len(vals)), dtype=object)
            for idx, v in enumerate(vals):
                if getattr(v, 'ndim', -1) == 0:
                    for j in range(lens[0]):
                        out[j, idx] = v
                else:
                    try:
                        for j in range(lens[0]):
                            out[j, idx] = v[j]
                    except TypeError:
                        out[:, idx] = v
            return out, False

    def _conditional_from_dict(self, cont, by_name, names):
        if isinstance(cont, dict):
            if not by_name:
                raise ValueError("not by name, yet a dictionary was passed.")
            cont, tp = self._array_from_dict(cont, names)
        else:
            tp = False
        return cont, tp

    def to_arrays(self, x, y, p, callbacks=None):
        try:
            nx = len(x)
        except TypeError:
            _x = 0*x, x
        else:
            _x = (0*x[0], x[0]) if nx == 0 else x

        _names = [n for n in self.names if n not in self.taken_names]
        if self._indep_autonomous_key:
            if isinstance(y, dict):
                if self._indep_autonomous_key not in y:
                    y = y.copy()
                    y[self._indep_autonomous_key] = _x[0]
            else:  # y is array like
                y = np.atleast_1d(y)
                if y.shape[-1] == self.ny:
                    pass
                elif y.shape[-1] == self.ny - 1:
                    y = np.concatenate((y, _x[0]*np.ones(y.shape[:-1] + (1,))), axis=-1)
                else:
                    raise ValueError("y of incorrect size")

        _y, tp_y = self._conditional_from_dict(y, self.dep_by_name, _names)
        _p, tp_p = self._conditional_from_dict(p, self.par_by_name, self.param_names)
        del _names

        callbacks = callbacks or self.to_arrays_callbacks
        if callbacks is not None:  # e.g. dedimensionalisation
            if len(callbacks) != 3:
                raise ValueError("Need 3 callbacks/None values.")
            _x, _y, _p = [e if cb is None else cb(e) for cb, e in zip(callbacks, [_x, _y, _p])]

        arrs = [arr.T if tp else arr for tp, arr in
                zip([False, tp_y, tp_p], map(np.atleast_1d, (_x, _y, _p)))]
        extra_shape = None
        for a in arrs:
            if a.ndim == 1:
                continue
            elif a.ndim == 2:
                if extra_shape is None:
                    extra_shape = a.shape[0]
                else:
                    if extra_shape != a.shape[0]:
                        raise ValueError("Size mismatch!")
            else:
                raise NotImplementedError("Only 2 dimensions currently supported.")
        if extra_shape is not None:
            arrs = [a if a.ndim == 2 else np.tile(a, (extra_shape, 1)) for a in arrs]
        return arrs

    def pre_process(self, xout, y0, params=()):
        """ Transforms input to internal values, used internally. """
        for pre_processor in self.pre_processors:
            xout, y0, params = pre_processor(xout, y0, params)
        return [np.atleast_1d(arr) for arr in (xout, y0, params)]

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

    def integrate(self, x, y0, params=(), atol=1e-8, rtol=1e-8, **kwargs):
        """ Integrate the system of ordinary differential equations.

        Solves the initial value problem (IVP).

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
        \\*\\*kwargs :
            Additional keyword arguments for ``_integrate_$(integrator)``.

        Returns
        -------
        Length 3 tuple: (x, yout, info)
            x : array of values of the independent variable
            yout : array of the dependent variable(s) for the different
                values of x.
            info : dict ('nfev' is guaranteed to be a key)
        """
        arrs = self.to_arrays(x, y0, params)
        _x, _y, _p = _arrs = self.pre_process(*arrs)
        ndims = [a.ndim for a in _arrs]
        if ndims == [1, 1, 1]:
            twodim = False
        elif ndims == [2, 2, 2]:
            twodim = True
        else:
            raise ValueError("Pre-processor made ndims inconsistent?")

        if self.append_iv:
            _p = np.concatenate((_p, _y), axis=-1)

        if hasattr(self, 'ny'):
            if _y.shape[-1] != self.ny:
                raise ValueError("Incorrect shape of intern_y0")
        if isinstance(atol, dict):
            kwargs['atol'] = [atol[k] for k in self.names]
        else:
            kwargs['atol'] = atol
        kwargs['rtol'] = rtol

        integrator = kwargs.pop('integrator', None)
        if integrator is None:
            integrator = os.environ.get('PYODESYS_INTEGRATOR', 'scipy')

        args = tuple(map(np.atleast_2d, (_x, _y, _p)))

        self._current_integration_kwargs = kwargs
        if isinstance(integrator, str):
            nfo = getattr(self, '_integrate_' + integrator)(*args, **kwargs)
        else:
            kwargs['with_jacobian'] = getattr(integrator, 'with_jacobian', None)
            nfo = self._integrate(integrator.integrate_adaptive,
                                  integrator.integrate_predefined,
                                  *args, **kwargs)
        if twodim:
            _xout = [d['internal_xout'] for d in nfo]
            _yout = [d['internal_yout'] for d in nfo]
            _params = [d['internal_params'] for d in nfo]
            res = [Result(*(self.post_process(_xout[i], _yout[i], _params[i]) + (nfo[i], self)))
                   for i in range(len(nfo))]
        else:
            _xout = nfo[0]['internal_xout']
            _yout = nfo[0]['internal_yout']

            self._internal = _xout.copy(), _yout.copy(), _p.copy()
            nfo = nfo[0]
            res = Result(*(self.post_process(_xout, _yout, _p) + (nfo, self)))
        return res

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
                    warnings.warn("'adaptive' mode with SciPy's integrator is unreliable, consider using e.g. cvode")
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
                'internal_params': _p,
                'success': r.successful(),
                'nfev': rhs.ncall,
                'n_steps': -1,  # don't know how to obtain this number
                'name': name,
                'mode': mode,
                'atol': atol,
                'rtol': rtol
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
        kwargs['autonomous_exprs'] = self.autonomous_exprs
        return self._integrate(pycvodes.integrate_adaptive,
                               pycvodes.integrate_predefined,
                               *args, **kwargs)

    def _plot(self, cb, internal_xout=None, internal_yout=None,
              internal_params=None, **kwargs):
        kwargs = kwargs.copy()
        if 'x' in kwargs or 'y' in kwargs or 'params' in kwargs:
            raise ValueError("x and y from internal_xout and internal_yout")

        _internal = getattr(self, '_internal', [None]*3)
        x, y, p = (_default(internal_xout, _internal[0]),
                   _default(internal_yout, _internal[1]),
                   _default(internal_params, _internal[2]))
        for post_processor in self.post_processors:
            x, y, p = post_processor(x, y, p)

        if 'names' not in kwargs:
            kwargs['names'] = getattr(self, 'names', None)
        else:
            if 'indices' not in kwargs and getattr(self, 'names', None) is not None:
                kwargs['indices'] = [self.names.index(n) for n in kwargs['names']]
                kwargs['names'] = self.names
        return cb(x, y, **kwargs)

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


class OdeSys(ODESys):
    """ DEPRECATED, use ODESys instead. """
    pass


def _new_x(xout, x, guaranteed_autonomous):
    if guaranteed_autonomous:
        return 0, abs(x[-1] - xout[-1])  # rounding
    else:
        return xout[-1], x[-1]


def integrate_auto_switch(odes, kw, x, y0, params=(), **kwargs):
    """ Auto-switching between formulations of ODE system.

    In case one has a formulation of a system of ODEs which is preferential in
    the beginning of the integration, this function allows the user to run the
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
        res = odes[oi].integrate(x, y0, params, **_int_kw)

        if multimode:
            for idx in range(multimode):
                tot_x[idx] = np.concatenate((tot_x[idx], res[idx].xout[1:] + glob_x[idx]))
                tot_y[idx] = np.concatenate((tot_y[idx], res[idx].yout[1:, :]))
                for k in nfo_keys:
                    if k in res[idx].info:
                        tot_nfo[idx][k] += res[idx].info[k]
                tot_nfo[idx]['success'] = res[idx].info['success']
        else:
            tot_x = np.concatenate((tot_x, res.xout[1:] + glob_x))
            tot_y = np.concatenate((tot_y, res.yout[1:, :]))
            for k in nfo_keys:
                if k in res.info:
                    tot_nfo[k] += res.info[k]
            tot_nfo['success'] = res.info['success']

        if multimode:
            if all([r.info['success'] for r in res]):
                break
        else:
            if res.info['success']:
                break
        if oi < len(odes) - 1:
            if multimode:
                _x, y0 = [], []
                for idx in range(multimode):
                    _x.append(_new_x(res[idx].xout, x[idx], next_autonomous))
                    y0.append(res[idx].yout[-1, :])
                    if next_autonomous:
                        glob_x[idx] += res[idx].xout[-1]
                x = _x
            else:
                x = _new_x(res.xout, x, next_autonomous)
                y0 = res.yout[-1, :]
                if next_autonomous:
                    glob_x += res.xout[-1]
    if multimode:  # don't return defaultdict
        tot_nfo = [dict(nsys=oi+1, **_nfo) for _nfo in tot_nfo]
        return [Result(tot_x[idx], tot_y[idx], res[idx].params, tot_nfo[idx], odes[0])
                for idx in range(len(res))]
    else:
        tot_nfo = dict(nsys=oi+1, **tot_nfo)
        return Result(tot_x, tot_y, res.params, tot_nfo, odes[0])


integrate_chained = integrate_auto_switch  # deprecated name


def chained_parameter_variation(subject, durations, y0, varied_params, default_params=None,
                                integrate_kwargs=None, x0=None, npoints=1):
    """ Integrate an ODE-system for a serie of durations with some parameters changed in-between

    Parameters
    ----------
    subject : function or ODESys instance
        If a function: should have the signature of :meth:`pyodesys.ODESys.integrate`
        (and resturn a :class:`pyodesys.results.Result` object).
        If a ODESys instance: the ``integrate`` method will be used.
    durations : iterable of floats
        Spans of the independent variable.
    y0 : dict or array_like
    varied_params : dict mapping parameter name (or index) to array_like
        Each array_like need to be of same length as durations.
    default_params : dict or array_like
        Default values for the parameters of the ODE system.
    integrate_kwargs : dict
        Keyword arguments passed on to ``integrate``.
    x0 : float-like
        First value of independent variable. default: 0.
    npoints : int
        Number of points per sub-interval.

    """
    assert len(durations) > 0, 'need at least 1 duration (preferably many)'
    for k, v in varied_params.items():
        if len(v) != len(durations):
            raise ValueError("Mismathced lengths of durations and varied_params")

    if isinstance(subject, ODESys):
        integrate = subject.integrate
    else:
        integrate = subject

    default_params = default_params or {}
    integrate_kwargs = integrate_kwargs or {}

    def _get_idx(cont, idx):
        if isinstance(cont, dict):
            return {k: (v[idx] if hasattr(v, '__len__') and getattr(v, 'ndim', 1) > 0 else v)
                    for k, v in cont.items()}
        else:
            return cont[idx]

    durations = np.cumsum(durations)
    for idx_dur in range(len(durations)):
        params = default_params.copy()
        for k, v in varied_params.items():
            params[k] = v[idx_dur]
        if idx_dur == 0:
            if x0 is None:
                x0 = durations[0]*0
            out = integrate(np.linspace(x0, durations[0], npoints + 1), y0, params, **integrate_kwargs)
        else:
            if isinstance(out, Result):
                out.extend_by_integration(durations[idx_dur], params, npoints=npoints, **integrate_kwargs)
            else:
                for idx_res, r in enumerate(out):
                    r.extend_by_integration(durations[idx_dur], _get_idx(params, idx_res),
                                            npoints=npoints, **integrate_kwargs)

    return out
