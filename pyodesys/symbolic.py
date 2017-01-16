# -*- coding: utf-8 -*-
"""
This module contains a subclass of ODESys which allows the user to generate
auxiliary expressions from a canonical set of symbolic expressions. Subclasses
are also provided for dealing with variable transformations and partially
solved systems.
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from itertools import repeat

import numpy as np

try:
    from sym import Backend
except ImportError:
    class Backend(object):
        def __call__(self, *args, **kwargs):
            raise ImportError("Could not import package 'sym'.")

from .core import ODESys, RecoverableError
from .util import (
    transform_exprs_dep, transform_exprs_indep, _ensure_4args, _Wrapper
)


def _get_ny_nparams_from_kw(ny, nparams, kwargs):
    if kwargs.get('dep_by_name', False):
        if 'names' not in kwargs:
            raise ValueError("Need ``names`` in kwargs.")
        if ny is None:
            ny = len(kwargs['names'])
        elif ny != len(kwargs['names']):
            raise ValueError("Inconsistent between ``ny`` and length of ``names``.")

    if kwargs.get('par_by_name', False):
        if 'param_names' not in kwargs:
            raise ValueError("Need ``param_names`` in kwargs.")
        if nparams is None:
            nparams = len(kwargs['param_names'])
        elif nparams != len(kwargs['param_names']):
            raise ValueError("Inconsistent between ``nparams`` and length of ``param_names``.")

    if nparams is None:
        nparams = 0

    if ny is None:
        raise ValueError("Need ``ny`` or ``names`` together with ``dep_by_name==True``.")

    if kwargs.get('names', None) is not None and kwargs.get('param_names', None) is not None:
        all_names = set.union(set(kwargs['names']), set(kwargs['param_names']))
        if len(all_names) < len(kwargs['names']) + len(kwargs['param_names']):
            raise ValueError("Names of dependent variables cannot be used a parameter names")

    return ny, nparams


class SymbolicSys(ODESys):
    """ ODE System from symbolic expressions

    Creates a :class:`ODESys` instance
    from symbolic expressions. Jacboian and second derivatives
    are derived when needed.

    Parameters
    ----------
    dep_exprs : iterable of (symbol, expression)-pairs
    indep : Symbol
        Independent variable (default: None => autonomous system).
    params : iterable of symbols
        Problem parameters.
    jac : ImmutableMatrix or bool (default: True)
        if True:
            calculate jacobian from exprs
        if False:
            do not compute jacobian (use explicit steppers)
        if instance of ImmutableMatrix:
            user provided expressions for the jacobian
    dfdx : iterable of expressions
        Derivatives of :attr:`exprs` with respect to :attr`indep`.
    first_step_expr : expression
        Closed form expression for calculating the first step. Be sure to pass
        ``first_step==0`` to enable its use during integration. If not given,
        the solver default behavior will be invoked.
    roots : iterable of expressions
        Equations to look for root's for during integration
        (currently available through cvode).
    backend : str or sym.Backend
        See documentation of `sym.Backend \
        <http://bjodah.github.io/sym/latest/sym.html#module-sym.backend>`_.
    lower_bounds : array_like
        Convenience option setting magnitude constraint. (requires integrator with
        support for recoverable errors)
    upper_bounds : array_like
        Convenience option setting magnitude constraint. (requires integrator with
        support for recoverable errors)
    linear_invariants : Matrix
        Matrix specifing linear combinations of dependent variables that
    nonlinear_invariants : iterable of expressions
        Iterable collection of expressions of nonlinear invariants.
    \*\*kwargs:
        See :py:class:`ODESys`

    Attributes
    ----------
    dep : tuple of symbols
        Dependent variables.
    exprs : tuple of expressions
        Expressions for the derivatives of the dependent variables
        (:attr:`dep`) with respect to the independent variable (:attr:`indep`).
    indep : Symbol or None
        Independent variable (``None`` indicates autonomous system).
    params : iterable of symbols
        Problem parameters.
    first_step_expr : expression
        Closed form expression for how to compute the first step.
    roots : iterable of expressions or None
        Roots to report for during integration.
    be : module
        Symbolic backend, e.g. ``sympy`` or ``symcxx``.
    ny : int
        ``len(self.dep)`` note that this is not neccessarily the expected length of
        ``y0`` in the case of e.g. :class:`PartiallySolvedSystem`.
    be : module
        Symbolic backend.

    """

    _attrs_to_copy = ('first_step_expr', 'names', 'param_names', 'dep_by_name', 'par_by_name',
                      'latex_names', 'latex_param_names')

    def __init__(self, dep_exprs, indep=None, params=None, jac=True, dfdx=True, first_step_expr=None,
                 roots=None, backend=None, lower_bounds=None, upper_bounds=None,
                 linear_invariants=None, nonlinear_invariants=None,
                 linear_invariant_names=None, nonlinear_invariant_names=None, **kwargs):
        self.dep, self.exprs = zip(*dep_exprs)
        self.indep = indep
        if params is None:
            params = tuple(filter(lambda x: x not in self.dep, set.union(*[expr.free_symbols for expr in self.exprs])))
        self.params = params
        self._jac = jac
        self._dfdx = dfdx
        self.first_step_expr = first_step_expr
        self.roots = roots
        self.be = Backend(backend)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        if linear_invariants is not None:
            linear_invariants = self.be.Matrix(linear_invariants)
            if len(linear_invariants.shape) != 2 or linear_invariants.shape[1] != self.ny:
                raise ValueError("Incorrect shape of linear_invariants Matrix: %s" % str(linear_invariants.shape))
        self.linear_invariants = linear_invariants
        self.nonlinear_invariants = nonlinear_invariants
        if linear_invariant_names is not None:
            if len(linear_invariant_names) != linear_invariants.shape[0]:
                raise ValueError("Incorrect length of linear_invariant_names: %d (expected %d)" % (
                    len(linear_invariant_names), linear_invariants.shape[0]))
        self.linear_invariant_names = linear_invariant_names
        if nonlinear_invariant_names is not None:
            if len(nonlinear_invariant_names) != len(nonlinear_invariants):
                raise ValueError("Incorrect length of nonlinear_invariant_names: %d (expected %d)" % (
                    len(nonlinear_invariant_names), len(nonlinear_invariants)))
        self.nonlinear_invariant_names = nonlinear_invariant_names
        _names = kwargs.get('names', None)
        if _names is True:
            kwargs['names'] = _names = [y.name for y in self.dep]
        if self.indep is not None and _names is not None:
            if self.indep.name in _names:
                raise ValueError("Independent variable cannot share name with any dependent variable")

        # we need self.band before super().__init__
        self.band = kwargs.get('band', None)
        super(SymbolicSys, self).__init__(
            self.get_f_ty_callback(),
            self.get_j_ty_callback(),
            self.get_dfdx_callback(),
            self.get_first_step_callback(),
            self.get_roots_callback(),
            nroots=None if roots is None else len(roots),
            **kwargs)

    def all_invariants(self):
        return (([] if self.linear_invariants is None else (self.linear_invariants * self.dep).tolist()) +
                ([] if self.nonlinear_invariants is None else self.nonlinear_invariants))

    def all_invariant_names(self):
        return (self.linear_invariant_names or []) + (self.nonlinear_invariant_names or [])

    def __getitem__(self, key):
        return self.dep[self.names.index(key)]

    def pre_process(self, xout, y0, params=()):
        if not self.dep_by_name and isinstance(y0, dict):
            y0 = [y0[symb] for symb in self.dep]
        if not self.par_by_name and isinstance(params, dict):
            params = [params[symb] for symb in self.params]
        return super(SymbolicSys, self).pre_process(xout, y0, params)

    @classmethod
    def from_callback(cls, rhs, ny=None, nparams=None, first_step_factory=None,
                      roots_cb=None, **kwargs):
        """ Create an instance from a callback.

        Parameters
        ----------
        rhs : callbable
            Signature ``rhs(x, y[:], p[:], backend=math) -> f[:]``.
        ny : int
            Length of ``y`` in ``rhs``.
        nparams : int
            Length of ``p`` in ``rhs``.
        first_step_factory : callabble
            Signature ``step1st(x, y[:], p[:]) -> dx0``.
        dep_by_name : bool
            Make ``y`` passed to ``rhs`` a dict (keys from :attr:`names`) and convert
            its return value from dict to array.
        par_by_name : bool
            Make ``p`` passed to ``rhs`` a dict (keys from :attr:`param_names`).
        roots_cb : callable
            Callback with signature ``roots(x, y[:], p[:], backend=math) -> r[:]``.
        \*\*kwargs :
            Keyword arguments passed onto :class:`SymbolicSys`.

        Examples
        --------
        >>> def decay(x, y, p, backend=None):
        ...     rate = y['Po-210']*p[0]
        ...     return {'Po-210': -rate, 'Pb-206': rate}
        ...
        >>> odesys = SymbolicSys.from_callback(decay, dep_by_name=True, names=('Po-210', 'Pb-206'), nparams=1)
        >>> xout, yout, info = odesys.integrate([0, 138.4*24*3600], {'Po-210': 1.0, 'Pb-206': 0.0}, [5.798e-8])
        >>> import numpy as np; np.allclose(yout[-1, :], [0.5, 0.5], rtol=1e-3, atol=1e-3)
        True


        Returns
        -------
        An instance of :class:`SymbolicSys`.
        """
        ny, nparams = _get_ny_nparams_from_kw(ny, nparams, kwargs)
        be = Backend(kwargs.pop('backend', None))
        x, = be.real_symarray('x', 1)
        y = be.real_symarray('y', ny)
        p = be.real_symarray('p', nparams)
        _y = dict(zip(kwargs['names'], y)) if kwargs.get('dep_by_name', False) else y
        _p = dict(zip(kwargs['param_names'], p)) if kwargs.get('par_by_name', False) else p
        try:
            exprs = rhs(x, _y, _p, be)
        except TypeError:
            exprs = _ensure_4args(rhs)(x, _y, _p, be)
        if roots_cb is not None:
            if 'roots' in kwargs:
                raise ValueError("Keyword argument ``roots`` already given.")

            try:
                roots = roots_cb(x, _y, _p, be)
            except TypeError:
                roots = _ensure_4args(roots_cb)(x, _y, _p, be)

            kwargs['roots'] = roots

        if first_step_factory is not None:
            if 'first_step_exprs' in kwargs:
                raise ValueError("Cannot override first_step_exprs.")
            try:
                kwargs['first_step_expr'] = first_step_factory(x, _y, _p, be)
            except TypeError:
                kwargs['first_step_expr'] = _ensure_4args(first_step_factory)(x, _y, _p, be)
        if kwargs.get('dep_by_name', False):
            exprs = [exprs[k] for k in kwargs['names']]
        return cls(zip(y, exprs), x, p, backend=be, **kwargs)

    @classmethod
    def from_other(cls, ori, **kwargs):  # provisional
        if ori.roots is not None:
            raise NotImplementedError('roots currently unsupported')
        for k in cls._attrs_to_copy + ('params',):
            if k not in kwargs:
                kwargs[k] = getattr(ori, k)
        if 'lower_bounds' not in kwargs and getattr(ori, 'lower_bounds'):
            kwargs['lower_bounds'] = ori.lower_bounds
        if 'upper_bounds' not in kwargs and getattr(ori, 'upper_bounds'):
            kwargs['upper_bounds'] = ori.upper_bounds

        if len(ori.pre_processors) > 0:
            if 'pre_processors' not in kwargs:
                kwargs['pre_processors'] = []
            kwargs['pre_processors'] = kwargs['pre_processors'] + ori.pre_processors

        if len(ori.post_processors) > 0:
            if 'post_processors' not in kwargs:
                kwargs['post_processors'] = []
            kwargs['post_processors'] = ori.post_processors + kwargs['post_processors']

        instance = cls(zip(ori.dep, ori.exprs), ori.indep, **kwargs)
        for attr in ori._attrs_to_copy:
            if attr not in cls._attrs_to_copy:
                setattr(instance, attr, getattr(ori, attr))
        return instance

    @property
    def ny(self):
        """ Number of dependent variables in the system. """
        return len(self.exprs)

    @property
    def autonomous_exprs(self):
        """ Whether the expressions for the dependent variables are autonomous.

        Note that the system may still behave as an autonomous system on the interface
        of :meth:`integrate` due to use of pre-/post-processors.
        """
        if hasattr(self, '_autonomous_exprs'):
            return self._autonomous_exprs
        if self.indep is None:
            self._autonomous_exprs = True
            return True
        for expr in self.exprs:
            try:
                in_there = self.indep in expr.free_symbols
            except:
                in_there = expr.has(self.indep)
            if in_there:
                self._autonomous_exprs = False
                return False
        self._autonomous_exprs = True
        return True

    def get_jac(self):
        """ Derives the jacobian from ``self.exprs`` and ``self.dep``. """
        if self._jac is True:
            if self.band is None:
                f = self.be.Matrix(1, self.ny, self.exprs)
                self._jac = f.jacobian(self.be.Matrix(1, self.ny, self.dep))
            else:  # Banded
                self._jac = self.be.banded_jacobian(self.exprs, self.dep, *self.band)
        elif self._jac is False:
            return False

        return self._jac

    def jacobian_singular(self):
        """ Returns True if Jacobian is singular, else False. """
        cses, (jac_in_cses,) = self.be.cse(self.get_jac())
        try:
            jac_in_cses.LUdecomposition()
        except ValueError:
            return True
        else:
            return False

    def get_dfdx(self):
        """ Calculates 2nd derivatives of ``self.exprs`` """
        if self._dfdx is True:
            if self.indep is None:
                zero = 0*self.be.Dummy()**0
                self._dfdx = self.be.Matrix(1, self.ny, [zero]*self.ny)
            else:
                self._dfdx = self.be.Matrix(1, self.ny, [expr.diff(self.indep) for expr in self.exprs])
        elif self._dfdx is False:
            return False
        return self._dfdx

    def _callback_factory(self, exprs):
        args = [self.indep] + list(self.dep) + list(self.params)
        return _Wrapper(self.be.Lambdify(args, exprs), len(self.dep))

    def get_f_ty_callback(self):
        """ Generates a callback for evaluating ``self.exprs``. """
        cb = self._callback_factory(self.exprs)
        lb = getattr(self, 'lower_bounds', None)
        ub = getattr(self, 'upper_bounds', None)
        if lb is not None or ub is not None:
            def _bounds_wrapper(t, y, p=(), be=None):
                if lb is not None:
                    if np.any(y < lb):
                        raise RecoverableError
                if ub is not None:
                    if np.any(y > ub):
                        raise RecoverableError
                return cb(t, y, p, be)
            return _bounds_wrapper
        else:
            return cb

    def get_j_ty_callback(self):
        """ Generates a callback for evaluating the jacobian. """
        j_exprs = self.get_jac()
        if j_exprs is False:
            return None
        return self._callback_factory(j_exprs)

    def get_dfdx_callback(self):
        """ Generate a callback for evaluating derivative of ``self.exprs`` """
        dfdx_exprs = self.get_dfdx()
        if dfdx_exprs is False:
            return None
        return self._callback_factory(dfdx_exprs)

    def get_first_step_callback(self):
        if self.first_step_expr is None:
            return None
        return self._callback_factory([self.first_step_expr])

    def get_roots_callback(self):
        """ Generate a callback for evaluating ``self.roots`` """
        if self.roots is None:
            return None
        return self._callback_factory(self.roots)

    def get_invariants_callback(self):
        invar = self.all_invariants()
        if len(invar) == 0:
            return None
        return self._callback_factory(invar)

    # Not working yet:
    def _integrate_mpmath(self, xout, y0, params=()):
        """ Not working at the moment, need to fix
        (low priority - taylor series is a poor method)"""
        raise NotImplementedError
        xout, y0, self.internal_params = self.pre_process(xout, y0, params)
        from mpmath import odefun

        def rhs(x, y):
            rhs.ncall += 1
            return [
                e.subs(
                    ([(self.indep, x)] if self.indep is not None else []) +
                    list(zip(self.dep, y))
                ) for e in self.exprs
            ]
        rhs.ncall = 0

        cb = odefun(rhs, xout[0], y0)
        yout = []
        for x in xout:
            yout.append(cb(x))
        info = {'nrhs': rhs.ncall}
        return self.post_process(xout, yout, self.internal_params)[:2] + (info,)

    def _get_analytic_stiffness_cb(self):
        J = self.get_jac()
        eig_vals = list(J.eigenvals().keys())
        return self._callback_factory(eig_vals)

    def analytic_stiffness(self, xyp=None):
        """ Running stiffness ratio from last integration.

        Calculate sittness ratio, i.e. the ratio between the largest and
        smallest absolute eigenvalue of the (analytic) jacobian matrix.

        See :meth:`ODESys.stiffness` for more info.
        """
        return self.stiffness(xyp, self._get_analytic_stiffness_cb())


class TransformedSys(SymbolicSys):
    """ SymbolicSys with variable transformations.

    Parameters
    ----------
    dep_exprs : iterable of pairs
        see :class:`SymbolicSys`
    indep : Symbol
        see :class:`SymbolicSys`
    dep_transf : iterable of (expression, expression) pairs
        pairs of (forward, backward) transformations for the dependents
        variables
    indep_transf : pair of expressions
        forward and backward transformation of the independent variable
    params :
        see :class:`SymbolicSys`
    exprs_process_cb : callbable
        Post processing of the expressions (signature: ``f(exprs) -> exprs``)
        for the derivatives of the dependent variables after transformation
        have been applied.
    check_transforms : bool
        Passed as keyword argument ``check`` to :func:`.util.transform_exprs_dep` and
        :func:`.util.transform_exprs_indep`.
    \*\*kwargs :
        Keyword arguments passed onto :class:`SymbolicSys`.

    """

    def __init__(self, dep_exprs, indep=None, dep_transf=None,
                 indep_transf=None, params=(), exprs_process_cb=None,
                 check_transforms=True, **kwargs):
        dep, exprs = zip(*dep_exprs)
        if dep_transf is not None:
            self.dep_fw, self.dep_bw = zip(*dep_transf)
            exprs = transform_exprs_dep(
                self.dep_fw, self.dep_bw, list(zip(dep, exprs)), check_transforms)
        else:
            self.dep_fw, self.dep_bw = None, None

        if indep_transf is not None:
            self.indep_fw, self.indep_bw = indep_transf
            exprs = transform_exprs_indep(
                self.indep_fw, self.indep_bw, list(zip(dep, exprs)), indep, check_transforms)
        else:
            self.indep_fw, self.indep_bw = None, None

        if exprs_process_cb is not None:
            exprs = exprs_process_cb(exprs)

        pre_processors = kwargs.pop('pre_processors', [])
        post_processors = kwargs.pop('post_processors', [])
        super(TransformedSys, self).__init__(
            zip(dep, exprs), indep, params,
            pre_processors=pre_processors + [self._forward_transform_xy],
            post_processors=[self._back_transform_out] + post_processors,
            **kwargs)
        # the pre- and post-processors need callbacks:
        self.f_dep = None if self.dep_fw is None else self._callback_factory(self.dep_fw)
        self.b_dep = None if self.dep_bw is None else self._callback_factory(self.dep_bw)
        self.f_indep = None if self.indep_fw is None else self._callback_factory([self.indep_fw])
        self.b_indep = None if self.indep_bw is None else self._callback_factory([self.indep_bw])

    @classmethod
    def from_callback(cls, cb, ny=None, nparams=None, dep_transf_cbs=None,
                      indep_transf_cbs=None, **kwargs):
        """
        Create an instance from a callback.

        Analogous to :func:`SymbolicSys.from_callback`.

        Parameters
        ----------
        cb : callable
            Signature ``rhs(x, y[:], p[:]) -> f[:]``
        ny : int
            length of y
        nparams : int
            length of p
        dep_transf_cbs : iterable of pairs callables
            callables should have the signature ``f(yi) -> expression`` in yi
        indep_transf_cbs : pair of callbacks
            callables should have the signature ``f(x) -> expression`` in x
        \*\*kwargs :
            Keyword arguments passed onto :class:`TransformedSys`.

        """
        ny, nparams = _get_ny_nparams_from_kw(ny, nparams, kwargs)
        be = Backend(kwargs.pop('backend', None))
        x, = be.real_symarray('x', 1)
        y = be.real_symarray('y', ny)
        p = be.real_symarray('p', nparams)
        _y = dict(zip(kwargs['names'], y)) if kwargs.get('dep_by_name', False) else y
        _p = dict(zip(kwargs['param_names'], p)) if kwargs.get('par_by_name', False) else p
        exprs = _ensure_4args(cb)(x, _y, _p, be)
        if dep_transf_cbs is not None:
            dep_transf = [(fw(yi), bw(yi)) for (fw, bw), yi
                          in zip(dep_transf_cbs, y)]
        else:
            dep_transf = None

        if indep_transf_cbs is not None:
            indep_transf = indep_transf_cbs[0](x), indep_transf_cbs[1](x)
        else:
            indep_transf = None
        if kwargs.get('dep_by_name', False):
            exprs = [exprs[k] for k in kwargs['names']]
        return cls(list(zip(y, exprs)), x, dep_transf,
                   indep_transf, p, backend=be, **kwargs)

    def _back_transform_out(self, xout, yout, params):
        try:
            yout[0][0, 0]
        except:
            pass
        else:
            return zip(*[self._back_transform_out(_x, _y, _p) for
                         _x, _y, _p in zip(xout, yout, params)])
        xout, yout, params = map(np.asarray, (xout, yout, params))
        xbt = np.empty(xout.size)
        ybt = np.empty((xout.size, self.ny))
        for idx, (x, y) in enumerate(zip(xout.flat, yout)):
            xbt[idx] = x if self.b_indep is None else self.b_indep(x, y, params)
            ybt[idx, :] = y if self.b_dep is None else self.b_dep(x, y, params)
        return xbt, ybt, params

    def _forward_transform_xy(self, x, y, p):
        x, y, p = map(np.asarray, (x, y, p))
        if y.ndim == 1:
            return (x if self.f_indep is None else
                    np.concatenate([self.f_indep(_, y, p) for _ in x]),
                    y if self.f_dep is None else self.f_dep(x[0], y, p), p)
        elif y.ndim == 2:
            return zip(*[self._forward_transform_xy(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
        else:
            raise NotImplementedError("Don't know what to do with %d dimensions." % y.ndim)


def symmetricsys(dep_tr=None, indep_tr=None, SuperClass=TransformedSys, **kwargs):
    """ A factory function for creating symmetrically transformed systems.

    Creates a new subclass which applies the same transformation for each dependent variable.

    Parameters
    ----------
    dep_tr : pair of callables (default: None)
        Forward and backward transformation callbacks to be applied to the
        dependent variables.
    indep_tr : pair of callables (default: None)
        Forward and backward transformation to be applied to the
        independent variable.
    SuperClass : class
    \*\*kwargs :
        Default keyword arguments for the TransformedSys subclass.

    Returns
    -------
    Subclass of SuperClass (by default :class:`TransformedSys`).

    Examples
    --------
    >>> import sympy
    >>> logexp = (sympy.log, sympy.exp)
    >>> def psimp(exprs):
    ...     return [sympy.powsimp(expr.expand(), force=True) for expr in exprs]
    ...
    >>> LogLogSys = symmetricsys(logexp, logexp, exprs_process_cb=psimp)
    >>> mysys = LogLogSys.from_callback(lambda x, y, p: [-y[0], y[0] - y[1]], 2, 0)
    >>> mysys.exprs
    (-exp(x_0), -exp(x_0) + exp(x_0 + y_0 - y_1))

    """
    if dep_tr is not None:
        if not callable(dep_tr[0]) or not callable(dep_tr[1]):
            raise ValueError("Exceptected dep_tr to be a pair of callables")
    if indep_tr is not None:
        if not callable(indep_tr[0]) or not callable(indep_tr[1]):
            raise ValueError("Exceptected indep_tr to be a pair of callables")

    class _SymmetricSys(SuperClass):
        def __init__(self, dep_exprs, indep=None, **inner_kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(inner_kwargs)
            dep, exprs = zip(*dep_exprs)
            super(_SymmetricSys, self).__init__(
                zip(dep, exprs), indep,
                dep_transf=list(zip(
                    list(map(dep_tr[0], dep)),
                    list(map(dep_tr[1], dep))
                )) if dep_tr is not None else None,
                indep_transf=((indep_tr[0](indep), indep_tr[1](indep))
                              if indep_tr is not None else None),
                **new_kwargs)

        @classmethod
        def from_callback(cls, cb, ny=None, nparams=None, **inner_kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(inner_kwargs)
            return SuperClass.from_callback(
                cb, ny, nparams,
                dep_transf_cbs=repeat(dep_tr) if dep_tr is not None else None,
                indep_transf_cbs=indep_tr,
                **new_kwargs)

    return _SymmetricSys


class ScaledSys(TransformedSys):
    """ Transformed system where the variables have been scaled linearly.

    Parameters
    ----------
    dep_exprs : iterable of (symbol, expression)-pairs
        see :class:`SymbolicSys`
    indep : Symbol
        see :class:`SymbolicSys`
    dep_scaling : number (>0) or iterable of numbers
        scaling of the dependent variables (default: 1)
    indep_scaling : number (>0)
        scaling of the independent variable (default: 1)
    params :
        see :class:`SymbolicSys`
    \*\*kwargs :
        Keyword arguments passed onto :class:`TransformedSys`.

    Examples
    --------
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> scaled1 = ScaledSys([(x, x*x)], dep_scaling=1000)
    >>> scaled1.exprs
    (x**2/1000,)
    >>> scaled2 = ScaledSys([(x, x**3)], dep_scaling=1000)
    >>> scaled2.exprs
    (x**3/1000000,)

    """

    @staticmethod
    def _scale_fw_bw(scaling):
        return (lambda x: scaling*x, lambda x: x/scaling)

    def __init__(self, dep_exprs, indep=None, dep_scaling=1, indep_scaling=1,
                 params=(), **kwargs):
        dep, exprs = list(zip(*dep_exprs))
        try:
            n = len(dep_scaling)
        except TypeError:
            n = len(dep_exprs)
            dep_scaling = [dep_scaling]*n
        transf_dep_cbs = [self._scale_fw_bw(s) for s in dep_scaling]
        transf_indep_cbs = self._scale_fw_bw(indep_scaling)
        super(ScaledSys, self).__init__(
            dep_exprs, indep,
            dep_transf=[(transf_cb[0](depi),
                         transf_cb[1](depi)) for transf_cb, depi
                        in zip(transf_dep_cbs, dep)],
            indep_transf=(transf_indep_cbs[0](indep),
                          transf_indep_cbs[0](indep)) if indep is not None else None,
            **kwargs)
        if self.autonomous_interface is None:
            self.autonomous_interface = self.autonomous_exprs

    @classmethod
    def from_callback(cls, cb, ny=None, nparams=None, dep_scaling=1, indep_scaling=1,
                      **kwargs):
        """
        Create an instance from a callback.

        Analogous to :func:`SymbolicSys.from_callback`.

        Parameters
        ----------
        cb : callable
            Signature rhs(x, y[:], p[:]) -> f[:]
        ny : int
            length of y
        nparams : int
            length of p
        dep_scaling : number (>0) or iterable of numbers
            scaling of the dependent variables (default: 1)
        indep_scaling: number (>0)
            scaling of the independent variable (default: 1)
        \*\*kwargs :
            Keyword arguments passed onto :class:`ScaledSys`.

        Examples
        --------
        >>> def f(x, y, p):
        ...     return [p[0]*y[0]**2]
        >>> odesys = ScaledSys.from_callback(f, 1, 1, dep_scaling=10)
        >>> odesys.exprs
        (p_0*y_0**2/10,)

        """
        res = TransformedSys.from_callback(
            cb, ny, nparams,
            dep_transf_cbs=repeat(cls._scale_fw_bw(dep_scaling)),
            indep_transf_cbs=cls._scale_fw_bw(indep_scaling),
            **kwargs
        )
        if res.autonomous_interface is None:
            res.autonomous_interface = res.autonomous_exprs
        return res


def _skip(indices, iterable):
    return np.asarray([elem for idx, elem in enumerate(iterable) if idx not in indices])


def _append(arr, *iterables):
    if isinstance(arr, np.ndarray):
        return np.concatenate((arr,) + iterables)
    arr = arr[:]
    for iterable in iterables:
        arr += type(arr)(iterable)
    return arr


def _concat(*args):
    return np.concatenate(list(map(np.atleast_1d, args)))


class PartiallySolvedSystem(SymbolicSys):
    """ Use analytic expressions for some dependent variables

    Parameters
    ----------
    original_system : SymbolicSys
    analytic_factory : callable
        User provided callback for expressing analytic solutions to a set of
        dependent variables in ``original_system``. The callback should have
        the signature: ``my_factory(x0, y0, p0, backend) -> dict``, where the returned
        dictionary maps dependent variabels (from ``original_system.dep``)
        to new expressions in remaining variables and initial conditions.
    \*\*kwargs : dict
        Keyword arguments passed onto :class:`SymbolicSys`.

    Attributes
    ----------
    free_names : list of str
    analytic_exprs : list of expressions
    analytic_cb : callback
    original_dep : dependent variable of original system

    Examples
    --------
    >>> odesys = SymbolicSys.from_callback(
    ...     lambda x, y, p: [
    ...         -p[0]*y[0],
    ...         p[0]*y[0] - p[1]*y[1]
    ...     ], 2, 2)
    >>> dep0 = odesys.dep[0]
    >>> partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be: {
    ...         dep0: y0[0]*be.exp(-p0[0]*(odesys.indep-x0))
    ...     })
    >>> print(partsys.exprs)  # doctest: +SKIP
    (_Dummy_29*p_0*exp(-p_0*(-_Dummy_28 + x)) - p_1*y_1,)
    >>> y0, k = [3, 2], [3.5, 2.5]
    >>> xout, yout, info = partsys.integrate([0, 1], y0, k, integrator='scipy')
    >>> info['success'], yout.shape[1]
    (True, 2)

    """

    _attrs_to_copy = SymbolicSys._attrs_to_copy + ('free_names', 'free_latex_names', 'original_dep')

    def __init__(self, original_system, analytic_factory, **kwargs):
        self._ori_sys = original_system
        self.analytic_factory = _ensure_4args(analytic_factory)
        if self._ori_sys.roots is not None:
            raise NotImplementedError('roots currently unsupported')
        _be = self._ori_sys.be
        _Dummy = _be.Dummy
        self.init_indep = _Dummy('init_indep')
        self.init_dep = [_Dummy('init_%s' % (idx if self._ori_sys.names is None else self._ori_sys.names[idx]))
                         for idx in range(self._ori_sys.ny)]

        if 'pre_processors' in kwargs or 'post_processors' in kwargs:
            raise NotImplementedError("Cannot override pre-/postprocessors")
        if 'backend' in kwargs and Backend(kwargs['backend']) != _be:
            raise ValueError("Cannot mix backends.")
        _pars = self._ori_sys.params
        if self._ori_sys.par_by_name:
            _pars = dict(zip(self._ori_sys.param_names, _pars))

        self.original_dep = self._ori_sys.dep
        _dep0 = (dict(zip(self.original_dep, self.init_dep)) if self._ori_sys.dep_by_name
                 else self.init_dep)
        self.analytic_exprs = self.analytic_factory(self.init_indep, _dep0, _pars, _be)
        if len(self.analytic_exprs) == 0:
            raise ValueError("Failed to produce any analytic expressions.")
        new_dep = []
        free_names = []
        free_latex_names = []
        for idx, dep in enumerate(self.original_dep):
            if dep not in self.analytic_exprs:
                new_dep.append(dep)
                if self._ori_sys.names is not None:
                    free_names.append(self._ori_sys.names[idx])
                if self._ori_sys.latex_names is not None:
                    free_latex_names.append(self._ori_sys.latex_names[idx])
        self.free_names = None if self._ori_sys.names is None else free_names
        self.free_latex_names = None if self._ori_sys.latex_names is None else free_latex_names
        new_params = _append(self._ori_sys.params, (self.init_indep,), self.init_dep)
        self.analytic_cb = self._get_analytic_cb(
            self._ori_sys, list(self.analytic_exprs.values()), new_dep, new_params)
        self.ori_analyt_idx_map = OrderedDict([(self.original_dep.index(dep), idx)
                                               for idx, dep in enumerate(self.analytic_exprs)])
        self.ori_remaining_idx_map = {self.original_dep.index(dep): idx for
                                      idx, dep in enumerate(new_dep)}
        nanalytic = len(self.analytic_exprs)
        new_exprs = [expr.subs(self.analytic_exprs) for idx, expr in
                     enumerate(self._ori_sys.exprs) if idx not in self.ori_analyt_idx_map]
        new_kw = kwargs.copy()
        for attr in self._attrs_to_copy:
            if attr not in new_kw and getattr(self._ori_sys, attr, None) is not None:
                new_kw[attr] = getattr(self._ori_sys, attr)

        if 'lower_bounds' not in new_kw and getattr(self._ori_sys, 'lower_bounds', None) is not None:
            new_kw['lower_bounds'] = _skip(self.ori_analyt_idx_map, self._ori_sys.lower_bounds)

        if 'upper_bounds' not in new_kw and getattr(self._ori_sys, 'upper_bounds', None) is not None:
            new_kw['upper_bounds'] = _skip(self.ori_analyt_idx_map, self._ori_sys.upper_bounds)

        def partially_solved_pre_processor(x, y, p):
            x, y, p = map(np.asarray, (x, y, p))
            if y.ndim == 2:
                return zip(*[partially_solved_pre_processor(_x, _y, _p)
                             for _x, _y, _p in zip(x, y, p)])
            return (x, _skip(self.ori_analyt_idx_map, y), _append(p, [x[0]], y))

        def partially_solved_post_processor(x, y, p):
            try:
                y[0][0, 0]
            except:
                pass
            else:
                return zip(*[partially_solved_post_processor(_x, _y, _p)
                             for _x, _y, _p in zip(x, y, p)])
            new_y = np.empty(y.shape[:-1] + (y.shape[-1]+nanalytic,))
            analyt_y = self.analytic_cb(x, y, p)
            for idx in range(self._ori_sys.ny):
                if idx in self.ori_analyt_idx_map:
                    new_y[..., idx] = analyt_y[..., self.ori_analyt_idx_map[idx]]
                else:
                    new_y[..., idx] = y[..., self.ori_remaining_idx_map[idx]]
            return x, new_y, p[:-(1+self._ori_sys.ny)]

        new_kw['pre_processors'] = self._ori_sys.pre_processors + [partially_solved_pre_processor]
        new_kw['post_processors'] = [partially_solved_post_processor] + self._ori_sys.post_processors

        super(PartiallySolvedSystem, self).__init__(
            zip(new_dep, new_exprs), self._ori_sys.indep, new_params,
            backend=_be, **new_kw)

    @classmethod
    def from_linear_invariants(cls, ori_sys, preferred=None):
        """ Reformulates the ODE system in fewer variables.

        Given linear invariant equations one can always reduce the number
        of dependent variables in the system by the rank of the matrix describing
        this linear system.

        Parameters
        ----------
        ori_sys : :class:`SymbolicSys` instance
        preferred : iterable of preferred dependent variables
            Due to numerical rounding it is preferable to choose the variables
            which are expected to be of the largest magnitude during integration.
        """
        _be = ori_sys.be
        A = _be.Matrix(ori_sys.linear_invariants)
        rA, pivots = A.rref()
        if len(pivots) < A.shape[0]:
            # If the linear system contains rows which a linearly dependent these could be removed.
            # The criterion for removal could be dictated by a user provided callback.
            #
            # An alternative would be to write the matrix in reduced row echelon form, however,
            # this would cause the invariants to become linear combinations of each other and
            # their intuitive meaning (original principles they were formulated from) will be lost.
            # Hence that is not the default behaviour. However, the user may choose to rewrite the
            # equations in reduced row echelon form if they choose to before calling this method.
            raise NotImplementedError("Linear invariants contain linear dependencies.")
        per_row_cols = [(ri, [ci for ci in range(A.cols) if A[ri, ci] != 0]) for ri in range(A.rows)]
        if preferred is None:
            preferred = ori_sys.names[:A.rows] if ori_sys.dep_by_name else list(range(A.rows))
        targets = [
            ori_sys.names.index(dep) if ori_sys.dep_by_name else (
                dep if isinstance(dep, int) else ori_sys.dep.index(dep))
            for dep in preferred]
        row_tgt = []
        for ri, colids in sorted(per_row_cols, key=lambda k: len(k[1])):
            for tgt in targets:
                if tgt in colids:
                    row_tgt.append((ri, tgt))
                    targets.remove(tgt)
                    break
            if len(targets) == 0:
                break
        else:
            raise ValueError("Could not find a solutions for: %s" % targets)

        def analytic_factory(x0, y0, p0, be):
            return {
                ori_sys.dep[tgt]: y0[ori_sys.dep[tgt] if ori_sys.dep_by_name else tgt] - sum(
                    [A[ri, ci]*(ori_sys.dep[ci] - y0[ori_sys.dep[ci] if ori_sys.dep_by_name else ci]) for
                     ci in range(A.cols) if ci != tgt])/A[ri, tgt] for ri, tgt in row_tgt
            }

        new_lin_invar = [row for ri, row in enumerate(A.tolist()) if ri not in list(zip(*row_tgt))[0]]
        return cls(ori_sys, analytic_factory, linear_invariants=new_lin_invar or None)

    @staticmethod
    def _get_analytic_cb(ori_sys, analytic_exprs, new_dep, new_params):
        args = _concat(ori_sys.indep, new_dep, new_params)
        return _Wrapper(ori_sys.be.Lambdify(args, analytic_exprs), len(new_dep))

    def __getitem__(self, key):
        ori_dep = self.original_dep[self.names.index(key)]
        return self.analytic_exprs.get(ori_dep, ori_dep)


def get_logexp(a=1, b=0, backend=None):
    """ Utility function for use with :func:symmetricsys.

    Creates a pair of callbacks for logarithmic transformation
    (including scaling and shifting): ``u = ln(a*x + b)``.

    Parameters
    ----------
    a : number
        Scaling.
    b : number
        Shift.

    Returns
    -------
    Pair of callbacks.

    """
    if backend is None:
        import sympy as backend
    return (lambda x: backend.log(a*x + b),
            lambda x: (backend.exp(x) - b)/a)
