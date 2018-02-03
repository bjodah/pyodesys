# -*- coding: utf-8 -*-
"""
This module contains a subclass of ODESys which allows the user to generate
auxiliary expressions from a canonical set of symbolic expressions. Subclasses
are also provided for dealing with variable transformations and partially
solved systems.
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from itertools import repeat, chain
import warnings

import numpy as np

from .util import import_
from .core import ODESys, RecoverableError
from .util import (
    transform_exprs_dep, transform_exprs_indep, _ensure_4args, _Callback
)

Backend = import_('sym', 'Backend')


def _get_indep_name(names):
    if 'x' not in names:
        indep_name = 'x'
    else:
        i = 0
        indep_name = 'indep0'
        while indep_name in names:
            i += 1
            indep_name = 'indep%d' % i
    return indep_name


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


def _get_lin_invar_mtx(lin_invar, be, ny, names=None):
    if lin_invar is None or lin_invar == []:
        return None
    else:
        if isinstance(lin_invar[0], dict) and names:
            lin_invar = [[d[n] for n in names] for d in lin_invar]
        li_mtx = be.Matrix(lin_invar)
        if len(li_mtx.shape) != 2 or li_mtx.shape[1] != ny:
            raise ValueError("Incorrect shape of linear_invariants Matrix: %s" % str(li_mtx.shape))
        return li_mtx


def _is_autonomous(indep, exprs):
    """ Whether the expressions for the dependent variables are autonomous.

    Note that the system may still behave as an autonomous system on the interface
    of :meth:`integrate` due to use of pre-/post-processors.
    """
    if indep is None:
        return True
    for expr in exprs:
        try:
            in_there = indep in expr.free_symbols
        except:
            in_there = expr.has(indep)
        if in_there:
            return False
    return True


def _skip(indices, iterable, as_array=True):
    result = [elem for idx, elem in enumerate(iterable) if idx not in indices]
    return np.asarray(result) if as_array else result


def _reinsert(indices, arr, new):
    trail_dim = arr.shape[-1]+len(indices)
    new_arr = np.empty(arr.shape[:-1] + (trail_dim,))
    idx_arr, idx_insert = 0, 0
    for glbl in range(trail_dim):
        if glbl in indices:
            new_arr[..., glbl] = new[..., idx_insert]
            idx_insert += 1
        else:
            new_arr[..., glbl] = arr[..., idx_arr]
            idx_arr += 1
    return new_arr


class SymbolicSys(ODESys):
    """ ODE System from symbolic expressions

    Creates a :class:`ODESys` instance
    from symbolic expressions. Jacobian and second derivatives
    are derived when needed.

    Parameters
    ----------
    dep_exprs : iterable of (symbol, expression)-pairs
    indep : Symbol
        Independent variable (default: None => autonomous system).
    params : iterable of symbols
        Problem parameters. If ``None``: zero parameters assumed (violation of this will
        raise a ValueError), If ``True``: params are deduced from (sorted) free_symbols.
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
    steady_state_root : bool or float
        Generate an expressions for roots which is the sum the smaller of
        aboslute values of derivatives or relative derivatives subtracted by
        the value of ``steady_state_root`` (default ``1e-10``).
    init_indep : Symbol,  ``True`` or ``None``
        When ``True`` construct using ``be.Symbol``. See also :attr:`init_indep`.
    init_dep : tuple of Symbols, ``True`` or ``None``
        When ``True`` construct using ``be.Symbol``. See also :attr:`init_dep`.
    clip_to_bounds : bool
        When bounds are given, ``f_cb`` and ``j_cb`` will be clipped to be within
        bounds prior to evaluation.
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
    backend : module
        Symbolic backend, e.g. 'sympy', 'symengine'.
    ny : int
        ``len(self.dep)`` note that this is not neccessarily the expected length of
        ``y0`` in the case of e.g. :class:`PartiallySolvedSystem`. i.e. ``ny`` refers
        to the number of dependent variables after pre processors have been applied.
    init_indep : Symbol,  ``None``
        Symbol for initial value of independent variable (before pre processors).
    init_dep : tuple of Symbols or ``None``
        Symbols for initial values of dependent variables (before pre processors).

    """

    _attrs_to_copy = ('first_step_expr', 'names', 'param_names', 'dep_by_name', 'par_by_name',
                      'latex_names', 'latex_param_names', 'description',
                      'linear_invariants', 'linear_invariant_names',
                      'nonlinear_invariants', 'nonlinear_invariant_names',
                      'to_arrays_callbacks', '_indep_autonomous_key',
                      'taken_names', 'numpy')
    append_iv = True

    @property
    def linear_invariants(self):
        return getattr(self, '_linear_invariants', None)

    @linear_invariants.setter
    def linear_invariants(self, lin_invar):
        self._linear_invariants = _get_lin_invar_mtx(lin_invar, self.be, self.ny, self.names)

    def __init__(self, dep_exprs, indep=None, params=None, jac=True, dfdx=True, first_step_expr=None,
                 roots=None, backend=None, lower_bounds=None, upper_bounds=None,
                 linear_invariants=None, nonlinear_invariants=None,
                 linear_invariant_names=None, nonlinear_invariant_names=None, steady_state_root=False,
                 init_indep=None, init_dep=None, clip_to_bounds=False, **kwargs):
        self.dep, self.exprs = zip(*dep_exprs.items()) if isinstance(dep_exprs, dict) else zip(*dep_exprs)
        self.indep = indep
        if params is True or params is None:
            all_free = tuple(filter(lambda x: x not in self.dep + (self.indep,),
                                    sorted(set.union(*[expr.free_symbols for expr in self.exprs]), key=str)))
            if params is None and all_free:
                raise ValueError("Pass params explicitly or pass True to have them deduced.")
            params = all_free

        self.params = tuple(params)
        self._jac = jac
        self._dfdx = dfdx
        self.first_step_expr = first_step_expr
        self.be = Backend(backend)
        if steady_state_root:
            if steady_state_root is True:
                steady_state_root = 1e-10
            if roots is not None:
                raise ValueError("Cannot give both steady_state_root & roots")
            roots = [sum([self.be.Min(self.be.Abs(expr),
                                      self.be.Abs(expr)/dep) for dep, expr in
                          zip(self.dep, self.exprs)]) - steady_state_root]
        self.roots = roots

        if init_indep is True:
            init_indep = self._mk_init_indep(name=self.indep)
        if init_dep is True:
            init_dep = self._mk_init_dep(names=kwargs.get('names'))
        self.init_indep = init_indep
        self.init_dep = init_dep
        if self.init_indep is not None or self.init_dep is not None:
            if self.init_indep is None or self.init_dep is None:
                raise ValueError("Need both or neither of init_indep & init_dep.")
            if kwargs.get('append_iv', True) is not True:
                raise ValueError("append_iv == False is not valid when giving init_indep/init_dep.")
            kwargs['append_iv'] = True
        _names = kwargs.get('names', None)
        if _names is True:
            kwargs['names'] = _names = [y.name for y in self.dep]
        if self.indep is not None and _names not in (None, ()):
            if self.indep.name in _names:
                raise ValueError("Independent variable cannot share name with any dependent variable")

        _param_names = kwargs.get('param_names', None)
        if _param_names is True:
            kwargs['param_names'] = [p.name for p in self.params]

        self.band = kwargs.get('band', None)  # needed by get_j_ty_callback
        # bounds needed by get_f_ty_callback:
        self.lower_bounds = None if lower_bounds is None else np.array(lower_bounds)*np.ones(self.ny)
        self.upper_bounds = None if upper_bounds is None else np.array(upper_bounds)*np.ones(self.ny)

        super(SymbolicSys, self).__init__(
            self.get_f_ty_callback(),
            self.get_j_ty_callback(),
            self.get_dfdx_callback(),
            self.get_first_step_callback(),
            self.get_roots_callback(),
            nroots=None if roots is None else len(roots),
            autonomous_exprs=_is_autonomous(self.indep, self.exprs),
            **kwargs)

        self.linear_invariants = linear_invariants
        self.nonlinear_invariants = nonlinear_invariants
        self.linear_invariant_names = linear_invariant_names or None
        if self.linear_invariant_names is not None:
            if len(self.linear_invariant_names) != self.linear_invariants.shape[0]:
                raise ValueError("Incorrect length of linear_invariant_names: %d (expected %d)" % (
                    len(self.linear_invariant_names), linear_invariants.shape[0]))
        self.nonlinear_invariant_names = nonlinear_invariant_names or None
        if self.nonlinear_invariant_names is not None:
            if len(self.nonlinear_invariant_names) != len(nonlinear_invariants):
                raise ValueError("Incorrect length of nonlinear_invariant_names: %d (expected %d)" % (
                    len(self.nonlinear_invariant_names), len(nonlinear_invariants)))

        if self.autonomous_interface is None:
            self.autonomous_interface = self.autonomous_exprs

    def _Symbol(self, name, be=None):
        be = be or self.be
        try:
            return be.Symbol(name, real=True)
        except TypeError:
            return be.Symbol(name)

    def _mk_init_indep(self, name, be=None, prefix='i_', suffix=''):
        name = name or 'indep'
        be = be or self.be
        name = prefix + str(name) + suffix
        if getattr(self, 'indep', None) is not None:
            if self.indep.name == name:
                raise ValueError("Name ambiguity in independent variable name")
        return self._Symbol(name, be)

    def _mk_init_dep(self, names=None, be=None, ny=None, prefix='i_', suffix=''):
        be = be or self.be
        ny = ny or self.ny
        names = names or getattr(self, 'names', [str(i) for i in range(ny)])
        if getattr(self, 'dep', None) is not None:
            for dep in self.dep:
                if dep.name.startswith(prefix):
                    raise ValueError("Name ambiguity in dependent variable names")
        use_names = names is not None and len(names) > 0
        return tuple(self._Symbol(prefix + names[idx] if use_names else str(idx) + suffix, be)
                     for idx in range(ny))

    def all_invariants(self, linear_invariants=None, nonlinear_invariants=None, dep=None, backend=None):
        linear_invariants = linear_invariants or getattr(self, 'linear_invariants', None)
        return (([] if linear_invariants is None else (linear_invariants * (backend or self.be).Matrix(
            len(dep or self.dep), 1, dep or self.dep
        )).T.tolist()[0]) + (nonlinear_invariants or getattr(self, 'nonlinear_invariants', []) or []))

    def all_invariant_names(self):
        return (self.linear_invariant_names or []) + (self.nonlinear_invariant_names or [])

    def __getitem__(self, key):
        return self.dep[self.names.index(key)]

    @staticmethod
    def _to_array(cont, by_name, names, keys):
        if isinstance(cont, dict) and (not by_name or names is None or len(names) == 0):
            cont = [cont[k] for k in keys]
        return cont

    def to_arrays(self, x, y, p, **kwargs):
        y = self._to_array(y, self.dep_by_name, self.names, self.dep)
        p = self._to_array(p, self.par_by_name, self.param_names, self.params)
        return super(SymbolicSys, self).to_arrays(x, y, p, **kwargs)

    @staticmethod
    def _kwargs_roots_from_roots_cb(roots_cb, kwargs, x, _y, _p, be):
        if roots_cb is not None:
            if 'roots' in kwargs:
                raise ValueError("Keyword argument ``roots`` already given.")

            try:
                roots = roots_cb(x, _y, _p, be)
            except TypeError:
                roots = _ensure_4args(roots_cb)(x, _y, _p, be)

            kwargs['roots'] = roots

    @staticmethod
    def _kwargs_bounds_from_bounds_cb(bounds_key, kwargs, x, _y, _p, be):
        bounds_cb = kwargs.pop(bounds_key + '_cb', None)
        if bounds_cb is not None:
            if bounds_key in kwargs:
                raise ValueError("Keyword argument ``%s`` already given." % bounds_key)
            kwargs[bounds_key] = bounds_cb(x, _y, _p, be)

    @classmethod
    def from_callback(cls, rhs, ny=None, nparams=None, first_step_factory=None, roots_cb=None,
                      indep_name=None, lower_bounds_cb=None, upper_bounds_cb=None, **kwargs):
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
        roots_cb : callable
            Callback with signature ``roots(x, y[:], p[:], backend=math) -> r[:]``.
        indep_name : str
            Default 'x' if not already in ``names``, otherwise indep0, or indep1, or ...
        dep_by_name : bool
            Make ``y`` passed to ``rhs`` a dict (keys from :attr:`names`) and convert
            its return value from dict to array.
        par_by_name : bool
            Make ``p`` passed to ``rhs`` a dict (keys from :attr:`param_names`).
        lower_bounds_cb : callable
            Same signature as ``rhs``.
        upper_bounds_cb : callable
            Same signature as ``rhs``.
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
        names = tuple(kwargs.pop('names', ''))
        indep_name = indep_name or _get_indep_name(names)
        try:
            x = be.Symbol(indep_name, real=True)
        except TypeError:
            x = be.Symbol(indep_name)
        y = be.real_symarray('y', ny)
        p = be.real_symarray('p', nparams)
        _y = dict(zip(names, y)) if kwargs.get('dep_by_name', False) else y
        _p = dict(zip(kwargs['param_names'], p)) if kwargs.get('par_by_name', False) else p

        try:
            exprs = rhs(x, _y, _p, be)
        except TypeError:
            exprs = _ensure_4args(rhs)(x, _y, _p, be)

        try:
            if len(exprs) != ny:
                raise ValueError("Callback returned unexpected (%d) number of expressions: %d" % (ny, len(exprs)))
        except TypeError:
            raise ValueError("Callback did not return an array_like of expressions: %s" % str(exprs))

        cls._kwargs_roots_from_roots_cb(roots_cb, kwargs, x, _y, _p, be)
        cls._kwargs_bounds_from_bounds_cb('lower_bounds', kwargs, x, _y, _p, be)
        cls._kwargs_bounds_from_bounds_cb('upper_bounds', kwargs, x, _y, _p, be)

        if first_step_factory is not None:
            if 'first_step_exprs' in kwargs:
                raise ValueError("Cannot override first_step_exprs.")
            try:
                kwargs['first_step_expr'] = first_step_factory(x, _y, _p, be)
            except TypeError:
                kwargs['first_step_expr'] = _ensure_4args(first_step_factory)(x, _y, _p, be)
        if kwargs.get('dep_by_name', False):
            exprs = [exprs[k] for k in names]
        return cls(zip(y, exprs), x, kwargs.pop('params', None) if len(p) == 0 else p,
                   backend=be, names=names, **kwargs)

    @classmethod
    def from_other(cls, ori, **kwargs):
        """ Creates a new instance with an existing one as a template.

        Parameters
        ----------
        ori : SymbolicSys instance
        \\*\\*kwargs:
            Keyword arguments used to create the new instance.

        Returns
        -------
        A new instance of the class.

        """
        for k in cls._attrs_to_copy + ('params', 'roots', 'init_indep', 'init_dep'):
            if k not in kwargs:
                val = getattr(ori, k)
                if val is not None:
                    kwargs[k] = val
        if 'lower_bounds' not in kwargs and getattr(ori, 'lower_bounds') is not None:
            kwargs['lower_bounds'] = ori.lower_bounds
        if 'upper_bounds' not in kwargs and getattr(ori, 'upper_bounds') is not None:
            kwargs['upper_bounds'] = ori.upper_bounds

        if len(ori.pre_processors) > 0:
            if 'pre_processors' not in kwargs:
                kwargs['pre_processors'] = []
            kwargs['pre_processors'] = kwargs['pre_processors'] + ori.pre_processors

        if len(ori.post_processors) > 0:
            if 'post_processors' not in kwargs:
                kwargs['post_processors'] = []
            kwargs['post_processors'] = ori.post_processors + kwargs['post_processors']
        if 'dep_exprs' not in kwargs:
            kwargs['dep_exprs'] = zip(ori.dep, ori.exprs)
        if 'indep' not in kwargs:
            kwargs['indep'] = ori.indep

        instance = cls(**kwargs)
        for attr in ori._attrs_to_copy:
            if attr not in cls._attrs_to_copy:
                setattr(instance, attr, getattr(ori, attr))
        return instance

    @classmethod
    def from_other_new_params(cls, ori, par_subs, new_pars, new_par_names=None,
                              new_latex_par_names=None, **kwargs):
        """ Creates a new instance with an existing one as a template (with new parameters)

        Calls ``.from_other`` but first it replaces some parameters according to ``par_subs``
        and (optionally) introduces new parameters given in ``new_pars``.

        Parameters
        ----------
        ori : SymbolicSys instance
        par_subs : dict
            Dictionary with substitutions (mapping symbols to new expressions) for parameters.
            Parameters appearing in this instance will be omitted in the new instance.
        new_pars : iterable (optional)
            Iterable of symbols for new parameters.
        new_par_names : iterable of str
            Names of the new parameters given in ``new_pars``.
        new_latex_par_names : iterable of str
            TeX formatted names of the new parameters given in ``new_pars``.
        \\*\\*kwargs:
            Keyword arguments passed to ``.from_other``.

        Returns
        -------
        Intance of the class
        extra : dict with keys:
            - recalc_params : ``f(t, y, p1) -> p0``

        """
        new_exprs = [expr.subs(par_subs) for expr in ori.exprs]
        drop_idxs = [ori.params.index(par) for par in par_subs]
        params = _skip(drop_idxs, ori.params, False) + list(new_pars)
        back_substitute = _Callback(ori.indep, ori.dep, params, list(par_subs.values()),
                                    ori.be.Lambdify)

        def recalc_params(t, y, p):
            rev = back_substitute(t, y, p)
            return _reinsert(drop_idxs, np.repeat(np.atleast_2d(p), rev.shape[0], axis=0),
                             rev)[..., :len(ori.params)]

        return cls.from_other(
            ori, dep_exprs=zip(ori.dep, new_exprs),
            params=params,
            param_names=_skip(drop_idxs, ori.param_names, False) + list(new_par_names or []),
            latex_param_names=_skip(drop_idxs, ori.latex_param_names, False) + list(new_latex_par_names or []),
            **kwargs
        ), {'recalc_params': recalc_params}

    @classmethod
    def from_other_new_params_by_name(cls, ori, par_subs, new_par_names=(), **kwargs):
        """ Creates a new instance with an existing one as a template (with new parameters)

        Calls ``.from_other_new_params`` but first it creates the new instances from user provided
        callbacks generating the expressions the parameter substitutions.

        Parameters
        ----------
        ori : SymbolicSys instance
        par_subs : dict mapping str to ``f(t, y{}, p{}) -> expr``
            User provided callbacks for parameter names in ``ori``.
        new_par_names : iterable of str
        \\*\\*kwargs:
            Keyword arguments passed to ``.from_other_new_params``.

        """
        if not ori.dep_by_name:
            warnings.warn('dep_by_name is not True')
        if not ori.par_by_name:
            warnings.warn('par_by_name is not True')
        dep = dict(zip(ori.names, ori.dep))
        new_pars = ori.be.real_symarray(
            'p', len(ori.params) + len(new_par_names))[len(ori.params):]
        par = dict(chain(zip(ori.param_names, ori.params), zip(new_par_names, new_pars)))
        par_symb_subs = OrderedDict([(ori.params[ori.param_names.index(pk)], cb(
            ori.indep, dep, par, backend=ori.be)) for pk, cb in par_subs.items()])
        return cls.from_other_new_params(
            ori, par_symb_subs, new_pars, new_par_names=new_par_names, **kwargs)

    @property
    def ny(self):
        """ Number of dependent variables in the system. """
        return len(self.exprs)

    def as_autonomous(self, new_indep_name=None, new_latex_indep_name=None):
        if self.autonomous_exprs:
            return self
        old_indep_name = self.indep_name or _get_indep_name(self.names)
        new_names = () if not self.names else (self.names + (old_indep_name,))
        new_indep_name = new_indep_name or _get_indep_name(new_names)
        new_latex_indep_name = new_latex_indep_name or new_indep_name
        new_latex_names = () if not self.latex_names else (
            self.latex_names + (new_latex_indep_name,))
        new_indep = self.be.Symbol(new_indep_name)
        new_dep = self.dep + (self.indep,)
        new_exprs = self.exprs + (self.indep**0,)
        new_kw = dict(
            names=new_names,
            indep_name=new_indep_name,
            latex_names=new_latex_names,
            latex_indep_name=new_latex_indep_name,
            autonomous_interface=False,  # see pre-processor below
        )
        if new_names:
            new_kw['taken_names'] = self.taken_names + (old_indep_name,)
        if self.linear_invariants:
            new_kw['linear_invariants'] = np.concatenate(
                (self.linear_invariants, np.zeros((self.linear_invariants.shape[0], 1))), axis=-1)
        for attr in filter(lambda k: k not in new_kw, self._attrs_to_copy):
            new_kw[attr] = getattr(self, attr)

        def autonomous_post_processor(x, y, p):
            try:
                y[0][0, 0]
            except:
                pass
            else:
                return zip(*[autonomous_post_processor(_x, _y, _p)
                             for _x, _y, _p in zip(x, y, p)])
            # one could check here that x and y[..., -1] do not differ too much
            return y[..., -1], y[..., :-1], p

        new_kw['post_processors'] = [autonomous_post_processor] + self.post_processors

        new_kw['_indep_autonomous_key'] = new_names[-1] if new_names else True
        return self.__class__(zip(new_dep, new_exprs), indep=new_indep, params=self.params, **new_kw)

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
        if jac_in_cses.nullspace():
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
        return _Callback(self.indep, self.dep, self.params, exprs, self.be.Lambdify)

    def _mk_bounds_wrapper(self, cb):
        lb = self.lower_bounds
        ub = self.upper_bounds
        if lb is None and ub is None:
            return cb

        def wrapper(t, y, p=(), be=None):
            if lb is not None:
                if not self.clip_to_bounds and np.any(y < lb - 10*self._current_integration_kwargs['atol']):
                    raise RecoverableError
                y = np.array(y)
                y[y < lb] = lb[y < lb]
            if ub is not None:
                if not self.clip_to_bounds and np.any(y > ub + 10*self._current_integration_kwargs['atol']):
                    raise RecoverableError
                y = np.array(y)
                y[y > ub] = ub[y > ub]
            return cb(t, y, p, be)
        return wrapper

    def get_f_ty_callback(self):
        """ Generates a callback for evaluating ``self.exprs``. """
        return self._mk_bounds_wrapper(self._callback_factory(self.exprs))

    def get_j_ty_callback(self):
        """ Generates a callback for evaluating the jacobian. """
        j_exprs = self.get_jac()
        if j_exprs is False:
            return None
        return self._mk_bounds_wrapper(self._callback_factory(j_exprs))

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


def _group_invariants(all_invar, deps, be, names=None):
    linear_invar = []
    nonlinear_invar = []
    lin_names, nonlin_names = [], []
    use_names = names is not None and len(names) > 0
    for idx, invar in enumerate(all_invar):
        derivs = [invar.diff(dep) for dep in deps]
        if all([deriv.is_Number for deriv in derivs]):
            linear_invar.append(derivs)
            if use_names:
                lin_names.append(names[idx])
        else:
            nonlinear_invar.append(invar)
            if use_names:
                nonlin_names.append(names[idx])
    if names is None:
        return linear_invar, nonlinear_invar
    else:
        return linear_invar, nonlinear_invar, lin_names, nonlin_names


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
        roots = kwargs.pop('roots', None)
        be = Backend(kwargs.pop('backend', None))
        kwargs['backend'] = be
        all_invariants = self.all_invariants(
            _get_lin_invar_mtx(kwargs.pop('linear_invariants', None), be, len(dep)),
            kwargs.pop('nonlinear_invariants', None), dep, backend=be)
        if dep_transf is not None:
            self.dep_fw, self.dep_bw = zip(*dep_transf)
            exprs = transform_exprs_dep(
                self.dep_fw, self.dep_bw, list(zip(dep, exprs)), check_transforms)
            bw_subs = list(zip(dep, self.dep_bw))
            all_invariants = [invar.subs(bw_subs) for invar in all_invariants]
            if roots is not None:
                roots = [r.subs(bw_subs) for r in roots]
        else:
            self.dep_fw, self.dep_bw = None, None

        if indep_transf is not None:
            self.indep_fw, self.indep_bw = indep_transf
            exprs = transform_exprs_indep(
                self.indep_fw, self.indep_bw, list(zip(dep, exprs)), indep, check_transforms)
            all_invariants = [invar.subs(indep, self.indep_bw) for invar in all_invariants]
            if roots is not None:
                roots = [r.subs(indep, self.indep_bw) for r in roots]
        else:
            self.indep_fw, self.indep_bw = None, None

        kwargs['linear_invariants'], kwargs['nonlinear_invariants'], \
            kwargs['linear_invariant_names'], kwargs['nonlinear_invariant_names'] = _group_invariants(
                all_invariants, dep, be, (list(kwargs.get('linear_invariant_names') or ()) +
                                          list(kwargs.get('nonlinear_invariant_names') or ())))

        lower_b = kwargs.pop('lower_bounds', None)
        if lower_b is not None:
            lower_b *= np.ones(len(dep))
        upper_b = kwargs.pop('upper_bounds', None)
        if upper_b is not None:
            upper_b *= np.ones(len(dep))
        pre_processors = kwargs.pop('pre_processors', [])
        post_processors = kwargs.pop('post_processors', [])
        super(TransformedSys, self).__init__(
            zip(dep, exprs_process_cb(exprs) if exprs_process_cb is not None else exprs), indep, params, roots=roots,
            pre_processors=pre_processors + [self._forward_transform_xy],
            post_processors=[self._back_transform_out] + post_processors,
            **kwargs)
        # the pre- and post-processors need callbacks:
        self.f_dep = None if self.dep_fw is None else self._callback_factory(self.dep_fw)
        self.b_dep = None if self.dep_bw is None else self._callback_factory(self.dep_bw)
        self.f_indep = None if self.indep_fw is None else self._callback_factory([self.indep_fw])
        self.b_indep = None if self.indep_bw is None else self._callback_factory([self.indep_bw])
        _x, _p = float('nan'), [float('nan')]*len(self.params)
        self.lower_bounds = lower_b if self.f_dep is None or lower_b is None else self.f_dep(_x, lower_b, _p)
        self.upper_bounds = upper_b if self.f_dep is None or upper_b is None else self.f_dep(_x, upper_b, _p)

    @classmethod
    def from_callback(cls, cb, ny=None, nparams=None, dep_transf_cbs=None,
                      indep_transf_cbs=None, roots_cb=None, **kwargs):
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
        roots_cb : callable
            Callback with signature ``roots(x, y[:], p[:], backend=math) -> r[:]``.
            Callback should return untransformed roots.
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

        cls._kwargs_roots_from_roots_cb(roots_cb, kwargs, x, _y, _p, be)
        cls._kwargs_bounds_from_bounds_cb('lower_bounds', kwargs, x, _y, _p, be)
        cls._kwargs_bounds_from_bounds_cb('upper_bounds', kwargs, x, _y, _p, be)

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
        x = xout if self.b_indep is None else self.b_indep(xout, yout, params).squeeze(axis=-1)
        y = yout if self.b_dep is None else self.b_dep(xout, yout, params)
        return x, y, params

    def _forward_transform_xy(self, x, y, p):
        x, y, p = map(np.asarray, (x, y, p))
        if y.ndim == 1:
            _x = x if self.f_indep is None else self.f_indep(x, y[..., None, :], p[..., None, :])[..., 0]
            _y = y if self.f_dep is None else self.f_dep(x[..., 0], y, p)
            return _x, _y, p
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
    >>> scaled1.exprs == (x**2/1000,)
    True
    >>> scaled2 = ScaledSys([(x, x**3)], dep_scaling=1000)
    >>> scaled2.exprs == (x**3/1000000,)
    True

    """

    @staticmethod
    def _scale_fw_bw(scaling):
        return (lambda x: scaling*x, lambda x: x/scaling)

    def __init__(self, dep_exprs, indep=None, dep_scaling=1, indep_scaling=1,
                 params=(), **kwargs):
        dep_exprs = list(dep_exprs)
        dep, exprs = list(zip(*dep_exprs))
        try:
            n = len(dep_scaling)
        except TypeError:
            n = len(dep_exprs)
            dep_scaling = [dep_scaling]*n
        transf_dep_cbs = [self._scale_fw_bw(s) for s in dep_scaling]
        transf_indep_cbs = self._scale_fw_bw(indep_scaling)
        super(ScaledSys, self).__init__(
            dep_exprs, indep, params=params,
            dep_transf=[(transf_cb[0](depi),
                         transf_cb[1](depi)) for transf_cb, depi
                        in zip(transf_dep_cbs, dep)],
            indep_transf=(transf_indep_cbs[0](indep),
                          transf_indep_cbs[1](indep)) if indep is not None else None,
            **kwargs)

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
        return TransformedSys.from_callback(
            cb, ny, nparams,
            dep_transf_cbs=repeat(cls._scale_fw_bw(dep_scaling)),
            indep_transf_cbs=cls._scale_fw_bw(indep_scaling),
            **kwargs
        )


def _append(arr, *iterables):
    if isinstance(arr, np.ndarray):
        return np.concatenate((arr,) + iterables)
    arr = arr[:]
    for iterable in iterables:
        arr += type(arr)(iterable)
    return arr


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
        roots = kwargs.pop('roots', self._ori_sys.roots)
        _be = self._ori_sys.be
        init_indep = self._ori_sys.init_indep or self._mk_init_indep(name=self._ori_sys.indep, be=self._ori_sys.be)
        init_dep = self._ori_sys.init_dep or self._ori_sys._mk_init_dep()
        if 'pre_processors' in kwargs or 'post_processors' in kwargs:
            raise NotImplementedError("Cannot override pre-/postprocessors")
        if 'backend' in kwargs and Backend(kwargs['backend']) != _be:
            raise ValueError("Cannot mix backends.")
        _pars = self._ori_sys.params
        if self._ori_sys.par_by_name:
            _pars = dict(zip(self._ori_sys.param_names, _pars))

        self.original_dep = self._ori_sys.dep
        _dep0 = (dict(zip(self.original_dep, init_dep)) if self._ori_sys.dep_by_name
                 else init_dep)
        self.analytic_exprs = self.analytic_factory(init_indep, _dep0, _pars, _be)
        if len(self.analytic_exprs) == 0:
            raise ValueError("Failed to produce any analytic expressions.")
        new_dep = []
        free_names = []
        free_latex_names = []
        for idx, dep in enumerate(self.original_dep):
            if dep not in self.analytic_exprs:
                new_dep.append(dep)
                if self._ori_sys.names is not None and len(self._ori_sys.names) > 0:
                    free_names.append(self._ori_sys.names[idx])
                if self._ori_sys.latex_names is not None and len(self._ori_sys.latex_names) > 0:
                    free_latex_names.append(self._ori_sys.latex_names[idx])
        self.free_names = None if self._ori_sys.names is None else free_names
        self.free_latex_names = None if self._ori_sys.latex_names is None else free_latex_names
        self.append_iv = kwargs.get('append_iv', False)
        new_pars = _append(self._ori_sys.params, (init_indep,), init_dep)
        self.analytic_cb = self._get_analytic_callback(
            self._ori_sys, list(self.analytic_exprs.values()), new_dep, new_pars)
        self.ori_analyt_idx_map = OrderedDict([(self.original_dep.index(dep), idx)
                                               for idx, dep in enumerate(self.analytic_exprs)])
        self.ori_remaining_idx_map = {self.original_dep.index(dep): idx for
                                      idx, dep in enumerate(new_dep)}
        new_exprs = [expr.subs(self.analytic_exprs) for idx, expr in
                     enumerate(self._ori_sys.exprs) if idx not in self.ori_analyt_idx_map]
        new_roots = None if roots is None else [expr.subs(self.analytic_exprs) for expr in roots]
        new_kw = kwargs.copy()
        for attr in self._attrs_to_copy:
            if attr not in new_kw and getattr(self._ori_sys, attr, None) is not None:
                new_kw[attr] = getattr(self._ori_sys, attr)

        if 'lower_bounds' not in new_kw and getattr(self._ori_sys, 'lower_bounds', None) is not None:
            new_kw['lower_bounds'] = _skip(self.ori_analyt_idx_map, self._ori_sys.lower_bounds)

        if 'upper_bounds' not in new_kw and getattr(self._ori_sys, 'upper_bounds', None) is not None:
            new_kw['upper_bounds'] = _skip(self.ori_analyt_idx_map, self._ori_sys.upper_bounds)

        if kwargs.get('linear_invariants', None) is None:
            if new_kw.get('linear_invariants', None) is not None:
                if new_kw['linear_invariants'].shape[1] != self._ori_sys.ny:
                    raise ValueError("Unexpected number of columns in original linear_invariants.")
                new_kw['linear_invariants'] = new_kw['linear_invariants'][:, [i for i in range(self._ori_sys.ny)
                                                                              if i not in self.ori_analyt_idx_map]]

        def partially_solved_pre_processor(x, y, p):
            # if isinstance(y, dict) and not self.dep_by_name:
            #     y = [y[k] for k in self._ori_sys.dep]
            # if isinstance(p, dict) and not self.par_by_name:
            #     p = [p[k] for k in self._ori_sys.params]
            # x, y, p = map(np.atleast_1d, (x, y, p))
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
            new_y = np.empty(y.shape[:-1] + (y.shape[-1]+len(self.analytic_exprs),))
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
            zip(new_dep, new_exprs), self._ori_sys.indep, new_pars, backend=_be, roots=new_roots,
            init_indep=init_indep, init_dep=init_dep, **new_kw)

    @classmethod
    def from_linear_invariants(cls, ori_sys, preferred=None, **kwargs):
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
        \*\*kwargs :
            Keyword arguments passed on to constructor.
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

        ori_li_nms = ori_sys.linear_invariant_names or ()
        new_lin_invar = [[cell for ci, cell in enumerate(row) if ci not in list(zip(*row_tgt))[1]]
                         for ri, row in enumerate(A.tolist()) if ri not in list(zip(*row_tgt))[0]]
        new_lin_i_nms = [nam for ri, nam in enumerate(ori_li_nms) if ri not in list(zip(*row_tgt))[0]]
        return cls(ori_sys, analytic_factory, linear_invariants=new_lin_invar,
                   linear_invariant_names=new_lin_i_nms, **kwargs)

    @staticmethod
    def _get_analytic_callback(ori_sys, analytic_exprs, new_dep, new_params):
        return _Callback(ori_sys.indep, new_dep, new_params, analytic_exprs, ori_sys.be.Lambdify)

    def __getitem__(self, key):
        ori_dep = self.original_dep[self.names.index(key)]
        return self.analytic_exprs.get(ori_dep, ori_dep)

    def integrate(self, *args, **kwargs):
        if 'atol' in kwargs:
            atol = kwargs.pop('atol')
            if isinstance(atol, dict):
                atol = [atol[k] for k in self.free_names]
            else:
                try:
                    len(atol)
                except TypeError:
                    pass
                else:
                    atol = [atol[idx] for idx in _skip(self.ori_analyt_idx_map, atol)]
            kwargs['atol'] = atol
        return super(PartiallySolvedSystem, self).integrate(*args, **kwargs)


def get_logexp(a=1, b=0, a2=None, b2=None, backend=None):
    """ Utility function for use with :func:symmetricsys.

    Creates a pair of callbacks for logarithmic transformation
    (including scaling and shifting): ``u = ln(a*x + b)``.

    Parameters
    ----------
    a : number
        Scaling (forward).
    b : number
        Shift (forward).
    a2 : number
        Scaling (backward).
    b2 : number
        Shift (backward).

    Returns
    -------
    Pair of callbacks.

    """
    if a2 is None:
        a2 = a
    if b2 is None:
        b2 = b
    if backend is None:
        import sympy as backend
    return (lambda x: backend.log(a*x + b),
            lambda x: (backend.exp(x) - b2)/a2)
