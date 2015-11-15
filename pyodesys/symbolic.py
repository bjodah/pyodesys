# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import chain, repeat

import numpy as np
import sympy as sp

from .core import OdeSys
from .util import (
    banded_jacobian, transform_exprs_dep,
    transform_exprs_indep, ensure_3args
)


class SymbolicSys(OdeSys):
    """
    ODE System from symbolic expressions

    Creates a :class:OdeSys from symbolic expressions

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
    roots: iterable of expressions
        equations to look for root's for during integration
        (currently available through cvode)
    lambdify: callback
        default: :py:func:`sympy.lambdify`
    lambdify_unpack: bool (default: True)
        whether or not unpacking of args needed when calling lambdify callback
    Matrix: class
        default: :py:class:`sympy.Matrix`
    \*\*kwargs:
        See :py:class:`OdeSys`

    Attributes
    ----------
    dep : iterable of symbols
        dependent variables
    indep : symbol
        independent variable
    params : iterable of symbols
        parameters

    Notes
    -----
    Works for a moderate number of unknowns, :py:func:`sympy.lambdify` has
    an upper limit on number of arguments.
    """

    def __init__(self, dep_exprs, indep=None, params=(), jac=True, dfdx=True,
                 roots=None, lambdify=None, lambdify_unpack=True, Matrix=None,
                 **kwargs):
        self.dep, self.exprs = zip(*dep_exprs)
        self.indep = indep
        self.params = params
        self._jac = jac
        self._dfdx = dfdx
        self.roots = roots
        if lambdify is not None:
            self.lambdify = lambdify
        self.lambdify_unpack = lambdify_unpack
        if Matrix is None:
            import sympy
            self.Matrix = sympy.ImmutableMatrix
        else:
            self.Matrix = Matrix
        # we need self.band before super().__init__
        self.band = kwargs.get('band', None)
        if kwargs.get('names', None) is True:
            kwargs['names'] = [y.name for y in self.dep]
        super(SymbolicSys, self).__init__(
            self.get_f_ty_callback(),
            self.get_j_ty_callback(),
            self.get_dfdx_callback(),
            self.get_roots_callback(),
            nroots=None if roots is None else len(roots),
            **kwargs)

    @staticmethod
    def Symbol(name):
        """ Create a new Symbol (uses :py:class:`sympy.Symol`) """
        return sp.Symbol(name, real=True)

    @staticmethod
    def Dummy():
        """ Create a new Dummy (symbol) (uses :py:class:`sympy.Dummy`) """
        return sp.Dummy()

    @classmethod
    def symarray(cls, prefix, shape, Symbol=None):
        """ Creates an nd-array of symbols

        Parameters
        ----------
        prefix: str
        shape: tuple
        Symbol: callable
            (defualt :func:`Symbol`)
        """
        # see https://github.com/sympy/sympy/pull/9939
        # when released: return sp.symarray(key, n, real=True)
        arr = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            arr[index] = (Symbol or cls.Symbol)('%s_%s' % (
                prefix, '_'.join(map(str, index))))
        return arr

    @staticmethod
    def lambdify(*args, **kwargs):
        """ Creates a callback for numerical evaluation of symbolic expreesion

        Uses :py:func:`sympy.lambdify`
        """        
        if 'modules' not in kwargs:
            kwargs['modules'] = [{'ImmutableMatrix': np.array}, 'numpy']
        return sp.lambdify(*args, **kwargs)

    @classmethod
    def from_callback(cls, cb, ny, nparams=0, *args, **kwargs):
        """ Create an instance from a callback.

        Parameters
        ----------
        cb: callbable
            Signature rhs(x, y[:], p[:]) -> f[:]
        ny: int
            length of y
        nparams: int (default: 0)
            length of p
        \*args:
            arguments passed onto :class:`SymbolicSys`
        \*\*kwargs:
            keyword arguments passed onto :class:`SymbolicSys`

        Returns
        -------
        An instance of :class:`SymbolicSys`.
        """
        x = cls.Symbol('x')
        y = cls.symarray('y', ny)
        p = cls.symarray('p', nparams)
        exprs = ensure_3args(cb)(x, y, p)
        return cls(zip(y, exprs), x, p, *args, **kwargs)

    @property
    def ny(self):
        """ Number of dependent variables in the system. """
        return len(self.exprs)

    def _args(self, x=None, y=None, params=()):
        if x is None:
            x = self.indep
        if y is None:
            y = self.dep
        args = tuple(y)
        if self.indep is not None:
            args = (x,) + args
        return args + tuple(params)

    def get_jac(self):
        """ Derive the jacobian from ``self.exprs`` and ``self.dep``. """
        if self._jac is True:
            if self.band is None:
                f = self.Matrix(1, self.ny, lambda _, q: self.exprs[q])
                self._jac = f.jacobian(self.dep)
            else:
                # Banded
                self._jac = self.Matrix(banded_jacobian(
                    self.exprs, self.dep, *self.band))
        elif self._jac is False:
            return False

        return self._jac

    def get_dfdx(self):
        """ Derive the derivatives of ``self.exprs`` with respect to ``self.indep``. """
        if self._dfdx is True:
            if self.indep is None:
                self._dfdx = [0]*self.ny
            else:
                self._dfdx = [expr.diff(self.indep) for expr in self.exprs]
        elif self._dfdx is False:
            return False
        return self._dfdx

    def get_f_ty_callback(self):
        """ Generate a callback for evaluating ``self.exprs``. """
        cb = self.lambdify(list(chain(self._args(), self.params)), self.exprs)

        def f(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self._args(x, y, params)))
            else:
                return np.asarray(cb(self._args(x, y, params)))
        return f

    def get_j_ty_callback(self):
        """ Generate a callback for evaluating the jacobian. """
        j_exprs = self.get_jac()
        if j_exprs is False:
            return None
        cb = self.lambdify(list(chain(self._args(), self.params)), j_exprs)

        def j(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self._args(x, y, params)))
            else:
                return np.asarray(cb(self._args(x, y, params)))
        return j

    def get_dfdx_callback(self):
        """ Generate a callback for evaluating derivative of ``self.exprs`` """
        dfdx_exprs = self.get_dfdx()
        if dfdx_exprs is False:
            return None
        cb = self.lambdify(list(chain(self._args(), self.params)), dfdx_exprs)

        def dfdx(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self._args(x, y, params)))
            else:
                return np.asarray(cb(self._args(x, y, params)))
        return dfdx

    def get_roots_callback(self):
        """ Generate a callback for evaluating ``self.roots`` """
        if self.roots is None:
            return None
        cb = self.lambdify(list(chain(self._args(), self.params)), self.roots)

        def roots(x, y, params=()):
            if self.lambdify_unpack:
                return np.asarray(cb(*self._args(x, y, params)))
            else:
                return np.asarray(cb(self._args(x, y, params)))
        return roots

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

        cb = odefun(lambda x, y: rhs, xout[0], y0)
        yout = []
        for x in xout:
            yout.append(cb(x))
        info = {'nrhs': rhs.ncall}
        return self.post_process(
            xout, yout, self.internal_params)[:2] + (info,)


class TransformedSys(SymbolicSys):
    """ SymbolicSys with abstracted variable transformations.
    
    Parameters
    ----------
    dep_exprs: see :class:`SymbolicSys`
    indep: see :class:`SymbolicSys`
    dep_transf: iterable of (expression, expression) pairs
        pairs of (forward, backward) transformations for the dependents
        variables
    indep_transf: pair of expressions
        forward and backward transformation of the independent variable
    params: see :class:`SymbolicSys`
    exprs_process_cb: callbable
        signatrue f(exprs) -> exprs
        post processing of the expressions for the derivatives of the
        dependent variables after transformation have been applied.
    \*\*kwargs:
        keyword arguments passed onto :class:`SymbolicSys`
    """

    def __init__(self, dep_exprs, indep=None, dep_transf=None,
                 indep_transf=None, params=(), exprs_process_cb=None,
                 **kwargs):
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

        if exprs_process_cb is not None:
            exprs = exprs_process_cb(exprs)

        super(TransformedSys, self).__init__(
            zip(dep, exprs), indep, params,
            pre_processors=[self._forward_transform_xy],
            post_processors=[self._back_transform_out], **kwargs)
        # the pre- and post-processors need callbacks:
        args = self._args(indep, dep, params)
        self.f_dep = self.lambdify(args, self.dep_fw)
        self.b_dep = self.lambdify(args, self.dep_bw)
        if (self.indep_fw, self.indep_bw) != (None, None):
            self.f_indep = self.lambdify(args, self.indep_fw)
            self.b_indep = self.lambdify(args, self.indep_bw)
        else:
            self.f_indep = None
            self.b_indep = None

    @classmethod
    def from_callback(cls, cb, ny, nparams=0, dep_transf_cbs=None,
                      indep_transf_cbs=None, **kwargs):
        """
        Create an instance from a callback.

        Analogous to :func:`SymbolicSys.from_callback`.

        Parameters
        ----------
        cb: callable
            Signature rhs(x, y[:], p[:]) -> f[:]
        ny: int
            length of y
        nparams: int
            length of p
        dep_transf_cbs: iterable of pairs callables
            callables should have the signature f(yi) -> expression in yi
        indep_transf_cbs: pair of callbacks
            callables should have the signature f(x) -> expression in x
        \*\*kwargs:
            keyword arguments passed onto :class:`TransformedSys`
        """
        x = cls.Symbol('x')
        y = cls.symarray('y', ny)
        p = cls.symarray('p', nparams)
        exprs = ensure_3args(cb)(x, y, p)
        if dep_transf_cbs is not None:
            dep_transf = [(fw(yi), bw(yi)) for (fw, bw), yi
                          in zip(dep_transf_cbs, y)]
        else:
            dep_transf = None

        if indep_transf_cbs is not None:
            indep_transf = indep_transf_cbs[0](x), indep_transf_cbs[1](x)
        else:
            indep_transf = None

        return cls(list(zip(y, exprs)), x, dep_transf,
                   indep_transf, p, **kwargs)

    def _back_transform_out(self, xout, yout, params):
        args = self._args(xout, yout.T, params)
        if self.lambdify_unpack:
            return (xout if self.b_indep is None else self.b_indep(*args),
                    np.array(self.b_dep(*args)).T, params)
        else:
            return (xout if self.b_indep is None else self.b_indep(args),
                    np.array(self.b_dep(args)).T, params)

    def _forward_transform_xy(self, x, y, p):
        args = self._args(x, y, p)
        if self.lambdify_unpack:
            return (x if self.f_indep is None else self.f_indep(*args),
                    self.f_dep(*args), p)
        else:
            return (x if self.f_indep is None else self.f_indep(args),
                    self.f_dep(args), p)


def symmetricsys(dep_tr=None, indep_tr=None, **kwargs):
    """
    A factory function for creating symmetrically transformed systems.

    Parameters
    ----------
    dep_tr: pair of callables (default: None)
        Forward and backward transformation to be applied to the
        dependent variables
    indep_tr: pair of callables (default: None)
        Forward and backward transformation to be applied to the
        independent variable
    \*\*kwargs:
        default keyword arguments for the TransformedSys subclass

    Returns
    -------
    Subclass of TransformedSys

    Examples
    --------
    >>> import sympy
    >>> logexp = (sympy.log, sympy.exp)
    >>> LogLogSys = symmetricsys(
    ...     logexp, logexp, exprs_process_cb=lambda exprs: [
    ...         sympy.powsimp(expr.expand(), force=True) for expr in exprs])

    """
    if dep_tr is not None:
        if not callable(dep_tr[0]) or not callable(dep_tr[1]):
            raise ValueError("Exceptected dep_tr to be a pair of callables")
    if indep_tr is not None:
        if not callable(indep_tr[0]) or not callable(indep_tr[1]):
            raise ValueError("Exceptected indep_tr to be a pair of callables")

    class _Sys(TransformedSys):
        def __init__(self, dep_exprs, indep=None, **inner_kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(inner_kwargs)
            dep, exprs = zip(*dep_exprs)
            super(_Sys, self).__init__(
                zip(dep, exprs), indep,
                dep_transf=list(zip(
                    list(map(dep_tr[0], dep)),
                    list(map(dep_tr[1], dep))
                )) if dep_tr is not None else None,
                indep_transf=((indep_tr[0](indep), indep_tr[1](indep))
                              if indep_tr is not None else None),
                **new_kwargs)

        @classmethod
        def from_callback(cls, cb, n, nparams=0, **inner_kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(inner_kwargs)
            return TransformedSys.from_callback(
                cb, n, nparams,
                dep_transf_cbs=repeat(dep_tr) if dep_tr is not None else None,
                indep_transf_cbs=indep_tr,
                **new_kwargs)
    return _Sys


class ScaledSys(TransformedSys):
    """
    Special case of TransformedSys where the variables have been scaled.

    Parameters
    ----------
    dep_exprs: see :class:`SymbolicSys`
    indep: see :class:`SymbolicSys`
    dep_scaling: number (>0) or iterable of numbers
        scaling of the dependent variables
    indep_scaling: number (>0)
        scaling of the independent variable
    params: see :class:`SymbolicSys`
    \*\*kwargs:
        keyword arguments passed onto TransformedSys

    """

    @staticmethod
    def scale_fw_bw(scaling):
        return (lambda x: scaling*x, lambda x: x/scaling)

    def __init__(self, dep_exprs, indep=None, dep_scaling=1, indep_scaling=1,
                 params=(), **kwargs):
        dep, exprs = zip(*dep_exprs)
        try:
            n = len(dep_scaling)
        except TypeError:
            n = len(dep_exprs)
            dep_scaling = [dep_scaling]*n
        transf_dep_cbs = [self.scale_fw_bw(s) for s in dep_scaling]
        transf_indep_cbs = self.scale_fw_bw(indep_scaling)
        super(ScaledSys, self).__init__(
            dep_exprs, indep,
            dep_transf=[(transf_cb[0](depi),
                         transf_cb[1](depi)) for transf_cb, depi
                        in zip(transf_dep_cbs, dep)],
            indep_transf=(transf_indep_cbs[0](indep),
                          transf_indep_cbs[0](indep)) if indep is not None else
            None, **kwargs)

    @classmethod
    def from_callback(cls, cb, n, nparams=0, dep_scaling=1, indep_scaling=1,
                      **kwargs):
        return TransformedSys.from_callback(
            cb, n, nparams,
            dep_transf_cbs=repeat(cls.scale_fw_bw(dep_scaling)),
            indep_transf_cbs=cls.scale_fw_bw(indep_scaling),
        )


def _take(indices, iterable):
    return np.asarray([elem for idx, elem in enumerate(
        iterable) if idx in indices])


def _skip(indices, iterable):
    return np.asarray([elem for idx, elem in enumerate(
        iterable) if idx not in indices])


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
    """ Use analytic expressions some dependent variables

    Parameters
    ----------
    original_system: SymbolicSys
    analytic_factory: callable
        signature: solved(x0, y0, p0) -> dict, where dict maps
        independent variables as analytic expressions in remaining variables

    Examples
    --------
    >>> odesys = SymbolicSys.from_callback(
    ...     lambda x, y, p: [
    ...         -p[0]*y[0],
    ...         p[0]*y[0] - p[1]*y[1]
    ...     ], 2, 2)
    >>> dep0 = odesys.dep[0]
    >>> partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0: {
    ...         dep0: y0[0]*sp.exp(-p0[0]*(odesys.indep-x0))
    ...     })
    >>> print(partsys.exprs)  # doctest: +SKIP
    (_Dummy_29*p_0*exp(-p_0*(-_Dummy_28 + x)) - p_1*y_1,)
    >>> y0, k = [3, 2], [3.5, 2.5]
    >>> xout, yout, info = partsys.integrate('scipy', [0, 1], y0, k)
    >>> info['success'], yout.shape[1]
    (True, 2)

    """

    def __init__(self, original_system, analytic_factory, Dummy=None,
                 **kwargs):
        self.original_system = original_system
        self.analytic_factory = analytic_factory
        if original_system.roots is not None:
            raise NotImplementedError('roots unsupported')
        if Dummy is None:
            Dummy = self.Dummy
        self.init_indep = Dummy()
        self.init_dep = [Dummy() for _ in range(original_system.ny)]

        if 'pre_processors' in kwargs or 'post_processors' in kwargs:
            raise NotImplementedError("Cannot override pre-/postprocessors")
        analytic = self.analytic_factory(self.init_indep, self.init_dep,
                                         original_system.params)
        new_dep = [dep for dep in original_system.dep if dep not in analytic]
        new_params = _append(original_system.params, (self.init_indep,),
                             self.init_dep)
        self.analytic_cb = self._get_analytic_cb(
            original_system, list(analytic.values()), new_params)
        analytic_ids = [original_system.dep.index(dep) for dep in analytic]
        nanalytic = len(analytic_ids)
        new_exprs = [expr.subs(analytic) for idx, expr in enumerate(
            original_system.exprs) if idx not in analytic_ids]
        new_kw = kwargs.copy()
        if 'name' not in new_kw and original_system.names is not None:
            new_kw['names'] = original_system.names
        if 'band' not in new_kw and original_system.band is not None:
            new_kw['band'] = original_system.band

        def pre_processor(x, y, p):
            return (x, _skip(analytic_ids, y), _append(
                p, [x[0]], y))

        def post_processor(x, y, p):
            new_y = np.empty(y.shape[:-1] + (y.shape[-1]+nanalytic,))
            analyt_y = self.analytic_cb(x, p)
            analyt_idx = 0
            intern_idx = 0
            for idx in range(original_system.ny):
                if idx in analytic_ids:
                    new_y[..., idx] = analyt_y[analyt_idx]
                    analyt_idx += 1
                else:
                    new_y[..., idx] = y[..., intern_idx]
                    intern_idx += 1
            return x, new_y, p[:-(1+original_system.ny)]

        new_kw['pre_processors'] = [
            pre_processor] + original_system.pre_processors
        new_kw['post_processors'] = original_system.post_processors + [
            post_processor]

        super(PartiallySolvedSystem, self).__init__(
            zip(new_dep, new_exprs), original_system.indep, new_params,
            lambdify=original_system.lambdify,
            lambdify_unpack=original_system.lambdify_unpack,
            Matrix=original_system.Matrix, **new_kw)

    def _get_analytic_cb(self, ori_sys, analytic_exprs, new_params):
        cb = ori_sys.lambdify(_concat(ori_sys.indep, new_params),
                              analytic_exprs)

        def analytic(x, params):
            args = np.empty((len(x), 1+len(params)))
            args[:, 0] = x
            args[:, 1:] = params
            if ori_sys.lambdify_unpack:
                return np.asarray(cb(*(args.T)))
            else:
                return np.asarray(cb(args.T))
        return analytic
