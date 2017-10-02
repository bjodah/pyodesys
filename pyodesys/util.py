# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from functools import reduce
import inspect
import math
import operator

from pkg_resources import parse_requirements, parse_version

import numpy as np


def stack_1d_on_left(x, y):
    """ Stack a 1D array on the left side of a 2D array

    Parameters
    ----------
    x: 1D array
    y: 2D array
        Requirement: ``shape[0] == x.size``
    """
    _x = np.atleast_1d(x)
    _y = np.atleast_1d(y)
    return np.hstack((_x.reshape(_x.size, 1), _y))


def check_transforms(fw, bw, symbs):
    """ Verify validity of a pair of forward and backward transformations

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    symbs: iterable of symbols
        the variables that are transformed
    """
    for f, b, y in zip(fw, bw, symbs):
        if f.subs(y, b) - y != 0:
            raise ValueError('Incorrect (did you set real=True?) fw: %s'
                             % str(f))
        if b.subs(y, f) - y != 0:
            raise ValueError('Incorrect (did you set real=True?) bw: %s'
                             % str(b))


def transform_exprs_dep(fw, bw, dep_exprs, check=True):
    """ Transform y[:] in dydx

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    dep_exprs: iterable of (symbol, expression) pairs
        pairs of (dependent variable, derivative expressions),
        i.e. (y, dydx) pairs
    check: bool (default: True)
        whether to verification of the analytic correctness should
        be performed

    Returns
    -------
    List of transformed expressions for dydx

    """
    if len(fw) != len(dep_exprs) or \
       len(fw) != len(bw):
        raise ValueError("Incompatible lengths")
    dep, exprs = zip(*dep_exprs)
    if check:
        check_transforms(fw, bw, dep)
    bw_subs = list(zip(dep, bw))
    return [(e*f.diff(y)).subs(bw_subs) for f, y, e in zip(fw, dep, exprs)]


def transform_exprs_indep(fw, bw, dep_exprs, indep, check=True):
    """ Transform x in dydx

    Parameters
    ----------
    fw: expression
        forward transformation
    bw: expression
        backward transformation
    dep_exprs: iterable of (symbol, expression) pairs
        pairs of (dependent variable, derivative expressions)
    check: bool (default: True)
        whether to verification of the analytic correctness should
        be performed

    Returns
    -------
    List of transformed expressions for dydx
    """
    if check:
        if fw.subs(indep, bw) - indep != 0:
            fmtstr = 'Incorrect (did you set real=True?) fw: %s'
            raise ValueError(fmtstr % str(fw))
        if bw.subs(indep, fw) - indep != 0:
            fmtstr = 'Incorrect (did you set real=True?) bw: %s'
            raise ValueError(fmtstr % str(bw))
    dep, exprs = zip(*dep_exprs)
    return [(e/fw.diff(indep)).subs(indep, bw) for e in exprs]


class _Blessed(object):
    pass


def _ensure_4args(func):
    """ Conditionally wrap function to ensure 4 input arguments

    Parameters
    ----------
    func: callable
        with two, three or four positional arguments

    Returns
    -------
    callable which possibly ignores 0, 1 or 2 positional arguments

    """
    if func is None:
        return None
    if isinstance(func, _Blessed):  # inspect on __call__ is a hassle...
        return func

    self_arg = 1 if inspect.ismethod(func) else 0
    if len(inspect.getargspec(func)[0]) == 4 + self_arg:
        return func
    if len(inspect.getargspec(func)[0]) == 3 + self_arg:
        return lambda x, y, p=(), backend=math: func(x, y, p)
    elif len(inspect.getargspec(func)[0]) == 2 + self_arg:
        return lambda x, y, p=(), backend=math: func(x, y)
    else:
        raise ValueError("Incorrect numer of arguments")


def _default(arg, default):
    return default if arg is None else arg


def _concat(*args):
    return np.concatenate(list(map(np.atleast_1d, args)))


class _Callback(_Blessed):

    def __init__(self, indep, dep, params, exprs, Lambdify=None):
        self.indep, self.dep, self.params = indep, dep, params
        if indep is None:
            self.args = _concat(self.dep, self.params)
        else:
            self.args = _concat(self.indep, self.dep, self.params)
        self.input_width = len(self.args)
        self.exprs = exprs
        self.callback = Lambdify(self.args, self.exprs)
        self.ny = len(dep)
        self.take_params = len(params)

    def __call__(self, x, y, params=(), backend=None):
        _x = np.asarray(x)
        _y = np.asarray(y)
        _p = np.asarray(params)[..., :self.take_params]
        if _y.shape[-1] != self.ny:
            raise TypeError("Incorrect shape of y")
        inp = np.empty(_x.shape + (self.input_width,))
        if self.indep is None:
            nx = 0
        else:
            inp[..., 0] = _x = np.asarray(x)
            nx = 1
        inp[..., nx:(nx+self.ny)] = _y
        inp[..., (nx+self.ny):] = _p
        result = self.callback(inp)
        return result


class requires(object):
    """ Conditional skipping (on requirements) of tests in pytest

    Examples
    --------
    >>> @requires('numpy', 'scipy')
    ... def test_sqrt():
    ...     import numpy as np
    ...     assert np.sqrt(4) == 2
    ...     from scipy.special import zeta
    ...     assert zeta(2) < 2
    ...
    >>> @requires('numpy>=1.9.0')
    ... def test_nanmedian():
    ...     import numpy as np
    ...     a = np.array([[10.0, 7, 4], [3, 2, 1]])
    ...     a[0, 1] = np.nan
    ...     assert np.nanmedian(a) == 3
    ...

    """
    from operator import lt, le, eq, ne, ge, gt
    _relop = dict(zip('< <= == != >= >'.split(), [getattr(operator, attr) for attr in
                                                  'lt le eq ne ge gt'.split()]))

    def __init__(self, *reqs):
        self.missing = []
        self.incomp = []
        self.requirements = list(parse_requirements(reqs))
        for req in self.requirements:
            try:
                mod = __import__(req.project_name)
            except ImportError:
                self.missing.append(req.project_name)
            else:
                try:
                    ver = parse_version(mod.__version__)
                except AttributeError:
                    pass
                else:
                    for rel, vstr in req.specs:
                        if not self._relop[rel](ver, parse_version(vstr)):
                            self.incomp.append(str(req))

    def __call__(self, cb):
        import pytest
        r = 'Unfulfilled requirements.'
        if self.missing:
            r += " Missing modules: %s." % ', '.join(self.missing)
        if self.incomp:
            r += " Incomp versions: %s." % ', '.join(self.incomp)
        return pytest.mark.skipif(self.missing or self.incomp, reason=r)(cb)


class MissingImport(object):

    def __init__(self, modname, exc):
        self._modname = modname
        self._exc = exc

    def __getattribute__(self, attr):
        if attr in ('_modname', '_exc'):
            return object.__getattribute__(self, attr)
        else:
            raise self._exc  # ImportError("Failed to import %s" % self._modname)

    def __call__(self, *args, **kwargs):
        raise self._exc  # ImportError("Failed to import %s" % self._modname)


def import_(modname, *args):
    if len(args) == 0:
        try:
            return __import__(modname)
        except ImportError as e:
            return MissingImport(modname, e)

    mods = []
    for arg in args:
        try:
            mod = __import__(modname, globals(), locals(), [arg])
        except ImportError as e:
            mods.append(MissingImport(modname + '.' + arg, e))
        else:
            try:
                attr = getattr(mod, arg)
            except AttributeError as e:
                mods.append(MissingImport(modname + '.' + arg, e))
            else:
                mods.append(attr)
    return mods if len(args) > 1 else mods[0]


def merge_dicts(*dicts):
    """ Merges dictionaries with incresing priority.

    Parameters
    ----------
    \\*dicts: dictionaries

    Examples
    --------
    >>> d1, d2 = {'a': 1, 'b': 2}, {'a': 2, 'c': 3}
    >>> merge_dicts(d1, d2, {'a': 3}) == {'a': 3, 'b': 2, 'c': 3}
    True
    >>> d1 == {'a': 1, 'b': 2}
    True
    >>> from collections import defaultdict
    >>> dd1 = defaultdict(lambda: 3, {'b': 4})
    >>> dd2 = merge_dicts(dd1, {'c': 5}, {'c': 17})
    >>> dd2['c'] - dd2['a'] - dd2['b'] == 10
    True

    Returns
    -------
    dict

    """
    return reduce(lambda x, y: x.update(y) or x, (dicts[0].copy(),) + dicts[1:])
