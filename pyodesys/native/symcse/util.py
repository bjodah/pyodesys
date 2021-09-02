"""Utilities used in the package."""
import os
import sys
from functools import reduce
from operator import mul, add

import sympy
from sympy.printing.c import C99CodePrinter
from sympy.printing.cxx import CXX17CodePrinter
from sympy.printing.pycode import PythonCodePrinter

try:
    import symengine as se
except ModuleNotFoundError:
    se = None
else:
    from symengine.lib.symengine_wrapper import sympy2symengine


def partn(it, pred):
    """Partition an iterable into False/True groups based on predicate."""
    result = ([], [])
    for elem in it:
        result[pred(elem)].append(elem)
    return result


def prod(it):
    return reduce(mul, it)


def map_multi(funcs, its):
    return (f(it) for f, it in zip(funcs, its))


def idty(x):
    """Identity operattion (idempotent)."""
    return x


class UnevaluatedRealPropagatingExpr(sympy.UnevaluatedExpr):
    """Propagate .is_real, but nothings else from wrapped expression."""

    def _eval_is_real(self):
        return self.args[0].is_real


class _UnevaluatedExprPrinterMixin:
    def _print_UnevaluatedExpr(self, arg):
        return "(%s)" % super()._print_UnevaluatedExpr(arg)

    def _print_Integer(self, arg):
        if abs(arg) > 2**53:
            return self._print(sympy.Float(arg))
        else:
            return super()._print_Integer(arg)

    @staticmethod
    def _replace_re(arg):
        if isinstance(arg, sympy.UnevaluatedExpr) and arg.args[0].is_real:
            return UnevaluatedRealPropagatingExpr(arg.args[0])
        else:
            return sympy.re(arg)

    def doprint(self, x):
        result = super().doprint(x.replace(sympy.re, self._replace_re))
        return result


class CXXPrinter(_UnevaluatedExprPrinterMixin, CXX17CodePrinter):
    """Patched C++-printer (https://github.com/sympy/sympy/issues/21955)."""


class CPrinter(_UnevaluatedExprPrinterMixin, C99CodePrinter):
    """Patched C-printer (https://github.com/sympy/sympy/issues/21955)."""


class PyPrinter(_UnevaluatedExprPrinterMixin, PythonCodePrinter):
    """Patched Python-printer (https://github.com/sympy/sympy/issues/21955)."""


def _cse_symengine(exprs, *, se2sympy, ignore=(), symbols=None, **kwargs):
    exprs = list(exprs)
    se_exprs = [sympy2symengine(e, raise_error=True) for e in exprs]
    repl, red = se.cse(se_exprs, **kwargs)
    repl = [(se2sympy(lhs), se2sympy(rhs)) for lhs, rhs in repl]
    red = [se2sympy(e) for e in red]
    if ignore:
        ignore = [se2sympy(sympy2symengine(ig)) for ig in ignore]

        def has_ig(e):
            return any(e.has(ig) for ig in ignore)

        def is_only_ig(e):
            return all(s in ignore for s in e.free_symbols)

        keep, reintro = {}, {}
        for lhs, rhs in repl:
            rem = rhs.xreplace(reintro)  # remaining
            if not has_ig(rem):
                for k, v in keep.items():
                    if v == rem:
                        reintro[lhs] = k
                        break
                else:
                    keep[lhs] = rem
                continue

            candidates = []
            for part in sympy.preorder_traversal(rem):
                if not has_ig(part):
                    candidates.append(part)
                elif part.is_Add:
                    good, bad = partn(part.args, has_ig)
                    if good:
                        candidates.append(sum(good))
                elif part.is_Mul:
                    good, bad = partn(part.args, has_ig)
                    if good:
                        candidates.append(prod(good))
            if candidates:
                part = sorted(candidates, key=sympy.count_ops)[-1]
                if part.count_ops() == 0 or len(part.free_symbols) == 0:
                    reintro[lhs] = rem
                else:
                    for k, v in keep.items():
                        if part == k or part == v:
                            reintro[lhs] = rem
                            break
                    else:
                        keep[lhs] = part
                        reintro[lhs] = rem.subs(part, lhs)
            else:
                reintro[lhs] = rem
        repl = list(keep.items())
        red = [e.xreplace(reintro) for e in red]
    recreated = [e.subs(reversed(repl)) for e in red]
    assert recreated == exprs
    if symbols is not None:
        remap = {}
        updated = []
        for lhs, rhs in repl:
            remap[lhs] = next(symbols)
            updated.append((remap[lhs], rhs.xreplace(remap)))
        repl = updated
        red = [e.xreplace(remap) for e in red]
    return repl, red


class Backend:
    """Allow optional use of SymEngine."""

    def __init__(self, use_symengine=None, kw_cse=None, assume_real=True):
        """Initialize a symbolic backend.

        Parameters
        ----------
        use_symengine: bool
        kw_cse: keywords to use in CSE

        """
        if use_symengine is None:
            _req_backend = os.environ.get("SYMCSE_BACKEND", "").lower()
            if _req_backend == "symengine":
                use_symengine = True
            elif _req_backend == "sympy":
                use_symengine = False
            elif _req_backend == "":
                use_symengine = False  # se is not None
            else:
                raise ValueError("Unknown SYMCXSE_BACKEND: %s" % _req_backend)
        if use_symengine and se is None:
            raise ValueError("symengine missing (pip install symengine)")
        self.use_symengine = use_symengine
        self.kw_cse = kw_cse
        self.assume_real = assume_real


    def Symbol(self, name):
        if self.use_symengine:
            return se.Symbol(name)  # https://github.com/symengine/symengine.py/issues/286
        else:
            return sympy.Symbol(name, real=self.assume_real)

    def sympy2se(self, x):
        if hasattr(x, '_sympy_'):
            return x  # looks like that's already a SymEngine object
        return sympy2symengine(x, raise_error=True)

    def se2sympy(self, x):
        if not hasattr(x, '_sympy_'):
            assert isinstance(x, sympy.Basic)
            return x  # looks like that's already a SymPy object
        tmp = x._sympy_()
        return tmp.xreplace({s: sympy.Symbol(s.name, real=self.assume_real) for s in tmp.free_symbols})

    def cse(self, exprs, **kwargs):
        """Perform common sub-expression elimination."""
        exprs = list(exprs)
        new_kw = dict(**(self.kw_cse or {}), **kwargs)
        if self.use_symengine:
            return _cse_symengine(exprs, se2sympy=self.se2sympy, **new_kw)
        else:
            return sympy.cse(exprs, **new_kw)

    def _elems_colmaj(self, mtx):
        if not mtx:
            return []
        result = reduce(add, mtx.T.tolist())
        if self.use_symengine:
            return [self.se2sympy(e) for e in result]
        else:
            return result

    def _col_vec(self, v):
        args = len(v), 1, tuple(v)
        if self.use_symengine:
            return se.Matrix(*args)
        else:
            return sympy.Matrix(*args)

    def jacobian(self, exprs, wrt):
        args = len(exprs), 1, exprs
        if self.use_symengine:
            return se.Matrix(*args).jacobian(self._col_vec(wrt))
        else:
            return sympy.Matrix(*args).jacobian(wrt)

    def _LU(self, mat):
        if self.use_symengine:
            _L, _U = self.sympy2se(mat).LU()
            L, U = map(self.se2sympy, [_L, _U])
        else:
            L, U, piv = mat.LUdecomposition()
            if piv:
                raise NotImplementedError("TODO")
        for i in range(L.cols):
            for j in range(i, L.cols):
                L[i, j] = U[i, j]
        return L

    def matmul(self, A, b):
        if self.use_symengine:
            return A.mul_matrix(b)
        else:
            return A @ b

    def lambdify(self, args, exprs):
        if self.use_symengine:
            return se.Lambdify(args, exprs)
        else:
            return sympy.lambdify(args, exprs)

class BackendWithDisabledCSE(Backend):

    def cse(self, exprs, **kwargs):
        return [], exprs


def ccode(arg, **kwargs):
    p = CPrinter(settings=(kwargs or dict(math_macros={})))
    return p.doprint(arg)


def cxxcode(arg, **kwargs):
    p = CXXPrinter(settings=(kwargs or dict(math_macros={})))
    return p.doprint(arg)


def pycode(arg):
    p = PyPrinter()
    return p.doprint(arg)
