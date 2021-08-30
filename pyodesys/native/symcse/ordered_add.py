"""Utilities used in the package."""

from sympy import Expr, UnevaluatedExpr
from sympy.printing.precedence import PRECEDENCE, precedence
from sympy.printing.c import C99CodePrinter
from sympy.printing.cxx import CXX17CodePrinter


def ordered_add(*args):
    if len(args) == 0:
        return 0
    elif (len(args)) == 1:
        return args[0]
    a, b, *rest = args
    ue = UnevaluatedExpr(a+b)
    if len(rest) == 0:
        return ue
    else:
        return ordered_add(ue, *rest)
