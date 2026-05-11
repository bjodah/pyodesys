"""Utilities used in the package."""

from .util import UnevaluatedRealPropagatingExpr


def ordered_add(*args):
    """Preserve the order of terms."""
    if len(args) == 0:
        return 0
    elif (len(args)) == 1:
        return args[0]
    a, b, *rest = args
    ue = UnevaluatedRealPropagatingExpr(a + b)
    if len(rest) == 0:
        return ue
    else:
        return ordered_add(ue, *rest)
