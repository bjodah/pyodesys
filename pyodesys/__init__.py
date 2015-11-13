# -*- coding: utf-8 -*-
"""
Package for solving (symbolic) systems of first order differential equations
numerically.

pyodesys ties computer algebra systems like SymPy and symengine, and numerical
solvers such as ODEPACK in SciPy, CVode in sundials, GSL or odeint together.
"""

from __future__ import absolute_import

from .core import OdeSys
from .symbolic import SymbolicSys
assert OdeSys, SymbolicSys  # silence pyflakes
