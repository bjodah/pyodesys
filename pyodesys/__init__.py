# -*- coding: utf-8 -*-
"""
Straightforward numerical integration of ODE systems from SymPy.

The ``pyodesys`` package for enables intuitive solving of systems
of first order differential equations numerically. The system may be
symbolically defined.

``pyodesys`` ties computer algebra systems (like SymPy and symengine), and
numerical solvers (such as ODEPACK in SciPy, CVode in sundials, GSL or odeint)
together.
"""

from __future__ import absolute_import

from ._release import __version__
from .core import OdeSys
