# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .core import OdeSys
from .symbolic import SymbolicSys
assert OdeSys, SymbolicSys  # silence pyflakes
