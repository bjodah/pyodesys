# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys,
    _test_symmetricsys_nativesys, _test_multiple_adaptive,
    _test_multiple_predefined
)
from ..cvode import NativeCvodeSys as NativeSys


def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='cvode')


def test_NativeSys_two():
    _test_NativeSys_two(NativeSys)


def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys)


def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys)


def test_multiple_adaptive():
    _test_multiple_adaptive(NativeSys, nsteps=700)


def test_multiple_predefined():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10)
