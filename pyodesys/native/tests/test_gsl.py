# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)


from ._tests import _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys, _test_symmetricsys_nativesys
from ..gsl import NativeGSLSys as NativeSys


def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='gsl')


def test_NativeSys_two():
    _test_NativeSys_two(NativeSys)


def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys)


def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys)
