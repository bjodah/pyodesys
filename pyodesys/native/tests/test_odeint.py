# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)


from ._tests import _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys, _test_symmetricsys_nativesys
from ..odeint import NativeOdeintSys as NativeSys


def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='odeint')


def test_NativeSys_two():
    _test_NativeSys_two(NativeSys, nsteps=800)


def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys, nsteps=4000)


def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys, nsteps=16000, forgive=750)
