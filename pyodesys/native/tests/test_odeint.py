# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from pyodesys.util import requires

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys,
    _test_symmetricsys_nativesys, _test_NativeSys__first_step_cb
)
from ..odeint import NativeOdeintSys as NativeSys


@requires('pyodeint')
def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='odeint')


@requires('pyodeint')
def test_NativeSys_two():
    _test_NativeSys_two(NativeSys, nsteps=800)


@requires('pyodeint')
def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys, nsteps=4000)


@requires('pyodeint')
def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys, nsteps=16000, forgive=750)


@requires('pyodeint')
def test_NativeSys_first_step_expr__decay():
    _test_NativeSys__first_step_cb(NativeSys)
