# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from pyodesys.util import requires

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys,
    _test_symmetricsys_nativesys, _test_multiple_adaptive,
    _test_multiple_predefined, _test_multiple_adaptive_chained,
    _test_PartiallySolved_symmetric_native,
    _test_PartiallySolved_symmetric_native_multi,
    _test_Decay_nonnegative, _test_NativeSys__first_step_cb
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
def test_multiple_adaptive():
    _test_multiple_adaptive(NativeSys, nsteps=700)


@requires('pyodeint')
def test_multiple_predefined():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10)


@requires('pyodeint')
def test_multiple_adaptive_chained():
    _test_multiple_adaptive_chained(
        NativeSys, {'nsteps': (850, 1100), 'autorestart': (0, 3), 'return_on_error': (True, False)})


@requires('pyodeint')
def test_PartiallySolved_symmetric_native():
    _test_PartiallySolved_symmetric_native(NativeSys, first_step=1e-10, nsteps=3000, forgive=1e4)


@requires('pyodeint')
def test_PartiallySolved_symmetric_native_multi():
    _test_PartiallySolved_symmetric_native_multi(NativeSys, first_step=1e-10, nsteps=3000, forgive=1e4)


@requires('pyodeint')
def test_Decay_nonnegative():
    _test_Decay_nonnegative(NativeSys)


@requires('pyodeint')
def test_NativeSys_first_step_expr__decay():
    _test_NativeSys__first_step_cb(NativeSys)
