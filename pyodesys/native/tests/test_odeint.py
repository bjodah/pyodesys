# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from pyodesys.util import requires
import pytest

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys,
    _test_symmetricsys_nativesys, _test_multiple_adaptive,
    _test_multiple_predefined, _test_multiple_adaptive_chained,
    _test_PartiallySolved_symmetric_native,
    _test_PartiallySolved_symmetric_native_multi,
    _test_Decay_nonnegative, _test_NativeSys__first_step_cb,
)
from ._test_robertson_native import _test_chained_multi_native
from ..odeint import NativeOdeintSys as NativeSys


@pytest.mark.veryslow
@requires('pyodeint')
def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='odeint')


@pytest.mark.veryslow
@requires('pyodeint')
def test_NativeSys_two():
    _test_NativeSys_two(NativeSys, nsteps=800)


@pytest.mark.slow
@requires('pyodeint')
def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys, nsteps=4000)


@pytest.mark.slow
@requires('pyodeint')
def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys, nsteps=16000, forgive=750)


@requires('pyodeint')
def test_multiple_adaptive():
    _test_multiple_adaptive(NativeSys, nsteps=700)


@pytest.mark.slow
@requires('pyodeint')
def test_multiple_predefined():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10)


@pytest.mark.veryslow
@requires('pyodeint')
def test_multiple_adaptive_chained():
    _test_multiple_adaptive_chained(
        NativeSys, {'nsteps': (850, 1100), 'autorestart': (0, 3), 'return_on_error': (True, False)})


@pytest.mark.slow
@requires('pyodeint')
def test_PartiallySolved_symmetric_native():
    _test_PartiallySolved_symmetric_native(NativeSys, first_step=1e-10, nsteps=3000, forgive=1e4)


@pytest.mark.veryslow
@requires('pyodeint')
def test_PartiallySolved_symmetric_native_multi():
    _test_PartiallySolved_symmetric_native_multi(NativeSys, first_step=1e-10, nsteps=3000, forgive=1e4)


@pytest.mark.slow
@requires('pyodeint')
def test_Decay_nonnegative():
    _test_Decay_nonnegative(NativeSys)


@pytest.mark.slow
@requires('pyodeint')
def test_NativeSys_first_step_expr__decay():
    _test_NativeSys__first_step_cb(NativeSys)


@pytest.mark.veryslow
@requires('pyodeint')
def test_chained_multi_native__dx_max_scalar():
    _test_chained_multi_native(
        NativeSys, logc=True, logt=True, reduced=0, zero_time=1e-12, atol=1e-12, rtol=1e-14,
        zero_conc=1e-18, nonnegative=None, integrator='odeint', dx_max=1e10, rtol_close=0.03,
        first_step=1e-14, steps_fact=2.6
    )
