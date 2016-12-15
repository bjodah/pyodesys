# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest

from pyodesys.util import requires

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys,
    _test_symmetricsys_nativesys, _test_multiple_adaptive,
    _test_multiple_predefined, _test_multiple_adaptive_chained,
    _test_PartiallySolved_symmetric_native,
    _test_PartiallySolved_symmetric_native_multi,
    _test_Decay_nonnegative, _test_NativeSys__first_step_cb,
    _test_NativeSys__first_step_cb_source_code, _test_NativeSys__roots
)
from ._test_robertson_native import _test_chained_multi_native
from ..cvode import NativeCvodeSys as NativeSys


@requires('pycvodes')
def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='cvode')


@requires('pycvodes')
def test_NativeSys_two():
    _test_NativeSys_two(NativeSys)


@requires('pycvodes')
def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys)


@requires('pycvodes')
def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys)


@requires('pycvodes')
def test_multiple_adaptive():
    _test_multiple_adaptive(NativeSys, nsteps=700)


@requires('pycvodes')
def test_multiple_predefined():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10)


@requires('pycvodes')
def test_multiple_adaptive_chained():
    _test_multiple_adaptive_chained(
        NativeSys, {'nsteps': (850, 1100), 'autorestart': (0, 3), 'return_on_error': (True, False)})


@requires('pycvodes')
@pytest.mark.parametrize('multiple', [False, True])
def test_PartiallySolved_symmetric_native(multiple):
    _test_PartiallySolved_symmetric_native(NativeSys, multiple, forgive=1e2)


@requires('pycvodes')
@pytest.mark.parametrize('multiple', [False, True])
def test_PartiallySolved_symmetric_native_multi(multiple):
    _test_PartiallySolved_symmetric_native_multi(NativeSys, multiple, forgive=1e2)


@requires('pycvodes')
@pytest.mark.parametrize('reduced', [0, 3])
def test_chained_multi_native(reduced):
    _test_chained_multi_native(
        NativeSys, 'cvode', logc=True, logt=True, reduced=reduced, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None
    )


@requires('pycvodes')
def test_chained_multi_native_nonnegative():
    _test_chained_multi_native(
        NativeSys, 'cvode', logc=True, logt=True, reduced=0, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=True
    )


@requires('pycvodes')
def test_Decay_nonnegative():
    _test_Decay_nonnegative(NativeSys)


@requires('pycvodes')
def test_NativeSys_first_step_expr__decay():
    _test_NativeSys__first_step_cb(NativeSys)


@requires('pycvodes')
def test_NativeSys_first_step_expr__source_code():
    _test_NativeSys__first_step_cb_source_code(NativeSys, -30, True, return_on_error=True, atol=1e-8, rtol=1e-8)
    _test_NativeSys__first_step_cb_source_code(NativeSys, -10, False, return_on_error=True, atol=1e-8, rtol=1e-8)


@requires('pycvodes')
def test_chained_multi_native__dx_max_scalar():
    _test_chained_multi_native(
        NativeSys, logc=True, logt=True, reduced=0, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None, integrator='cvode', dx_max=1e10
    )


@requires('pycvodes')
def test_roots():
    _test_NativeSys__roots(NativeSys)
