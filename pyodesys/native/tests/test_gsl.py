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
    _test_NativeSys__first_step_cb_source_code, _test_y_preprocessing
)
from ._test_robertson_native import _test_chained_multi_native
from ..gsl import NativeGSLSys as NativeSys


@pytest.mark.veryslow
@requires('pygslodeiv2')
def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='gsl')


@pytest.mark.veryslow
@requires('pygslodeiv2')
def test_NativeSys_two():
    _test_NativeSys_two(NativeSys)


@pytest.mark.slow
@requires('pygslodeiv2')
def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys)


@pytest.mark.slow
@requires('pygslodeiv2')
def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys)


@pytest.mark.slow
@requires('pygslodeiv2')
def test_multiple_adaptive():
    _test_multiple_adaptive(NativeSys, nsteps=700)


@requires('pygslodeiv2')
def test_multiple_predefined():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10)


@pytest.mark.veryslow
@requires('pygslodeiv2')
def test_multiple_adaptive_chained():
    _test_multiple_adaptive_chained(
        NativeSys, {'nsteps': (850, 1100), 'autorestart': (0, 3), 'return_on_error': (True, False)})


@pytest.mark.slow
@requires('pygslodeiv2')
def test_PartiallySolved_symmetric_native():
    _test_PartiallySolved_symmetric_native(NativeSys, first_step=1e-10, forgive=10)


@pytest.mark.slow
@requires('pygslodeiv2')
def test_PartiallySolved_symmetric_native_multi():
    _test_PartiallySolved_symmetric_native_multi(NativeSys, first_step=1e-10, forgive=10)


@pytest.mark.veryslow
@requires('pygslodeiv2')
@pytest.mark.parametrize('reduced', [0, 3])
def test_chained_multi_native(reduced):
    _test_chained_multi_native(
        NativeSys, logc=True, logt=True, reduced=reduced, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None, integrator='gsl'
    )


@pytest.mark.slow
@requires('pygslodeiv2')
def test_Decay_nonnegative():
    _test_Decay_nonnegative(NativeSys)


@pytest.mark.slow
@requires('pygslodeiv2')
def test_NativeSys_first_step_expr__decay():
    _test_NativeSys__first_step_cb(NativeSys)


@pytest.mark.veryslow
@requires('pygslodeiv2')
def test_NativeSys_first_step_expr__source_code():
    _test_NativeSys__first_step_cb_source_code(NativeSys, -30, True, return_on_error=True, atol=1e-8, rtol=1e-8)
    _test_NativeSys__first_step_cb_source_code(NativeSys, -10, None, return_on_error=True, atol=1e-8, rtol=1e-8)


@pytest.mark.veryslow
@requires('pygslodeiv2')
def test_chained_multi_native__dx_max_scalar():
    _test_chained_multi_native(
        NativeSys, logc=True, logt=True, reduced=0, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None, integrator='gsl', dx_max=1e10
    )


@pytest.mark.slow
@requires('sym', 'pycvodes')
def test_y_preprocessing():
    _test_y_preprocessing(NativeSys)
