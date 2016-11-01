# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys,
    _test_symmetricsys_nativesys, _test_multiple_adaptive,
    _test_multiple_predefined, _test_multiple_adaptive_chained,
    _test_PartiallySolved_symmetric_native,
    _test_PartiallySolved_symmetric_native_multi,
    _test_Decay_nonnegative
)
from ._test_robertson_native import _test_chained_multi_native
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


def test_multiple_adaptive_chained():
    _test_multiple_adaptive_chained(
        NativeSys, {'nsteps': (850, 1100), 'autorestart': (0, 3), 'return_on_error': (True, False)})


@pytest.mark.parametrize('multiple', [False, True])
def test_PartiallySolved_symmetric_native(multiple):
    _test_PartiallySolved_symmetric_native(NativeSys, multiple)


@pytest.mark.parametrize('multiple', [False, True])
def test_PartiallySolved_symmetric_native_multi(multiple):
    _test_PartiallySolved_symmetric_native_multi(NativeSys, multiple)


@pytest.mark.parametrize('reduced', [0, 3])
def test_chained_multi_native(reduced):
    _test_chained_multi_native(
        NativeSys, 'cvode', logc=True, logt=True, reduced=reduced, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None
    )


def test_Decay_nonnegative():
    _test_Decay_nonnegative(NativeSys)
