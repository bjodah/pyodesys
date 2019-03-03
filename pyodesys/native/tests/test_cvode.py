# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest

from pyodesys.util import requires
from pyodesys.symbolic import SymbolicSys, PartiallySolvedSystem

from ._tests import (
    _test_NativeSys, _test_NativeSys_two, _test_ScaledSys_NativeSys, _test_symmetricsys_nativesys,
    _test_multiple_adaptive, _test_multiple_predefined, _test_multiple_adaptive_chained,
    _test_PartiallySolved_symmetric_native, _test_PartiallySolved_symmetric_native_multi,
    _test_Decay_nonnegative, _test_NativeSys__first_step_cb, _test_NativeSys__first_step_cb_source_code,
    _test_NativeSys__roots, _test_NativeSys__get_dx_max_source_code, _test_NativeSys__band,
    _test_NativeSys__dep_by_name__single_varied, _test_PartiallySolvedSystem_Native,
    _test_return_on_error_success
)
from ._test_robertson_native import _test_chained_multi_native
from ._test_proper_rendering import _test_render_native_code_cse
from ..cvode import NativeCvodeSys as NativeSys
from pyodesys.tests.test_symbolic import _test_chained_parameter_variation


@pytest.mark.veryslow
@requires('pycvodes')
def test_NativeSys():
    _test_NativeSys(NativeSys, integrator='cvode')


@pytest.mark.veryslow
@requires('pycvodes')
def test_NativeSys_two():
    _test_NativeSys_two(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test_ScaledSys_NativeSys():
    _test_ScaledSys_NativeSys(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test_symmetricsys_nativesys():
    _test_symmetricsys_nativesys(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test_multiple_adaptive():
    _test_multiple_adaptive(NativeSys, nsteps=700)


@pytest.mark.slow
@requires('pycvodes')
def test_multiple_predefined():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10)


@pytest.mark.veryslow
@requires('pycvodes')
def test_multiple_adaptive_chained():
    _test_multiple_adaptive_chained(
        NativeSys, {'nsteps': (850, 1100), 'autorestart': (0, 3), 'return_on_error': (True, False)})


@pytest.mark.slow
@requires('pycvodes')
@pytest.mark.parametrize('multiple', [False, True])
def test_PartiallySolved_symmetric_native(multiple):
    _test_PartiallySolved_symmetric_native(NativeSys, multiple, forgive=1e2)


@pytest.mark.slow
@requires('pycvodes')
@pytest.mark.parametrize('multiple', [False, True])
def test_PartiallySolved_symmetric_native_multi(multiple):
    _test_PartiallySolved_symmetric_native_multi(NativeSys, multiple, forgive=1e2)


@pytest.mark.veryslow
@requires('pycvodes')
@pytest.mark.parametrize('reduced', [0, 3])
def test_chained_multi_native(reduced):
    _test_chained_multi_native(
        NativeSys, 'cvode', logc=True, logt=True, reduced=reduced, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None
    )


@pytest.mark.veryslow
@requires('pycvodes')
def test_chained_multi_native_nonnegative():
    _test_chained_multi_native(
        NativeSys, 'cvode', logc=True, logt=True, reduced=0, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=True
    )


@requires('pycvodes')
def test_Decay_nonnegative():
    _test_Decay_nonnegative(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test_NativeSys_first_step_expr__decay():
    _test_NativeSys__first_step_cb(NativeSys)


@pytest.mark.veryslow
@requires('pycvodes')
def test_NativeSys_first_step_expr__source_code():
    _test_NativeSys__first_step_cb_source_code(NativeSys, -30, True, return_on_error=True, atol=1e-8, rtol=1e-8)
    _test_NativeSys__first_step_cb_source_code(NativeSys, -10, False, return_on_error=True, atol=1e-8, rtol=1e-8)


@pytest.mark.slow
@requires('pycvodes')
def test_roots():
    _test_NativeSys__roots(NativeSys)


@pytest.mark.veryslow
@requires('pycvodes')
def test_chained_multi_native__dx_max_scalar():
    _test_chained_multi_native(
        NativeSys, logc=True, logt=True, reduced=0, zero_time=1e-10,
        zero_conc=1e-18, nonnegative=None, integrator='cvode', dx_max=1e10
    )


@pytest.mark.slow
@requires('pycvodes')
def test_NativeSys_get_dx_max_source_code():
    _test_NativeSys__get_dx_max_source_code(NativeSys, atol=1e-8, rtol=1e-8, nsteps=1000)


@requires('pycvodes')
@pytest.mark.xfail  # not yet implemented
def test_NativeSys__band():
    _test_NativeSys__band(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test_NativeSys__dep_by_name__single_varied():
    _test_NativeSys__dep_by_name__single_varied(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test_NativeSys__roots():
    def f(t, y):
        return [y[0]]

    def roots(t, y, p, backend):
        return [y[0] - backend.exp(1)]

    odesys = NativeSys.from_callback(f, 1, roots_cb=roots)
    kwargs = dict(first_step=1e-12, atol=1e-12, rtol=1e-12, method='adams', integrator='cvode')
    for return_on_root in (False, True):
        result = odesys.integrate(2, [1], **kwargs)
        assert len(result.info['root_indices']) == 1
        assert result.info['success'] == True  # noqa
        assert np.min(np.abs(result.xout - 1)) < 1e-11


@pytest.mark.veryslow
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('idx', [0, 1, 2])
def test_NativeSys__PartiallySolvedSystem__roots(idx):
    def f(x, y, p):
        return [-p[0]*y[0], p[0]*y[0] - p[1]*y[1], p[1]*y[1]]

    def roots(x, y):
        return ([y[0] - y[1]], [y[0] - y[2]], [y[1] - y[2]])[idx]

    odesys = SymbolicSys.from_callback(f, 3, 2, roots_cb=roots)
    _p, _q, tend = 7, 3, 0.7
    dep0 = (1, 0, 0)
    ref = [0.11299628093544488, 0.20674119231833346, 0.3541828705348678]

    def check(odesys):
        res = odesys.integrate(tend, dep0, (_p, _q),
                               integrator='cvode', return_on_root=True)
        assert abs(res.xout[-1] - ref[idx]) < 1e-7

    check(odesys)
    native = NativeSys.from_other(odesys)
    check(native)

    psys = PartiallySolvedSystem(odesys, lambda t0, xyz, par0, be: {
        odesys.dep[0]: xyz[0]*be.exp(-par0[0]*(odesys.indep-t0))})
    check(psys)
    pnative = NativeSys.from_other(psys)
    check(pnative)


@pytest.mark.slow
@requires('pycvodes')
def test_return_on_error_success():
    _test_return_on_error_success(NativeSys)


@pytest.mark.slow
@requires('pycvodes')
def test__PartiallySolvedSystem_Native():
    _test_PartiallySolvedSystem_Native(NativeSys, 'cvode')


@pytest.mark.slow
@requires('sym', 'pycvodes')
def test_chained_parameter_variation_native_cvode():
    _test_chained_parameter_variation(NativeSys.from_other)


@pytest.mark.slow
@requires('pycvodes', 'sympy')
def test_render_native_cse_regression():
    _test_render_native_code_cse(NativeSys)

try:
    from pycvodes import sundials_version
except Exception:  # ModuleNotFoundError in newer versions of python
    sundials_version = (0, 0, 0)


@pytest.mark.slow
@requires('sym', 'pycvodes')
@pytest.mark.skipif(sundials_version[:2] < (3, 2), reason="constraints req. sundials >=3.2")
def test_constraints():
    _test_multiple_predefined(NativeSys, atol=1e-10, rtol=1e-10, constraints=[1.0, 1.0])


@pytest.mark.slow
@requires('sym', 'pycvodes')
def test_ew_ele():
    for tst in [_test_multiple_predefined, _test_multiple_adaptive]:
        results = tst(NativeSys, atol=1e-10, rtol=1e-10, ew_ele=True, nsteps=1400)
        for res in results:
            ee = res.info['ew_ele']
            assert ee.ndim == 3 and ee.shape[1] == 2
