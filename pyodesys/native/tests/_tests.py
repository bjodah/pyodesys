# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import numpy as np
import sympy as sp

from pyodesys.symbolic import ScaledSys, TransformedSys, symmetricsys

from pyodesys.tests.test_core import (
    vdp_f, _test_integrate_multiple_adaptive, _test_integrate_multiple_predefined, sine, decay
)
from pyodesys.tests.bateman import bateman_full  # analytic, never mind the details
from pyodesys.tests.test_symbolic import decay_rhs, decay_dydt_factory


def _test_NativeSys(NativeSys, **kwargs):
    native = NativeSys.from_callback(vdp_f, 2, 1)
    xout, yout, info = native.integrate([0, 1, 2], [1, 0], params=[2.0], **kwargs)
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    if 'nfev' in info:
        assert info['nfev'] > 0


def _test_NativeSys_two(NativeSys, nsteps=500):
    native1 = NativeSys.from_callback(vdp_f, 2, 1)

    tend2, k2, y02 = 2, [4, 3], (5, 4, 2)
    atol2, rtol2 = 1e-11, 1e-11

    native2 = NativeSys.from_callback(decay_dydt_factory(k2), len(k2)+1)

    xout1, yout1, info1 = native1.integrate([0, 1, 2], [1, 0], params=[2.0], nsteps=nsteps)
    xout2, yout2, info2 = native2.integrate(tend2, y02, atol=atol2, rtol=rtol2, nsteps=nsteps)

    # blessed values:
    ref1 = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout1, ref1)
    if 'nfev' in info1:
        assert info1['nfev'] > 0

    ref2 = np.array(bateman_full(y02, k2+[0], xout2 - xout2[0], exp=np.exp)).T
    assert np.allclose(yout2, ref2, rtol=150*rtol2, atol=150*atol2)


def _test_ScaledSys_NativeSys(NativeSys, nsteps=1000):
    class ScaledNativeSys(ScaledSys, NativeSys):
        pass

    k = k0, k1, k2 = [7., 3, 2]
    y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3', real=True, positive=True)
    # this is actually a silly example since it is linear
    l = [
        (y0, -7*y0),
        (y1, 7*y0 - 3*y1),
        (y2, 3*y1 - 2*y2),
        (y3, 2*y2)
    ]
    ss = ScaledNativeSys(l, dep_scaling=1e8)
    y0 = [0]*(len(k)+1)
    y0[0] = 1
    xout, yout, info = ss.integrate([1e-12, 1], y0,
                                    atol=1e-12, rtol=1e-12, nsteps=nsteps)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=2e-11, atol=2e-11)


def _test_symmetricsys_nativesys(NativeSys, nsteps=800, forgive=150):
    logexp = (sp.log, sp.exp)
    first_step = 1e-4
    rtol = atol = 1e-7
    k = [7., 3, 2]

    class TransformedNativeSys(TransformedSys, NativeSys):
        pass

    SS = symmetricsys(logexp, logexp, SuperClass=TransformedNativeSys)

    ts = SS.from_callback(decay_rhs, len(k)+1, len(k))
    y0 = [1e-20]*(len(k)+1)
    y0[0] = 1

    xout, yout, info = ts.integrate(
        [1e-12, 1], y0, k, atol=atol, rtol=rtol,
        first_step=first_step, nsteps=nsteps)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    np.set_printoptions(linewidth=240)
    assert np.allclose(yout, ref, rtol=rtol*forgive, atol=atol*forgive)


def _test_multiple_adaptive(NativeSys, **kwargs):
    native = NativeSys.from_callback(sine, 2, 1)
    _test_integrate_multiple_adaptive(native, integrator='native', **kwargs)


def _test_multiple_predefined(NativeSys, **kwargs):
    native = NativeSys.from_callback(decay, 2, 1)
    _test_integrate_multiple_predefined(native, integrator='native', **kwargs)
