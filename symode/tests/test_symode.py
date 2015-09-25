from __future__ import print_function, absolute_import, division

import math

import numpy as np
import sympy as sp
import pytest
import time

from .. import OdeSystem
from .bateman import bateman_full  # analytic, never mind the details

def analytic1(i, p, a):
    from scipy.special import binom
    return binom(p+i-1, p) * a**(-1-p) * ((a-1)/a)**(i-1)


def decay_dydt_factory(k):
    # Generates a callback for evaluating a dydt for
    # a chain of len(k) + 1 species with len(k) decays
    # with corresponding decay constants k
    ny = len(k) + 1
    def dydt(t, y):
        exprs = []
        for idx in range(ny):
            expr = 0
            if idx < ny-1:
                expr -= y[idx]*k[idx]
            if idx > 0:
                expr += y[idx-1]*k[idx-1]
            exprs.append(expr)
        return exprs
    return dydt


@pytest.mark.parametrize('bands', [(1, 0), (None, None)])
def test_OdeSystem__from_callback(bands):
    # Decay chain of 3 species (2 decays)
    # A --[k0=4]--> B --[k1=3]--> C
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    atol, rtol = 1e-11, 1e-11
    odesys = OdeSystem.from_callback(decay_dydt_factory(k), len(k)+1,
                                     lband=bands[0], uband=bands[1])
    out = odesys.integrate_scipy(tend, y0, atol=atol, rtol=rtol)
    ref = out.copy()
    ref[:, 1:] = np.array(bateman_full(y0, k+[0], ref[:, 0],
                                       exp=np.exp)).T
    assert np.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('bands', [(1, 0), (None, None)])
def test_OdeSystem(bands):
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    y = sp.symarray('y', len(k)+1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = OdeSystem(zip(y, f), lband=bands[0], uband=bands[1])
    out = odesys.integrate_scipy(tend, y0)
    ref = out.copy()
    ref[:, 1:] = np.array(bateman_full(y0, k+[0], ref[:, 0],
                                       exp=np.exp)).T
    assert np.allclose(out, ref)


@pytest.mark.parametrize('bands', [(1, 0), (None, None)])
def test_long_chain(bands):
    n, p, a = 42, 1, 42
    y0 = np.zeros(n)
    y0[0] = 1
    k = [(i+p+1)*math.log(a) for i in range(n-1)]
    atol, rtol = 1e-11, 1e-11
    odesys = OdeSystem.from_callback(decay_dydt_factory(k), len(k)+1,
                                     lband=bands[0], uband=bands[1])

    tim = time.time()
    out = odesys.integrate_scipy(1, y0, atol=atol, rtol=rtol)
                              # name='vode', method='adams')
    print(time.time() - tim)

    # Check solution vs analytic reference:
    forgiveness = 1
    for i in range(n-1):
        ref = analytic1(i+1, p, a)
        val = out[-1, i+1]
        diff = val - ref
        #print(val, ref, diff, (atol + abs(val)*rtol)*forgiveness)
        assert abs(diff) < (atol + abs(val)*rtol)*forgiveness
