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

def timeit(callback, *args, **kwargs):
    t0 = time.time()
    result = callback(*args, **kwargs)
    return time.time() - t0, result


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


# Check solution vs analytic reference:
def check(vals, n, p, a, atol, rtol, forgiveness=1):
    for i in range(n-1):
        val = vals[i]
        ref = analytic1(i+1, p, a)
        diff = val - ref
        acceptance = (atol + abs(val)*rtol)*forgiveness
        print(val, ref, diff, acceptance)
        assert abs(diff) < acceptance

@pytest.mark.parametrize('name,forgive', zip(
    'dopri5 dop853 vode'.split(), (1, 1, 1e6)))
def test_scipy(name, forgive):
    n, p, a = 13, 1, 13
    y0 = np.zeros(n)
    y0[0] = 1
    k = [(i+p+1)*math.log(a) for i in range(n-1)]
    atol, rtol = 1e-10, 1e-10

    dydt = decay_dydt_factory(k)
    odesys_dens = OdeSystem.from_callback(dydt, len(k)+1)
    out = odesys_dens.integrate_scipy(
        1, y0, name=name, atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'rosenbrock4 dopri5 bs'.split(), (1, 1, 1)))
def test_odeint(method, forgive):
    n, p, a = 13, 1, 13
    y0 = np.zeros(n)
    y0[0] = 1
    k = [(i+p+1)*math.log(a) for i in range(n-1)]
    atol, rtol = 1e-10, 1e-10

    dydt = decay_dydt_factory(k)
    odesys_dens = OdeSystem.from_callback(dydt, len(k)+1)
    out = odesys_dens.integrate_odeint(
        1, y0, method=method, atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)



@pytest.mark.parametrize('n,forgive', [(4, 1), (17, 1), (42, 5)])
@pytest.mark.xfail
def test_long_chain_dense(n, forgive):
    p = 1
    a = n
    y0 = np.zeros(n)
    y0[0] = 1
    k = [(i+p+1)*math.log(a) for i in range(n-1)]
    atol, rtol = 1e-12, 1e-12

    dydt = decay_dydt_factory(k)
    odesys_dens = OdeSystem.from_callback(dydt, len(k)+1)
    out = odesys_dens.integrate_scipy(1, y0, atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)



@pytest.mark.parametrize('n', [4])
@pytest.mark.xfail
def test_long_chain_banded(n):
    p = 1
    a = n
    y0 = np.zeros(n)
    y0[0] = 1
    k = [(i+p+1)*math.log(a) for i in range(n-1)]
    atol, rtol = 1e-7, 1e-7

    dydt = decay_dydt_factory(k)
    odesys_dens = OdeSystem.from_callback(dydt, n)
    odesys_band = OdeSystem.from_callback(dydt, n, lband=1, uband=0)

    args = (1, y0)
    kwargs = dict(atol=atol*1e-6, rtol=rtol*1e-6,
                  name='vode', method='adams')

    time_dens, out_dens = timeit(odesys_dens.integrate_scipy,
                                 *args, **kwargs)
    time_band, out_band = timeit(odesys_band.integrate_scipy,
                                 *args, **kwargs)
    check(out_dens[-1, 1:], n, p, a, atol, rtol, 6e3)  # vode poor
    check(out_band[-1, 1:], n, p, a, atol, rtol, 6e3)  # lsoda
    assert time_dens > time_band  # will fail sometimes due to load
