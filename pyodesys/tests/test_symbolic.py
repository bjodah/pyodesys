from __future__ import print_function, absolute_import, division

import math

import numpy as np
import sympy as sp
import pytest
import time

from .. import SymbolicSys
from .bateman import bateman_full  # analytic, never mind the details


def timeit(callback, *args, **kwargs):
    t0 = time.time()
    result = callback(*args, **kwargs)
    return time.time() - t0, result


# Decay chain
# ===========

def decay_dydt_factory(k):
    # Generates a callback for evaluating a dydt-callback for
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


# Short decay chains, using Bateman's equation
# --------------------------------------------

@pytest.mark.parametrize('bands', [(1, 0), (None, None)])
def test_SymbolicSys__from_callback_bateman(bands):
    # Decay chain of 3 species (2 decays)
    # A --[k0=4]--> B --[k1=3]--> C
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    atol, rtol = 1e-11, 1e-11
    odesys = SymbolicSys.from_callback(decay_dydt_factory(k), len(k)+1,
                                       lband=bands[0], uband=bands[1])
    out = odesys.integrate_scipy(tend, y0, atol=atol, rtol=rtol)
    ref = out.copy()
    ref[:, 1:] = np.array(bateman_full(y0, k+[0], ref[:, 0],
                                       exp=np.exp)).T
    assert np.allclose(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('bands', [(1, 0), (None, None)])
def test_SymbolicSys_bateman(bands):
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    y = sp.symarray('y', len(k)+1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = SymbolicSys(zip(y, f), lband=bands[0], uband=bands[1])
    out = odesys.integrate_scipy(tend, y0)
    ref = out.copy()
    ref[:, 1:] = np.array(bateman_full(y0, k+[0], ref[:, 0],
                                       exp=np.exp)).T
    assert np.allclose(out, ref)


# Longer chains with careful choice of parameters
# -----------------------------------------------


def analytic1(i, p, a):
    assert i > 0
    assert p >= 0
    assert a > 0
    from scipy.special import binom
    return binom(p+i-1, p) * a**(i-1) * (a+1)**(-i-p)


def check(vals, n, p, a, atol, rtol, forgiveness=1):
    # Check solution vs analytic reference:
    for i in range(n-1):
        val = vals[i]
        ref = analytic1(i+1, p, a)
        diff = val - ref
        acceptance = (atol + abs(ref)*rtol)*forgiveness
        assert abs(diff) < acceptance


def get_special_chain(n, p, a, **kwargs):
    assert n > 1
    assert p >= 0
    assert a > 0
    y0 = np.zeros(n)
    y0[0] = 1
    k = [(i+p+1)*math.log(a+1) for i in range(n-1)]
    dydt = decay_dydt_factory(k)
    return y0, k, SymbolicSys.from_callback(dydt, n, **kwargs)


@pytest.mark.parametrize('p', [0, 1, 2, 3])
def test_check(p):
    n, a = 7, 5
    y0, k, _odesys = get_special_chain(n, p, a)
    vals = bateman_full(y0, k+[0], 1, exp=np.exp)
    check(vals, n, p, a, atol=1e-12, rtol=1e-12)


@pytest.mark.xfail
def test_mpmath():
    n, p, a = 3, 1, 3
    y0, k, odesys = get_special_chain(n, p, a)
    out = odesys.integrate_mpmath(1, y0)
    check(out[-1, 1:], n, p, a, 1e-12, 1e-12)


# adaptive stepsize with vode is performing ridiculously
# poorly for this problem
@pytest.mark.parametrize('name,forgive', zip(
    'dopri5 dop853 vode'.split(), (1, 1, (3, 3e6))))
def test_scipy(name, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    if name == 'vode':
        tout = [0]+[10**i for i in range(-10, 1)] if name == 'vode' else 1
        out = odesys_dens.integrate_scipy(
            tout, y0, name=name, atol=atol, rtol=rtol)
        check(out[-1, 1:], n, p, a, atol, rtol, forgive[0])

        out = odesys_dens.integrate_scipy(
            1, y0, name=name, atol=atol, rtol=rtol)
        check(out[-1, 1:], n, p, a, atol, rtol, forgive[1])

    else:
        out = odesys_dens.integrate_scipy(
            1, y0, name=name, atol=atol, rtol=rtol)
        check(out[-1, 1:], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'rosenbrock4 dopri5 bs'.split(), (.2, .2, .03)))
def test_odeint(method, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k)+1)
    # adaptive stepper fails to produce the accuracy asked for.
    out = odesys_dens.integrate_odeint(
        [10**i for i in range(-15, 1)], y0, method=method,
        atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)


def _gsl(tout, method, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k)+1)
    # adaptive stepper fails to produce the accuracy asked for.
    out = odesys_dens.integrate_gsl(tout, y0, method=method,
                                    atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'bsimp msadams msbdf rkck'.split(), (0.02, 5, 14, 0.2)))
def test_gsl_predefined(method, forgive):
    _gsl([10**i for i in range(-15, 1)], method, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'bsimp msadams msbdf rkck'.split(), (0.002, 4, 14, 0.21)))
def test_gsl_adaptive(method, forgive):
    _gsl(1, method, forgive)


def _cvode(tout, method, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k)+1)
    # adaptive stepper fails to produce the accuracy asked for.
    out = odesys_dens.integrate_cvode(tout, y0, method=method,
                                      atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'adams bdf'.split(), (.17, .13)))
def test_cvode_predefined(method, forgive):
    _cvode([10**i for i in range(-15, 1)], method, forgive)


# cvode performs significantly better than vode:
@pytest.mark.parametrize('method,forgive', zip(
    'adams bdf'.split(), (.18, 2)))
def test_cvode_adaptive(method, forgive):
    _cvode(1, method, forgive)


@pytest.mark.parametrize('n,forgive', [(4, 1), (17, 1), (42, 7)])
def test_long_chain_dense(n, forgive):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    atol, rtol = 1e-12, 1e-12
    tout = 1
    out = odesys_dens.integrate_scipy(tout, y0, atol=atol, rtol=rtol)
    check(out[-1, 1:], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('n', [17, 22])
def test_long_chain_banded_scipy(n):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    y0, k, odesys_band = get_special_chain(n, p, a, lband=1, uband=0)
    atol, rtol = 1e-7, 1e-7
    time_dens, out_dens = timeit(odesys_dens.integrate_scipy,
                                 1, y0, atol=atol, rtol=rtol)
    time_band, out_band = timeit(odesys_band.integrate_scipy,
                                 1, y0, atol=atol, rtol=rtol)
    check(out_dens[-1, 1:], n, p, a, atol, rtol, .4)
    check(out_band[-1, 1:], n, p, a, atol, rtol, .4)
    print(time_dens, time_band)
    assert time_dens > time_band  # will fail sometimes due to load


@pytest.mark.parametrize('n', [52])
def test_long_chain_banded_cvode(n):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    y0, k, odesys_band = get_special_chain(n, p, a, lband=1, uband=0)
    atol, rtol = 1e-9, 1e-9
    time_dens, out_dens = timeit(odesys_dens.integrate_cvode,
                                 1, y0, atol=atol, rtol=rtol)
    time_band, out_band = timeit(odesys_band.integrate_cvode,
                                 1, y0, atol=atol, rtol=rtol)
    check(out_dens[-1, 1:], n, p, a, atol, rtol, 0.5)
    check(out_band[-1, 1:], n, p, a, atol, rtol, 10)  # suspicious
    print(time_dens, time_band)
    assert time_dens > time_band  # will fail sometimes due to load
