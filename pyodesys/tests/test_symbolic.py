# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

import math

import numpy as np
import sympy as sp
import pytest
import time

from ..symbolic import SymbolicSys
from ..symbolic import ScaledSys, symmetricsys, PartiallySolvedSystem
from .bateman import bateman_full  # analytic, never mind the details


def identity(x):
    return x

idty2 = (identity, identity)
logexp = (sp.log, sp.exp)


def decay_rhs(t, y, k):
    ny = len(y)
    dydt = [0]*ny
    for idx in range(ny):
        if idx < ny-1:
            dydt[idx] -= y[idx]*k[idx]
        if idx > 0:
            dydt[idx] += y[idx-1]*k[idx-1]
    return dydt


def _test_TransformedSys(dep_tr, indep_tr, rtol, atol, first_step, forgive=1):
    k = [7., 3, 2]
    ts = symmetricsys(dep_tr, indep_tr).from_callback(
        decay_rhs, len(k)+1, len(k))
    y0 = [1e-20]*(len(k)+1)
    y0[0] = 1
    xout, yout, info = ts.integrate(
        [1e-12, 1], y0, k, integrator='cvode', atol=atol, rtol=rtol,
        first_step=first_step)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    np.set_printoptions(linewidth=240)
    assert np.allclose(yout, ref, rtol=rtol*forgive, atol=atol*forgive)


def test_TransformedSys_liny_linx():
    _test_TransformedSys(idty2, idty2, 1e-11, 1e-11, 0, 10)


def test_TransformedSys_logy_logx():
    _test_TransformedSys(logexp, logexp, 1e-7, 1e-7, 1e-4, 100)


def test_TransformedSys_logy_linx():
    _test_TransformedSys(logexp, idty2, 1e-8, 1e-8, 0, 100)


def test_TransformedSys_liny_logx():
    _test_TransformedSys(idty2, logexp, 1e-9, 1e-9, 0, 100)


def test_ScaledSys():
    k = k0, k1, k2 = [7., 3, 2]
    y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3', real=True, positive=True)
    l = [
        (y0, -7*y0),
        (y1, 7*y0 - 3*y1),
        (y2, 3*y1 - 2*y2),
        (y3, 2*y2)
    ]
    ss = ScaledSys(l, dep_scaling=1e8)
    y0 = [0]*(len(k)+1)
    y0[0] = 1
    xout, yout, info = ss.integrate([1e-12, 1], y0, integrator='cvode',
                                    atol=1e-12, rtol=1e-12)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=1e-12, atol=1e-12)


def test_ScaledSys_from_callback():
    def f(t, x, k):
        return [-k[0]*x[0],
                k[0]*x[0] - k[1]*x[1],
                k[1]*x[1] - k[2]*x[2],
                k[2]*x[2]]
    odesys = ScaledSys.from_callback(f, 4, 3, 1e8)
    k = [7, 3, 2]
    y0 = [0]*(len(k)+1)
    y0[0] = 1
    xout, yout, info = odesys.integrate([1e-12, 1], y0, k, integrator='scipy')
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=3e-11, atol=3e-11)


def timeit(callback, *args, **kwargs):
    t0 = time.clock()
    result = callback(*args, **kwargs)
    return time.clock() - t0, result


@pytest.mark.parametrize('method', ['bs', 'rosenbrock4'])
def test_exp(method):
    x = sp.Symbol('x')
    symsys = SymbolicSys([(x, sp.exp(x))])
    tout = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1]
    xout, yout, info = symsys.integrate(
        tout, [1], method=method, integrator='odeint', atol=1e-12, rtol=1e-12)
    e = math.e
    ref = -math.log(1/e - 0.1)
    assert abs(yout[-1, 0] - ref) < 4e-8


# @pytest.mark.xfail
def _test_mpmath():  # too slow
    x = sp.Symbol('x')
    symsys = SymbolicSys([(x, sp.exp(x))])
    tout = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1]
    # import pudb; pudb.set_trace()
    xout, yout, info = symsys.integrate(tout, [1], integrator='mpmath')
    e = math.e
    ref = -math.log(1/e - 0.1)
    assert abs(yout[-1, 0] - ref) < 4e-8


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
@pytest.mark.parametrize('band', [(1, 0), None])
def test_SymbolicSys__from_callback_bateman(band):
    # Decay chain of 3 species (2 decays)
    # A --[k0=4]--> B --[k1=3]--> C
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    atol, rtol = 1e-11, 1e-11
    odesys = SymbolicSys.from_callback(decay_dydt_factory(k), len(k)+1,
                                       band=band)
    xout, yout, info = odesys.integrate(
        tend, y0, atol=atol, integrator='scipy', rtol=rtol)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('band', [(1, 0), None])
def test_SymbolicSys_bateman(band):
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    y = sp.symarray('y', len(k)+1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = SymbolicSys(zip(y, f), band=band)
    xout, yout, info = odesys.integrate(tend, y0, integrator='scipy')
    ref = np.array(bateman_full(y0, k+[0], xout-xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref)


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
        xout, yout, info = odesys_dens.integrate(
            tout, y0, integrator='scipy', name=name, atol=atol, rtol=rtol)
        check(yout[-1, :], n, p, a, atol, rtol, forgive[0])

        xout, yout, info = odesys_dens.integrate(
            1, y0, integrator='scipy', name=name, atol=atol, rtol=rtol)
        check(yout[-1, :], n, p, a, atol, rtol, forgive[1])

    else:
        xout, yout, info = odesys_dens.integrate(
            1, y0, integrator='scipy', name=name, atol=atol, rtol=rtol)
        check(yout[-1, :], n, p, a, atol, rtol, forgive)


# (dopri5, .2), (bs, .03) <-- works in boost 1.59
@pytest.mark.parametrize('method,forgive', zip(
    'rosenbrock4'.split(), (.2,)))
def test_odeint(method, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k)+1)
    # adaptive stepper fails to produce the accuracy asked for.
    xout, yout, info = odesys_dens.integrate(
        [10**i for i in range(-15, 1)], y0, integrator='odeint',
        method=method, atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)


def _gsl(tout, method, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k)+1)
    # adaptive stepper fails to produce the accuracy asked for.
    xout, yout, info = odesys_dens.integrate(
        tout, y0, method=method, integrator='gsl', atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'bsimp msadams msbdf rkck'.split(), (0.02, 5, 14, 0.2)))
def test_gsl_predefined(method, forgive):
    _gsl([10**i for i in range(-15, 1)], method, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'bsimp msadams msbdf rkck'.split(), (0.004, 4, 14, 0.21)))
def test_gsl_adaptive(method, forgive):
    _gsl(1, method, forgive)


def _cvode(tout, method, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    dydt = decay_dydt_factory(k)
    odesys_dens = SymbolicSys.from_callback(dydt, len(k)+1)
    # adaptive stepper fails to produce the accuracy asked for.
    xout, yout, info = odesys_dens.integrate(
        tout, y0, method=method, integrator='cvode', atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('method,forgive', zip(
    'adams bdf'.split(), (.24, .9)))
def test_cvode_predefined(method, forgive):
    _cvode([10**i for i in range(-15, 1)], method, forgive)


# cvode performs significantly better than vode:
@pytest.mark.parametrize('method,forgive', zip(
    'adams bdf'.split(), (.36, 2)))
def test_cvode_adaptive(method, forgive):
    _cvode(1, method, forgive)


@pytest.mark.parametrize('n,forgive', [(4, 1), (17, 1), (42, 7)])
def test_long_chain_dense(n, forgive):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    atol, rtol = 1e-12, 1e-12
    tout = 1
    xout, yout, info = odesys_dens.integrate(
        tout, y0, integrator='scipy', atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)


@pytest.mark.parametrize('n', [29, 30])  # something maxes out at 31
def test_long_chain_banded_scipy(n):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    y0, k, odesys_band = get_special_chain(n, p, a, band=(1, 0))
    atol, rtol = 1e-7, 1e-7
    tout = np.logspace(-10, 0, 10)

    def mk_callback(odesys):
        def callback(*args, **kwargs):
            return odesys.integrate(*args, integrator='scipy', **kwargs)
        return callback
    for _ in range(2):  # warmup
        time_dens, (xout_dens, yout_dens, info) = timeit(
            mk_callback(odesys_dens), tout, y0, atol=atol, rtol=rtol,
            name='vode', method='bdf', first_step=1e-10)
        assert info['njev'] > 0
    for _ in range(2):  # warmup
        time_band, (xout_band, yout_band, info) = timeit(
            mk_callback(odesys_band), tout, y0, atol=atol, rtol=rtol,
            name='vode', method='bdf', first_step=1e-10)
        assert info['njev'] > 0
    check(yout_dens[-1, :], n, p, a, atol, rtol, 1.5)
    check(yout_band[-1, :], n, p, a, atol, rtol, 1.5)
    assert time_dens > time_band  # will fail sometimes due to load


@pytest.mark.parametrize('n', [29, 79])
def test_long_chain_banded_cvode(n):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    y0, k, odesys_band = get_special_chain(n, p, a, band=(1, 0))
    atol, rtol = 1e-9, 1e-9

    def mk_callback(odesys):
        def callback(*args, **kwargs):
            return odesys.integrate(*args, integrator='cvode', **kwargs)
        return callback
    for _ in range(2):  # warmup
        time_band, (xout_band, yout_band, info) = timeit(
            mk_callback(odesys_band), 1, y0, atol=atol, rtol=rtol)
        assert info['njev'] > 0
    for _ in range(2):  # warmup
        time_dens, (xout_dens, yout_dens, info) = timeit(
            mk_callback(odesys_dens), 1, y0, atol=atol, rtol=rtol)
        assert info['njev'] > 0
    check(yout_dens[-1, :], n, p, a, atol, rtol, 7)
    check(yout_band[-1, :], n, p, a, atol, rtol, 25)  # suspicious
    assert info['njev'] > 0
    try:
        assert time_dens > time_band
    except AssertionError:
        pass  # will fail sometimes due to load


def test_PartiallySolvedSystem():
    odesys = SymbolicSys.from_callback(
        lambda x, y, p: [
            -p[0]*y[0],
            p[0]*y[0] - p[1]*y[1],
            p[1]*y[1] - p[2]*y[2]
        ], 3, 3)
    dep0 = odesys.dep[0]
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0: {
        dep0: y0[0]*sp.exp(-p0[0]*(odesys.indep-x0))
    })
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 1.5]
    xout, yout, info = partsys.integrate([0, 1], y0, k, integrator='scipy')
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref)
