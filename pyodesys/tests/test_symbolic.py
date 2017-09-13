# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, division

from collections import defaultdict
from itertools import product
import math

import numpy as np
import pytest
import time

try:
    import sym
except ImportError:
    sym = None
    sym_backends = []
else:
    sym_backends = sym.Backend.backends.keys()

from .. import ODESys
from ..core import integrate_chained
from ..symbolic import SymbolicSys, ScaledSys, symmetricsys, PartiallySolvedSystem, get_logexp, _group_invariants
from ..util import requires
from .bateman import bateman_full  # analytic, never mind the details
from .test_core import vdp_f
from . import _cetsa


def identity(x):
    return x

idty2 = (identity, identity)


def _decay3(x, y, p):
    return [
        -p[0]*y[0],
        p[0]*y[0] - p[1]*y[1],
        p[1]*y[1] - p[2]*y[2]
    ]


def _get_decay3(**kwargs):
    return SymbolicSys.from_callback(_decay3, 3, 3, **kwargs)


def _get_decay3_names(yn, pn, **kwargs):
    def f(x, y, p):
        y = [y[n] for n in yn]
        p = [p[n] for n in pn]
        return dict(zip(yn, [
            -p[0]*y[0],
            p[0]*y[0] - p[1]*y[1],
            p[1]*y[1] - p[2]*y[2]
        ]))
    return SymbolicSys.from_callback(f, names=yn, param_names=pn, dep_by_name=True,
                                     par_by_name=True, indep_name='t', **kwargs)


@requires('sym', 'scipy')
def test_SymbolicSys():
    from pyodesys.integrators import RK4_example_integrator
    odesys = SymbolicSys.from_callback(lambda x, y, p, be: [-y[0], y[0]], 2,
                                       names=['foo', 'bar'])
    assert odesys.autonomous_interface is True
    assert isinstance(odesys.exprs, tuple)
    with pytest.raises(ValueError):
        odesys.integrate(1, [0])

    odesys2 = SymbolicSys.from_callback(lambda x, y, p, be: {'foo': -y['foo'],
                                                             'bar': y['foo']}, 2,
                                        names=['foo', 'bar'], dep_by_name=True)
    for system, y0 in zip([odesys, odesys2], [[2, 3], {'foo': 2, 'bar': 3}]):
        xout, yout, info = system.integrate(1, y0, integrator=RK4_example_integrator, first_step=1e-3)
        assert np.allclose(yout[:, 0], 2*np.exp(-xout))
        assert np.allclose(yout[:, 1], 3 + 2*(1 - np.exp(-xout)))

    with pytest.raises(ValueError):
        SymbolicSys.from_callback(lambda x, y, p, be: None, 2, names=['foo', 'bar'])

    with pytest.raises(ValueError):
        SymbolicSys.from_callback(lambda x, y, p, be: [], 2, names=['foo', 'bar'])


@requires('sym', 'scipy')
def test_SymbolicSys__indep_name():
    odesys = SymbolicSys.from_callback(
        lambda t, y, p: {
            'x': -p['a']*y['x'],
            'y': -p['b']*y['y'] + p['a']*y['x'],
            'z': p['b']*y['y']
        }, names='xyz', param_names='ab', dep_by_name=True, par_by_name=True)
    pars = {'a': [11, 17, 19], 'b': 13}
    results = odesys.integrate([42, 43, 44], {'x': 7, 'y': 5, 'z': 3}, pars)
    for r, a in zip(results, pars['a']):
        assert np.allclose(r.named_dep('x'), 7*np.exp(-a*(r.xout - r.xout[0])))


@requires('sym')
def test_SymbolicSys__init_indep__init_dep():
    odesys = SymbolicSys.from_callback(lambda x, y, p, be: [-y[0], y[0]], 2, names=['foo', 'bar'], indep_name='t',
                                       init_indep=True, init_dep=True)
    assert odesys.init_indep.name == 'i_t'
    assert [dep.name for dep in odesys.init_dep] == ['i_foo', 'i_bar']


def decay_rhs(t, y, k):
    ny = len(y)
    dydt = [0]*ny
    for idx in range(ny):
        if idx < ny-1:
            dydt[idx] -= y[idx]*k[idx]
        if idx > 0:
            dydt[idx] += y[idx-1]*k[idx-1]
    return dydt


def _test_TransformedSys(dep_tr, indep_tr, rtol, atol, first_step, forgive=1, y_zero=1e-20, t_zero=1e-12, **kwargs):
    k = [7., 3, 2]
    ts = symmetricsys(dep_tr, indep_tr).from_callback(
        decay_rhs, len(k)+1, len(k))
    y0 = [y_zero]*(len(k)+1)
    y0[0] = 1
    xout, yout, info = ts.integrate(
        [t_zero, 1], y0, k, integrator='cvode', atol=atol, rtol=rtol,
        first_step=first_step, **kwargs)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=rtol*forgive, atol=atol*forgive)


@requires('sym')
def test_SymbolicSys__jacobian_singular():
    k = (4, 3)
    odesys = SymbolicSys.from_callback(decay_dydt_factory(k), len(k)+1)
    assert odesys.jacobian_singular()


@requires('sym', 'pycvodes')
def test_TransformedSys_liny_linx():
    _test_TransformedSys(idty2, idty2, 1e-11, 1e-11, 0, 15)


@requires('sym', 'pycvodes')
def test_TransformedSys_logy_logx():
    _test_TransformedSys(get_logexp(), get_logexp(), 1e-7, 1e-7, 1e-4, 150, nsteps=800)


@requires('sym', 'pycvodes', 'sympy')
def test_TransformedSys_logy_logx_scaled_shifted():
    import sympy as sp
    em16 = (sp.S.One*10)**-16
    _test_TransformedSys(get_logexp(42, em16), get_logexp(42, em16), 1e-7, 1e-7, 1e-4,
                         150, y_zero=0, t_zero=0, nsteps=800)


@requires('sym', 'pycvodes')
def test_TransformedSys_logy_linx():
    _test_TransformedSys(get_logexp(), idty2, 1e-8, 1e-8, 0, 150, nsteps=1700)


@requires('sym', 'pycvodes')
def test_TransformedSys_liny_logx():
    _test_TransformedSys(idty2, get_logexp(), 1e-9, 1e-9, 0, 150)


@requires('sym', 'pycvodes')
def test_ScaledSys():
    import sympy as sp
    k = k0, k1, k2 = [7., 3, 2]
    y0, y1, y2, y3 = sp.symbols('y0 y1 y2 y3', real=True, positive=True)
    # this is actually a silly example since it is linear
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
                                    atol=1e-12, rtol=1e-12, nsteps=1000)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=2e-11, atol=2e-11)


@requires('sym', 'scipy', 'pycvodes')
@pytest.mark.parametrize('nbody', [2, 3, 4, 5])
def test_ScaledSysByName(nbody):
    sfact = nbody * 7
    kwargs = dict(names=['foo', 'bar'], dep_scaling=sfact)

    def nmerization(x, y, p):
        return [-nbody*p[0]*y[0]**nbody, nbody*p[0]*y[0]**nbody]

    odesys = ScaledSys.from_callback(nmerization, 2, 1, **kwargs)
    assert odesys.autonomous_interface is True
    with pytest.raises(TypeError):
        odesys.integrate(1, [0])

    def nmerization_name(x, y, p):
        return {'foo': -nbody*p[0]*y['foo']**nbody, 'bar': nbody*p[0]*y['foo']**nbody}

    odesys2 = ScaledSys.from_callback(nmerization_name, dep_by_name=True, nparams=1, **kwargs)
    assert odesys2.autonomous_interface is True
    k = 5
    foo0 = 2
    for system, y0 in zip([odesys, odesys2], [[foo0, 3], {'foo': foo0, 'bar': 3}]):
        xout, yout, info = system.integrate(1, y0, [k], integrator='cvode', nsteps=707*1.01,
                                            first_step=1e-3, atol=1e-10, rtol=1e-10)
        _r = (1/(foo0**(1-nbody) + nbody*k*xout*(nbody-1)))**(1/(nbody-1))
        assert np.allclose(yout[:, 0], _r, atol=1e-9, rtol=1e-9)
        assert np.allclose(yout[:, 1], 3 + 2 - _r, atol=1e-9, rtol=1e-9)


@requires('sym', 'scipy')
def test_ScaledSys_from_callback():
    # this is actually a silly example since it is linear
    def f(t, x, k):
        return [-k[0]*x[0],
                k[0]*x[0] - k[1]*x[1],
                k[1]*x[1] - k[2]*x[2],
                k[2]*x[2]]
    odesys = ScaledSys.from_callback(f, 4, 3, 3.14e8)
    k = [7, 3, 2]
    y0 = [0]*(len(k)+1)
    y0[0] = 1
    xout, yout, info = odesys.integrate([1e-12, 1], y0, k, integrator='scipy')
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, rtol=3e-11, atol=3e-11)

    with pytest.raises(TypeError):
        odesys.integrate([1e-12, 1], [0]*len(k), k, integrator='scipy')


@requires('sym', 'scipy')
def test_ScaledSys_from_callback__exprs():
    def f(t, x, k):
        return [-k[0]*x[0]*x[0]*t]
    x, y, nfo = SymbolicSys.from_callback(f, 1, 1).integrate(
        [0, 1], [3.14], [2.78])
    xs, ys, nfos = ScaledSys.from_callback(f, 1, 1, 100).integrate(
        [0, 1], [3.14], [2.78])
    from scipy.interpolate import interp1d
    cb = interp1d(x, y[:, 0])
    cbs = interp1d(xs, ys[:, 0])
    t = np.linspace(0, 1)
    assert np.allclose(cb(t), cbs(t))


def timeit(callback, *args, **kwargs):
    t0 = time.clock()
    result = callback(*args, **kwargs)
    return time.clock() - t0, result


@requires('sym', 'pyodeint')
@pytest.mark.parametrize('method', ['bs', 'rosenbrock4'])
def test_exp(method):
    import sympy as sp
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
    import sympy as sp
    x = sp.Symbol('x')
    symsys = SymbolicSys([(x, sp.exp(x))])
    tout = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1]
    xout, yout, info = symsys.integrate(tout, [1], integrator='mpmath')
    e = math.e
    ref = -math.log(1/e - 0.1)
    assert abs(yout[-1, 0] - ref) < 4e-8


def decay_dydt_factory(k, names=None):
    # Generates a callback for evaluating a dydt-callback for
    # a chain of len(k) + 1 species with len(k) decays
    # with corresponding decay constants k
    ny = len(k) + 1

    def dydt(t, y):
        exprs = []
        for idx in range(ny):
            expr = 0
            curr_key = idx
            prev_key = idx - 1
            if names is not None:
                curr_key = names[curr_key]
                prev_key = names[prev_key]
            if idx < ny-1:
                expr -= y[curr_key]*k[curr_key]
            if idx > 0:
                expr += y[prev_key]*k[prev_key]
            exprs.append(expr)
        if names is None:
            return exprs
        else:
            return dict(zip(names, exprs))

    return dydt


# Short decay chains, using Bateman's equation
# --------------------------------------------
@requires('sym', 'scipy')
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


def _test_bateman(SymbSys, **kwargs):
    import sympy as sp
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    y = sp.symarray('y', len(k)+1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = SymbSys(zip(y, f), **kwargs)
    xout, yout, info = odesys.integrate(tend, y0, integrator='scipy')
    ref = np.array(bateman_full(y0, k+[0], xout-xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref)


@requires('sym', 'scipy')
@pytest.mark.parametrize('band', [(1, 0), None])
def test_SymbolicSys_bateman(band):
    _test_bateman(SymbolicSys, band=band)


@requires('sym', 'scipy')
@pytest.mark.parametrize('band', [(1, 0), None])
def test_ScaledSys_bateman(band):
    _test_bateman(ScaledSys, band=band, dep_scaling=1e3)


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


@requires('sym', 'scipy')
@pytest.mark.parametrize('p', [0, 1, 2, 3])
def test_check(p):
    n, a = 7, 5
    y0, k, _odesys = get_special_chain(n, p, a)
    vals = bateman_full(y0, k+[0], 1, exp=np.exp)
    check(vals, n, p, a, atol=1e-12, rtol=1e-12)


# adaptive stepsize with vode is performing ridiculously
# poorly for this problem
@requires('sym', 'scipy')
@pytest.mark.parametrize('name,forgive', zip(
    'dopri5 dop853 vode'.split(), (1, 1, (3, 3e6))))
def test_scipy(name, forgive):
    n, p, a = 13, 1, 13
    atol, rtol = 1e-10, 1e-10
    y0, k, odesys_dens = get_special_chain(n, p, a)
    if name == 'vode':
        tout = [0]+[10**i for i in range(-10, 1)]
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
    assert yout.shape[0] > 2


# (dopri5, .2), (bs, .03) <-- works in boost 1.59
@pytest.mark.slow
@requires('sym', 'pyodeint')
@pytest.mark.parametrize('method,forgive', zip(
    'rosenbrock4 dopri5 bs'.split(), (.2, .2, .04)))
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


@pytest.mark.veryslow
@requires('sym', 'pygslodeiv2')
@pytest.mark.parametrize('method,forgive', zip(
    'msadams msbdf rkck bsimp'.split(), (5, 14, 0.2, 0.02)))
def test_gsl_predefined(method, forgive):
    _gsl([10**i for i in range(-14, 1)], method, forgive)


@pytest.mark.veryslow
@requires('sym', 'pygslodeiv2')
@pytest.mark.parametrize('method,forgive', zip(
    'bsimp msadams msbdf rkck'.split(), (0.01, 4, 14, 0.21)))
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


@pytest.mark.slow
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('method,forgive', zip(
    'adams bdf'.split(), (2.4, 5.0)))
def test_cvode_predefined(method, forgive):
    _cvode([10**i for i in range(-15, 1)], method, forgive)


# cvode performs significantly better than vode:
@pytest.mark.slow
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('method,forgive', zip(
    'adams bdf'.split(), (2.4, 5)))
def test_cvode_adaptive(method, forgive):
    _cvode(1, method, forgive)


@pytest.mark.veryslow
@requires('sym', 'scipy')
@pytest.mark.parametrize('n,forgive', [(4, 1), (17, 1), (42, 7)])
def test_long_chain_dense(n, forgive):
    p, a = 0, n
    y0, k, odesys_dens = get_special_chain(n, p, a)
    atol, rtol = 1e-12, 1e-12
    tout = 1
    xout, yout, info = odesys_dens.integrate(
        tout, y0, integrator='scipy', atol=atol, rtol=rtol)
    check(yout[-1, :], n, p, a, atol, rtol, forgive)


@pytest.mark.slow
@requires('sym', 'scipy')
@pytest.mark.parametrize('n', [29])  # something maxes out at 31
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
    min_time_dens, min_time_band = float('inf'), float('inf')
    for _ in range(3):  # warmup
        time_dens, (xout_dens, yout_dens, info) = timeit(
            mk_callback(odesys_dens), tout, y0, atol=atol, rtol=rtol,
            name='vode', method='bdf', first_step=1e-10)
        assert info['njev'] > 0
        min_time_dens = min(min_time_dens, time_dens)
    for _ in range(3):  # warmup
        time_band, (xout_band, yout_band, info) = timeit(
            mk_callback(odesys_band), tout, y0, atol=atol, rtol=rtol,
            name='vode', method='bdf', first_step=1e-10)
        assert info['njev'] > 0
        min_time_band = min(min_time_band, time_band)
    check(yout_dens[-1, :], n, p, a, atol, rtol, 1.5)
    check(yout_band[-1, :], n, p, a, atol, rtol, 1.5)
    assert min_time_dens*2 > min_time_band  # (2x: fails sometimes due to load)


@pytest.mark.veryslow
@requires('sym', 'pycvodes')
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


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_no_diff_adaptive_chained_single(integrator):
    odesys = _get_decay3()
    tout, y0, k = [3, 5], [3, 2, 1], [3.5, 2.5, 1.5]
    xout1, yout1, info1 = odesys.integrate(tout, y0, k, integrator=integrator)
    ref = np.array(bateman_full(y0, k, xout1 - xout1[0], exp=np.exp)).T
    assert info1['success']
    assert xout1.size > 10
    assert xout1.size == yout1.shape[0]
    assert np.allclose(yout1, ref)

    xout2, yout2, info2 = integrate_chained([odesys], {}, tout, y0, k, integrator=integrator)
    assert info1['success']
    assert xout2.size == xout1.size
    assert np.allclose(yout2, ref)


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_no_diff_adaptive_chained_single__multimode(integrator):
    odesys = _get_decay3()
    tout = [[3, 5], [4, 6], [6, 8], [9, 11]]
    _y0 = [3, 2, 1]
    y0 = [_y0]*4
    _k = [3.5, 2.5, 1.5]
    k = [_k]*4
    res1 = odesys.integrate(tout, y0, k, integrator=integrator, first_step=1e-14)
    for res in res1:
        xout1, yout1, info1 = res.xout, res.yout, res.info
        ref = np.array(bateman_full(_y0, _k, xout1 - xout1[0], exp=np.exp)).T
        assert info1['success']
        assert xout1.size > 10
        assert xout1.size == yout1.shape[0]
        assert np.allclose(yout1, ref)

    res2 = integrate_chained([odesys], {}, tout, y0, k, integrator=integrator, first_step=1e-14)
    for res in res2:
        xout2, yout2, info2 = res.xout, res.yout, res.info
        assert info2['success']
        assert xout2.size == xout1.size
        assert np.allclose(yout2, ref)


@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem(integrator):
    odesys = _get_decay3(lower_bounds=[0, 0, 0])
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be: {
        odesys.dep[0]: y0[0]*be.exp(-p0[0]*(odesys.indep-x0))
    })
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 1.5]
    xout, yout, info = partsys.integrate(
        [0, 1], y0, k, integrator=integrator)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert info['success']
    assert np.allclose(yout, ref)


@requires('sym', 'pycvodes')
def test_PartiallySolvedSystem_ScaledSys():
    odesys = _get_decay3(lower_bounds=[0, 0, 0])
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be: {
        odesys.dep[0]: y0[0]*be.exp(-p0[0]*(odesys.indep-x0))
    })
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 1.5]

    def _check(res):
        ref = np.array(bateman_full(y0, k, res.xout - res.xout[0], exp=np.exp)).T
        assert res.info['success']
        assert np.allclose(res.yout, ref)
    args = [0, 1], y0, k
    kwargs = dict(integrator='cvode')
    _check(odesys.integrate(*args, **kwargs))
    _check(partsys.integrate(*args, **kwargs))
    scaledsys = ScaledSys.from_other(partsys, dep_scaling=42, indep_scaling=17)
    _check(scaledsys.integrate(*args, **kwargs))


@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem_multi(integrator):
    odesys = _get_decay3()

    def _get_analytic(x0, y0, p0, be):
        a0 = y0[0]*be.exp(-p0[0]*(odesys.indep - x0))
        a1 = y0[0] + y0[1] + y0[2] - a0 - odesys.dep[2]
        return [a0, a1]

    def subst(x0, y0, p0, be):
        a0, a1 = _get_analytic(x0, y0, p0, be)
        return {odesys.dep[0]: a0, odesys.dep[1]: a1}

    partsys = PartiallySolvedSystem(odesys, subst)
    a0, a1 = _get_analytic(partsys.init_indep,
                           partsys.init_dep,
                           odesys.params,
                           odesys.be)
    assert partsys.ny == 1
    assert partsys.exprs[0].subs(odesys.params[2], 0) - odesys.params[1]*a1 == 0


@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem__using_y(integrator):
    odesys = _get_decay3()
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0: {
        odesys.dep[2]: y0[0] + y0[1] + y0[2] - odesys.dep[0] - odesys.dep[1]
    })
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 0]
    xout, yout, info = partsys.integrate([0, 1], y0, k, integrator=integrator)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert info['success']
    assert np.allclose(yout, ref)
    assert np.allclose(np.sum(yout, axis=1), sum(y0))


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem_multiple_subs(integrator):
    odesys = _get_decay3(lower_bounds=[0, 0, 0])

    def substitutions(x0, y0, p0, be):
        analytic0 = y0[0]*be.exp(-p0[0]*(odesys.indep-x0))
        analytic2 = y0[0] + y0[1] + y0[2] - analytic0 - odesys.dep[1]
        return {odesys.dep[0]: analytic0, odesys.dep[2]: analytic2}

    partsys = PartiallySolvedSystem(odesys, substitutions)
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 0]
    xout, yout, info = partsys.integrate([0, 1], y0, k, integrator=integrator)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert info['success']
    assert np.allclose(yout, ref)


@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem_multiple_subs__transformed(integrator):
    odesys = _get_decay3(lower_bounds=[0, 0, 0])

    def substitutions(x0, y0, p0, be):
        analytic0 = y0[0]*be.exp(-p0[0]*(odesys.indep-x0))
        analytic2 = y0[0] + y0[1] + y0[2] - analytic0 - odesys.dep[1]
        return {odesys.dep[0]: analytic0, odesys.dep[2]: analytic2}

    partsys = PartiallySolvedSystem(odesys, substitutions)
    LogLogSys = symmetricsys(get_logexp(), get_logexp())
    loglogpartsys = LogLogSys.from_other(partsys)
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 0]
    tend = 1
    for system, ny_internal in [(odesys, 3), (partsys, 1), (loglogpartsys, 1)]:
        xout, yout, info = system.integrate([1e-12, tend], y0, k, integrator=integrator,
                                            first_step=1e-14, nsteps=1000)
        ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
        assert info['success']
        assert info['internal_yout'].shape[-1] == ny_internal
        if system == loglogpartsys:
            assert info['internal_yout'][-1, 0] < 0  # ln(y[1])
        assert np.allclose(yout, ref)


def _get_transf_part_system():
    import sympy as sp
    odesys = _get_decay3()
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0: {
        odesys.dep[0]: y0[0]*sp.exp(-p0[0]*(odesys.indep-x0))
    })
    LogLogSys = symmetricsys(get_logexp(), get_logexp())
    return LogLogSys.from_other(partsys)


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem__symmetricsys(integrator):
    trnsfsys = _get_transf_part_system()
    y0 = [3., 2., 1.]
    k = [3.5, 2.5, 0]
    xout, yout, info = trnsfsys.integrate([1e-10, 1], y0, k, integrator=integrator, first_step=1e-14)
    ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
    assert info['success']
    assert np.allclose(yout, ref)
    assert np.allclose(np.sum(yout, axis=1), sum(y0))


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem__symmetricsys__multi(integrator):
    trnsfsys = _get_transf_part_system()
    y0s = [[3., 2., 1.], [3.1, 2.1, 1.1], [3.2, 2.3, 1.2], [3.6, 2.4, 1.3]]
    ks = [[3.5, 2.5, 0], [3.3, 2.4, 0], [3.2, 2.1, 0], [3.3, 2.4, 0]]
    results = trnsfsys.integrate([(1e-10, 1)]*len(ks), y0s, ks, integrator=integrator, first_step=1e-14)
    for i, (y0, k) in enumerate(zip(y0s, ks)):
        xout, yout, info = results[i]
        ref = np.array(bateman_full(y0, k, xout - xout[0], exp=np.exp)).T
        assert info['success'] and info['nfev'] > 10
        assert info['nfev'] > 1 and info['time_cpu'] < 100
        assert np.allclose(yout, ref) and np.allclose(np.sum(yout, axis=1), sum(y0))


def _get_nonlin(**kwargs):
    return SymbolicSys.from_callback(
        lambda x, y, p: [
            -p[0]*y[0]*y[1] + p[1]*y[2],
            -p[0]*y[0]*y[1] + p[1]*y[2],
            p[0]*y[0]*y[1] - p[1]*y[2]
        ], 3, 2, **kwargs)


def _get_nonlin_part_system():
    odesys = _get_nonlin()
    return PartiallySolvedSystem(odesys, lambda x0, y0, p0: {
        odesys.dep[0]: y0[0] + y0[2] - odesys.dep[2]
    })


def _ref_nonlin(y0, k, t):
    X, Y, Z = y0[2], max(y0[:2]), min(y0[:2])
    kf, kb = k
    x0 = Y*kf
    x1 = Z*kf
    x2 = 2*X*kf
    x3 = -kb - x0 - x1
    x4 = -x2 + x3
    x5 = np.sqrt(-4*kf*(X**2*kf + X*x0 + X*x1 + Z*x0) + x4**2)
    x6 = kb + x0 + x1 + x5
    x7 = (x3 + x5)*np.exp(-t*x5)
    x8 = x3 - x5
    return (x4*x8 + x5*x8 + x7*(x2 + x6))/(2*kf*(x6 + x7))


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl'])
def test_PartiallySolvedSystem__symmetricsys__nonlinear(integrator):
    partsys = _get_nonlin_part_system()
    logexp = get_logexp(7, partsys.indep**0/10**7)
    trnsfsys = symmetricsys(logexp, logexp).from_other(partsys)
    y0 = [3., 2., 1.]
    k = [9.351, 2.532]
    tend = 1.7
    atol, rtol = 1e-12, 1e-13
    for odesys, forgive in [(partsys, 21), (trnsfsys, 298)]:
        xout, yout, info = odesys.integrate(tend, y0, k, integrator=integrator,
                                            first_step=1e-14, atol=atol, rtol=rtol,
                                            nsteps=1000)
        assert info['success']
        yref = np.empty_like(yout)
        yref[:, 2] = _ref_nonlin(y0, k, xout - xout[0])
        yref[:, 0] = y0[0] + y0[2] - yref[:, 2]
        yref[:, 1] = y0[1] + y0[2] - yref[:, 2]
        assert np.allclose(yout, yref, atol=forgive*atol, rtol=forgive*rtol)


@requires('sym', 'scipy')
def test_SymbolicSys_from_other():
    scaled = ScaledSys.from_callback(lambda x, y: [y[0]*y[0]], 1,
                                     dep_scaling=101)
    LogLogSys = symmetricsys(get_logexp(), get_logexp())
    transformed_scaled = LogLogSys.from_other(scaled)
    tout = np.array([0, .2, .5])
    y0 = [1.]
    ref, nfo1 = ODESys(lambda x, y: y[0]*y[0]).predefined(
        y0, tout, first_step=1e-14)
    analytic = 1/(1-tout.reshape(ref.shape))
    assert np.allclose(ref, analytic)
    yout, nfo0 = transformed_scaled.predefined(y0, tout+1)
    assert np.allclose(yout, analytic)


@requires('sym', 'scipy')
def test_backend():

    def f(x, y, p, backend=math):
        return [backend.exp(p[0]*y[0])]

    def analytic(x, p, y0):
        # dydt = exp(p*y(t))
        # y(t) = - log(p*(c1-t))/p
        #
        # y(0) = - log(p*c1)/p
        # p*y(0) = -log(p) -log(c1)
        # c1 = exp(-log(p)-p*y(0))
        # c1 =
        #
        # y(t) = -log(p*(exp(-p*y(0))/p - t))/p
        return -np.log(p*(np.exp(-p*y0)/p - x))/p

    y0, tout, p = .07, [0, .1, .2], .3
    ref = analytic(tout, p, y0)

    def _test_odesys(odesys):
        yout, info = odesys.predefined([y0], tout, [p])
        assert np.allclose(yout.flatten(), ref)

    _test_odesys(ODESys(f))
    _test_odesys(SymbolicSys.from_callback(f, 1, 1))


@requires('sym', 'scipy')
def _test_SymbolicSys_from_callback__backend(backend):
    ss = SymbolicSys.from_callback(vdp_f, 2, 1, backend=backend)
    xout, yout, info = ss.integrate([0, 1, 2], [1, 0], params=[2.0])
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    assert info['nfev'] > 0


@requires('sym', 'sympy')
def test_SymbolicSys_from_callback__sympy():
    _test_SymbolicSys_from_callback__backend('sympy')


@requires('sym', 'symengine')
def test_SymbolicSys_from_callback__symengine():
    _test_SymbolicSys_from_callback__backend('symengine')


@requires('sym', 'symcxx')
def test_SymbolicSys_from_callback__symcxx():
    _test_SymbolicSys_from_callback__backend('symcxx')


@pytest.mark.slow
@requires('sym', 'pycvodes', 'pygslodeiv2')
@pytest.mark.parametrize('integrator,method', [('cvode', 'adams'), ('gsl', 'msadams')])
def test_integrate_chained(integrator, method):
    for p in (0, 1, 2, 3):
        n, a = 7, 5
        atol, rtol = 1e-10, 1e-10
        y0, k, linsys = get_special_chain(n, p, a)
        y0 += 1e-10
        LogLogSys = symmetricsys(get_logexp(), get_logexp())
        logsys = LogLogSys.from_other(linsys)
        tout = [10**-12, 1]
        kw = dict(
            integrator=integrator, method=method, atol=atol, rtol=rtol,
        )
        forgive = (5+p)*1.2

        xout, yout, info = integrate_chained([logsys, linsys], {'nsteps': [1, 1]}, tout, y0,
                                             return_on_error=True, **kw)
        assert info['success'] == False  # noqa
        ntot = 400
        nlinear = 60*(p+3)

        xout, yout, info = integrate_chained([logsys, linsys], {
            'nsteps': [ntot - nlinear, nlinear],
            'first_step': [30.0, 1e-5],
            'return_on_error': [True, False]
        }, tout, y0, **kw)
        assert info['success'] == True  # noqa
        check(yout[-1, :], n, p, a, atol, rtol, forgive)


def _test_cetsa(y0, params, extra=False, stepx=1, **kwargs):
    # real-world based test-case
    from ._cetsa import _get_cetsa_odesys
    molar_unitless = 1e9
    t0, tend = 1e-16, 180
    odesys = _get_cetsa_odesys(molar_unitless, False)
    tsys = _get_cetsa_odesys(molar_unitless, True)
    if y0.ndim == 1:
        tout = [t0, tend]
    elif y0.ndim == 2:
        tout = np.asarray([(t0, tend)]*y0.shape[0])

    comb_res = integrate_chained([tsys, odesys], {'nsteps': [500*stepx, 20*stepx]}, tout, y0/molar_unitless, params,
                                 return_on_error=True, autorestart=2, **kwargs)
    if isinstance(comb_res, list):
        for r in comb_res:
            assert r.info['success']
            assert r.info['nfev'] > 10
    else:
        assert comb_res.info['success']
        assert comb_res.info['nfev'] > 10

    if extra:
        with pytest.raises(RuntimeError):  # (failure)
            odesys.integrate(np.linspace(t0, tend, 20), y0/molar_unitless, params, atol=1e-7, rtol=1e-7,
                             nsteps=500, first_step=1e-14, **kwargs)

        res = odesys.integrate(np.linspace(t0, tend, 20), y0/molar_unitless, params, nsteps=int(38*1.1),
                               first_step=1e-14, **kwargs)
        assert np.min(res.yout[-1, :]) < -1e-6  # crazy! (failure of the linear formulation)
        tres = tsys.integrate([t0, tend], y0/molar_unitless, params, nsteps=int(1345*1.1), **kwargs)
        assert tres.info['success'] is True
        assert tres.info['nfev'] > 100


@pytest.mark.veryslow
@requires('sym', 'pycvodes', 'pygslodeiv2', 'pyodeint')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl', 'odeint'])
def test_cetsa(integrator):
    _test_cetsa(_cetsa.ys[1], _cetsa.ps[1], integrator=integrator, first_step=1e-14,
                stepx=2 if integrator == 'odeint' else 1)
    if integrator == 'cvode':
        _test_cetsa(_cetsa.ys[0], _cetsa.ps[0], extra=True, integrator=integrator)


@pytest.mark.veryslow
@requires('sym', 'pycvodes', 'pygslodeiv2', 'pyodeint')
@pytest.mark.parametrize('integrator', ['cvode', 'gsl', 'odeint'])
def test_cetsa_multi(integrator):
    _test_cetsa(np.asarray(_cetsa.ys), np.asarray(_cetsa.ps), integrator=integrator, first_step=1e-14,
                stepx=2 if integrator == 'odeint' else 1)


@requires('sym', 'pycvodes')
def test_dep_by_name():
    def _sin(t, y, p):
        return {'prim': y['bis'], 'bis': -p[0]**2 * y['prim']}
    odesys = SymbolicSys.from_callback(_sin, names=['prim', 'bis'], nparams=1, dep_by_name=True)
    A, k = 2, 3
    for y0 in ({'prim': 0, 'bis': A*k}, [0, A*k]):
        xout, yout, info = odesys.integrate(np.linspace(0, 1), y0, [k],
                                            integrator='cvode', method='adams')
        assert info['success']
        assert xout.size > 7
        ref = [
            A*np.sin(k*(xout - xout[0])),
            A*np.cos(k*(xout - xout[0]))*k
        ]
        assert np.allclose(yout[:, 0], ref[0], atol=1e-5, rtol=1e-5)
        assert np.allclose(yout[:, 1], ref[1], atol=1e-5, rtol=1e-5)


def _get_cetsa_isothermal():
    # tests rhs returning dict, integrate with y0 being a dict & multiple solved variables
    names = ('NL', 'N', 'L', 'U', 'A')
    k_names = ('dis', 'as', 'un', 'fo', 'ag')

    def i(n):
        return names.index(n)

    def k(n):
        return k_names.index(n)

    def rhs(x, y, p):
        r = {
            'diss': p['dis']*y['NL'],
            'asso': p['as']*y['N']*y['L'],
            'unfo': p['un']*y['N'],
            'fold': p['fo']*y['U'],
            'aggr': p['ag']*y['U']
        }
        return {
            'NL': r['asso'] - r['diss'],
            'N': r['diss'] - r['asso'] + r['fold'] - r['unfo'],
            'L': r['diss'] - r['asso'],
            'U': r['unfo'] - r['fold'] - r['aggr'],
            'A': r['aggr']
        }

    return SymbolicSys.from_callback(rhs, dep_by_name=True, par_by_name=True, names=names, param_names=k_names)


@requires('sym', 'scipy')
def test_cetsa_isothermal():
    odesys = _get_cetsa_isothermal()
    tout = (0, 300)
    par = {'dis': 10.0, 'as': 1e9, 'un': 0.1, 'fo': 2.0, 'ag': 0.05}
    conc0 = defaultdict(float, {'NL': 1, 'L': 5})
    xout, yout, nfo = odesys.integrate(tout, conc0, par)
    assert nfo['success']


@requires('sym', 'sympy', 'pycvodes')
def test_SymbolicSys__first_step_expr():
    import sympy
    tend, k, y0 = 5, [1e23, 3], (.7, .0, .0)
    kwargs = dict(integrator='cvode', atol=1e-8, rtol=1e-8)
    factory = decay_dydt_factory(k)
    dep = sympy.symbols('y0 y1 y2', real=True)
    exprs = factory(k, dep)
    odesys = SymbolicSys(zip(dep, exprs), jac=True, first_step_expr=dep[0]*1e-30)
    xout, yout, info = odesys.integrate(tend, y0, **kwargs)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, atol=10*kwargs['atol'], rtol=10*kwargs['rtol'])


@pytest.mark.slow
@requires('sym', 'pygslodeiv2')
def test_SymbolicSys__from_callback__first_step_expr():
    tend, k, y0 = 5, [1e23, 3], (.7, .0, .0)
    kwargs = dict(integrator='gsl', atol=1e-8, rtol=1e-8)
    factory = decay_dydt_factory(k)
    odesys = SymbolicSys.from_callback(factory, 3, first_step_factory=lambda x, y, p: y[0]*1e-30)
    xout, yout, info = odesys.integrate(tend, y0, **kwargs)
    ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
    assert np.allclose(yout, ref, atol=10*kwargs['atol'], rtol=10*kwargs['rtol'])


@requires('sym', 'pycvodes')
def test_SymbolicSys__from_callback__first_step_expr__by_name():
    kwargs = dict(integrator='cvode', atol=1e-8, rtol=1e-8)
    names = ['foo', 'bar', 'baz']
    par_names = 'first second third'.split()
    odesys = SymbolicSys.from_callback(
        lambda x, y, p, be: {
            'foo': -p['first']*y['foo'],
            'bar': p['first']*y['foo'] - p['second']*y['bar'],
            'baz': p['second']*y['bar'] - p['third']*y['baz']
        }, names=names, param_names=par_names,
        dep_by_name=True, par_by_name=True,
        first_step_factory=lambda x0, ic: 1e-30*ic['foo'])
    y0 = {'foo': .7, 'bar': 0, 'baz': 0}
    p = {'first': 1e23, 'second': 2, 'third': 3}
    result = odesys.integrate(5, y0, p, **kwargs)
    assert result.info['success']
    ref = bateman_full([y0[k] for k in names], [p[k] for k in par_names], result.xout - result.xout[0], exp=np.exp)
    for i, k in enumerate(odesys.names):
        assert np.allclose(result.named_dep(k), ref[i], atol=10*kwargs['atol'], rtol=10*kwargs['rtol'])
    for k, v in p.items():
        assert result.named_param(k) == v


@requires('sym', 'pyodeint')
def test_PartiallySolvedSystem__by_name():
    k = [math.log(2)/(138.4*24*3600)]
    names = 'Po-210 Pb-206'.split()
    with pytest.raises(ValueError):
        odesys = SymbolicSys.from_callback(decay_dydt_factory({'Po-210': k[0]}, names=names),
                                           dep_by_name=True, par_by_name=True, names=names, param_names=names)
    odesys = SymbolicSys.from_callback(decay_dydt_factory({'Po-210': k[0]}, names=names),
                                       dep_by_name=True, names=names)

    assert odesys.ny == 2
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be=None: {
        odesys['Pb-206']: y0[odesys['Pb-206']] + y0[odesys['Po-210']] - odesys['Po-210']
    })
    assert partsys.free_names == ['Po-210']
    assert partsys.ny == 1
    assert (partsys['Pb-206'] - partsys.init_dep[partsys.names.index('Pb-206')] -
            partsys.init_dep[partsys.names.index('Po-210')] + odesys['Po-210']) == 0
    duration = 7*k[0]
    atol, rtol, forgive = 1e-9, 1e-9, 10
    y0 = [1e-20]*(len(k)+1)
    y0[0] = 1
    for system in (odesys, partsys):
        xout, yout, info = system.integrate(duration, y0, integrator='odeint', rtol=rtol, atol=atol)
        ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref, rtol=rtol*forgive, atol=atol*forgive)
        assert yout.shape[1] == 2
        assert xout.shape[0] == yout.shape[0]
        assert yout.ndim == 2 and xout.ndim == 1


@requires('sym', 'pycvodes')
def test_PartiallySolvedSystem__by_name_2():
    yn, pn = 'x y z'.split(), 'p q r'.split()
    odesys = _get_decay3_names(yn, pn)
    partsys = PartiallySolvedSystem(odesys, lambda x0, y0, p0, be: {
        odesys['x']: y0[odesys['x']]*be.exp(-p0['p']*(odesys.indep-x0))
    })
    y0 = [3, 2, 1]
    k = [3.5, 2.5, 1.5]

    def _check(res):
        ref = np.array(bateman_full(y0, k, res.xout - res.xout[0], exp=np.exp)).T
        assert res.info['success']
        assert np.allclose(res.yout, ref)
    args = [0, 1], dict(zip(yn, y0)), dict(zip(pn, k))
    kwargs = dict(integrator='cvode')
    _check(odesys.integrate(*args, **kwargs))
    _check(partsys.integrate(*args, **kwargs))
    scaledsys = ScaledSys.from_other(partsys, dep_scaling=42, indep_scaling=17)
    _check(scaledsys.integrate(*args, **kwargs))


@requires('sym')
def test_symmetricsys__invariants():
    yn, pn = 'x y z'.split(), 'a b'.split()
    odesys = SymbolicSys.from_callback(
        lambda t, y, p: {
            'x': -p['a']*y['x'],
            'y': -p['b']*y['y'] + p['a']*y['x'],
            'z': p['b']*y['y']
        }, names=yn, param_names=pn, dep_by_name=True, par_by_name=True,
        linear_invariants=[[1, 1, 1]], linear_invariant_names=['mass-conservation'],
        indep_name='t')
    assert odesys.linear_invariants.tolist() == [[1, 1, 1]]
    assert odesys.linear_invariant_names == ['mass-conservation']
    assert odesys.nonlinear_invariants is None
    assert odesys.nonlinear_invariant_names is None

    logexp = get_logexp()
    LogLogSys = symmetricsys(logexp, logexp)
    tsys = LogLogSys.from_other(odesys)
    assert tsys.linear_invariants is None
    assert tsys.linear_invariant_names is None
    assert len(tsys.nonlinear_invariants) == 1
    E = odesys.be.exp
    assert tsys.nonlinear_invariants[0] - sum(E(odesys[k]) for k in yn) == 0
    assert tsys.nonlinear_invariant_names == ['mass-conservation']


@requires('sym', 'pycvodes')
def test_SymbolicSys__roots():
    def f(t, y):
        return [y[0]]

    def roots(t, y, p, backend):
        return [y[0] - backend.exp(1)]
    odesys = SymbolicSys.from_callback(f, 1, roots_cb=roots)
    kwargs = dict(first_step=1e-12, atol=1e-12, rtol=1e-12, method='adams', integrator='cvode')
    xout, yout, info = odesys.integrate(2, [1], **kwargs)
    assert len(info['root_indices']) == 1
    assert np.min(np.abs(xout - 1)) < 1e-11


@requires('sym', 'pyodeint')
@pytest.mark.parametrize('method', ['bs', 'rosenbrock4'])
def test_SymbolicSys__reference_parameters_using_symbols(method):
    be = sym.Backend('sympy')
    x, p = map(be.Symbol, 'x p'.split())
    symsys = SymbolicSys([(x, -p*x)], params=True)
    tout = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1]
    for y_symb in [False, True]:
        for p_symb in [False, True]:
            xout, yout, info = symsys.integrate(
                tout, {x: 2} if y_symb else [2], {p: 3} if p_symb else [3],
                method=method, integrator='odeint', atol=1e-12, rtol=1e-12)
            assert np.allclose(yout[:, 0], 2*np.exp(-3*xout))


@requires('sym', 'pygslodeiv2')
@pytest.mark.parametrize('method', ['rkck', 'rk4imp'])
def test_SymbolicSys__reference_parameters_using_symbols_from_callback(method):
    be = sym.Backend('sympy')
    k = be.Symbol('p')

    def dydt(t, y):       # external symbolic parameter 'k', should be allowed
        return [-k*y[0]]  # even though reminiscent of global variables.

    odesys1 = SymbolicSys.from_callback(dydt, 1, backend=be, params=True)
    odesys2 = SymbolicSys.from_callback(dydt, 1, backend=be, par_by_name=True, param_names=[], params=True)
    tout = [0, 1e-9, 1e-7, 1e-5, 1e-3, 0.1]
    for symsys in (odesys1, odesys2):
        for y_symb in [False, True]:
            for p_symb in [False, True]:
                xout, yout, info = symsys.integrate(
                    tout, {symsys.dep[0]: 2} if y_symb else [2], {k: 3} if p_symb else [3],
                    method=method, integrator='gsl', atol=1e-12, rtol=1e-12)
                assert xout.size > 4
                assert np.allclose(yout[:, 0], 2*np.exp(-3*xout))


@pytest.mark.slow
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('scaled', [False, True])
def test_PartiallySolvedSystem__from_linear_invariants(scaled):
    atol, rtol, forgive = 1e-11, 1e-11, 20
    k = [7., 3, 2]
    _ss = SymbolicSys.from_callback(decay_rhs, len(k)+1, len(k),
                                    linear_invariants=[[1]*(len(k)+1)],
                                    linear_invariant_names=['tot_amount'])
    if scaled:
        ss = ScaledSys.from_other(_ss, dep_scaling=1e3)
    else:
        ss = _ss

    y0 = [0]*(len(k)+1)
    y0[0] = 1

    def check_formulation(odesys):
        xout, yout, info = odesys.integrate(
            [0, 1], y0, k, integrator='cvode', atol=atol, rtol=rtol, nsteps=800)
        ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref, rtol=rtol*forgive, atol=atol*forgive)

    check_formulation(ss)

    ps = PartiallySolvedSystem.from_linear_invariants(ss)
    assert ps.ny == ss.ny - 1
    check_formulation(ps)

    ps2 = PartiallySolvedSystem(ss, lambda x0, y0, p0, be: {
        ss.dep[0]: y0[0]*be.exp(-p0[0]*(ss.indep-x0))})
    assert ps2.ny == ss.ny - 1
    check_formulation(ps2)


@requires('sym', 'pyodeint')
def test_PartiallySolvedSystem__by_name__from_linear_invariants():
    k = [math.log(2)/(138.4*24*3600)]
    names = 'Po-210 Pb-206'.split()
    odesys = SymbolicSys.from_callback(
        decay_dydt_factory({'Po-210': k[0]}, names=names),
        dep_by_name=True, names=names, linear_invariants=[[1, 1]])
    assert odesys.ny == 2
    partsys1 = PartiallySolvedSystem.from_linear_invariants(odesys)
    partsys2 = PartiallySolvedSystem.from_linear_invariants(odesys, ['Pb-206'])
    partsys3 = PartiallySolvedSystem.from_linear_invariants(odesys, ['Po-210'])

    assert partsys1.free_names in (['Po-210'], ['Pb-206'])
    assert partsys2.free_names == ['Po-210']
    assert partsys3.free_names == ['Pb-206']
    assert partsys1.ny == partsys2.ny == partsys3.ny == 1

    assert (partsys2['Pb-206'] - partsys2.init_dep[partsys2.names.index('Pb-206')] -
            partsys2.init_dep[partsys2.names.index('Po-210')] + odesys['Po-210']) == 0
    duration = 7*k[0]
    atol, rtol, forgive = 1e-9, 1e-9, 10
    y0 = [1e-20]*(len(k)+1)
    y0[0] = 1
    for system in (odesys, partsys1, partsys2, partsys3):
        xout, yout, info = system.integrate(duration, y0, integrator='odeint', rtol=rtol, atol=atol)
        ref = np.array(bateman_full(y0, k+[0], xout - xout[0], exp=np.exp)).T
        assert np.allclose(yout, ref, rtol=rtol*forgive, atol=atol*forgive)
        assert yout.shape[1] == 2
        assert xout.shape[0] == yout.shape[0]
        assert yout.ndim == 2 and xout.ndim == 1


@requires('sym')
def test_SymbolicSys__indep_in_exprs():
    def dydt(t, y, p):
        return [t*p[0]*y[0]]
    be = sym.Backend('sympy')
    t, y, p = map(be.Symbol, 't y p'.split())
    odesys = SymbolicSys([(y, dydt(t, [y], [p])[0])], t, params=True)
    fout = odesys.f_cb(2, [3], [4])
    assert len(fout) == 1
    assert abs(fout[0] - 2*3*4) < 1e-14


@requires('sym', 'pycvodes')
@pytest.mark.parametrize('idx', [0, 1, 2])
def test_PartiallySolvedSystem__roots(idx):
    import sympy as sp
    t, x, y, z, p, q = sp.symbols('t x y z, p, q')
    odesys = SymbolicSys({x: -p*x, y: p*x - q*y, z: q*y}, t, params=(p, q), roots=([x - y], [x - z], [y - z])[idx])
    _p, _q, tend = 7, 3, 0.7
    dep0 = {x: 1, y: 0, z: 0}
    ref = [0.11299628093544488, 0.20674119231833346, 0.3541828705348678]  # determined in notebook:
    # test_symbolic__test_PartiallySolvedSystem__roots.ipynb

    def check(odesys):
        res = odesys.integrate(tend, [dep0[k] for k in getattr(odesys, 'original_dep', odesys.dep)], (_p, _q),
                               integrator='cvode', return_on_root=True)
        assert abs(res.xout[-1] - ref[idx]) < 1e-7

    check(odesys)
    psys = PartiallySolvedSystem(odesys, lambda t0, xyz, par0: {x: xyz[odesys.dep.index(x)]*sp.exp(-p*(t-t0))})
    check(psys)


@pytest.mark.slow
@requires('sym', 'pycvodes')
@pytest.mark.parametrize('idx1,idx2,scaled,b2', product([0, 1, 2], [0, 1, 2], [True, False], [None, 0]))
def test_TransformedSys__roots(idx1, idx2, scaled, b2):
    def f(x, y, p):
        return [-p[0]*y[0], p[0]*y[0] - p[1]*y[1], p[1]*y[1]]

    def roots(x, y):
        return ([y[0] - 3*y[1]], [y[0] - 3*y[2]], [3*y[1] - y[2]])[idx1]

    if scaled:
        orisys = SymbolicSys.from_callback(f, 3, 2, roots_cb=roots)
    else:
        orisys = ScaledSys.from_callback(f, 3, 2, roots_cb=roots, dep_scaling=42)
    _p, _q, tend = 7, 3, 0.7
    dep0 = (1, .1, 0)
    ref = [0.02969588399749174, 0.1241509730780618, 0.6110670818418275]  # determined in notebook:
    # test_symbolic__test_PartiallySolvedSystem__roots.ipynb

    def check(odesys):
        res = odesys.integrate(tend, dep0, (_p, _q),
                               integrator='cvode', return_on_root=True)
        assert abs(res.xout[-1] - ref[idx1]) < 6e-7

    logexp = get_logexp(1, 1e-20, b2=None)
    LogLogSys = symmetricsys(logexp, logexp, check_transforms=False)

    if idx2 == 0:  # no need to rerun this code more than once
        check(orisys)
        loglog = LogLogSys.from_other(orisys)
        check(loglog)

        psys1 = PartiallySolvedSystem(orisys, lambda t0, xyz, par0, be: {
            orisys.dep[0]: xyz[0]*be.exp(-par0[0]*(orisys.indep-t0))})
        check(psys1)
        ploglog1 = LogLogSys.from_other(psys1)
        check(ploglog1)

    psys2 = PartiallySolvedSystem(orisys, lambda t0, iv, p0: {
        orisys.dep[idx2]: iv[0] + iv[1] + iv[2] - sum(orisys.dep[j] for j in range(3) if j != idx2)
    })
    ploglog2 = LogLogSys.from_other(psys2)
    check(ploglog2)


@requires('sym', 'sympy')
def test__group_invariants():
    be = sym.Backend('sympy')
    x, y, z = symbs = be.symbols('x y z')

    coeff1 = [3, 2, -1]
    expr1 = 3*x + 2*y - z
    lin, nonlin = _group_invariants([expr1], symbs, be)
    assert lin == [coeff1]
    assert nonlin == []

    expr2 = 3*x*x + 2*y - z
    lin, nonlin = _group_invariants([expr2], symbs, be)
    assert lin == []
    assert nonlin == [expr2]

    lin, nonlin = _group_invariants([expr1, expr2], symbs, be)
    assert lin == [coeff1]
    assert nonlin == [expr2]

    lin, nonlin = _group_invariants([x + be.exp(y)], symbs, be)
    assert lin == []
    assert nonlin == [x+be.exp(y)]


@requires('sym', 'pycvodes')
def test_SymbolicSys_as_autonomous():
    import sympy

    def rhs(t, y, p, backend=math):
        return [y[1], backend.sin(t)-p[0]*y[0]]
    odesys = SymbolicSys.from_callback(rhs, 2, 1)

    def analytic(tout, init_y, p):
        t, (k,) = odesys.indep, odesys.params
        c1, c2 = sympy.symbols('c1 c2')
        sqk = sympy.sqrt(k)
        f = c1*sympy.cos(sqk*t) + c2*sympy.sin(sqk*t) + sympy.sin(t)/(k-1)
        dfdt = f.diff(t)
        t0 = tout[0]
        sol, = sympy.solve([f.subs(t, t0) - init_y[0],
                           dfdt.subs(t, t0) - init_y[1]],
                           [c1, c2], dict=True)
        sol[k] = p[0]
        exprs = [f.subs(sol), dfdt.subs(sol)]
        cb = sympy.lambdify([t], exprs)
        return np.array(cb(tout)).T

    def integrate_and_check(system):
        init_y = [0, 0]
        p = [2]
        result = system.integrate([0, 80], init_y, p, integrator='cvode', nsteps=5000)
        yref = analytic(result.xout, init_y, p)
        assert np.all(result.yout - yref < 1.6e-5)

    integrate_and_check(odesys)
    assert len(odesys.dep) == 2
    assert not odesys.autonomous_interface
    assert not odesys.autonomous_exprs
    asys = odesys.as_autonomous()
    integrate_and_check(asys)
    assert len(asys.dep) == 3
    assert not asys.autonomous_interface
    assert asys.autonomous_exprs


@requires('sym', 'pycvodes')
def test_SymbolicSys_as_autonomous__linear_invariants():
    def rhs(t, y, p):
        k = t**p[0]
        return [-k*y[0], k*y[0]]

    def analytic(tout, init_y, params):
        y0ref = init_y[0]*np.exp(-tout**(params[0]+1)/(params[0]+1))
        return np.array([y0ref, init_y[0] - y0ref + init_y[1]]).T

    odes = SymbolicSys.from_callback(rhs, 2, 1, linear_invariants=[[1, 1]])
    for odesys in [odes, odes.as_autonomous()]:
        result = odesys.integrate(4, [5, 2], [3], integrator='cvode')
        ref = analytic(result.xout, result.yout[0, :], result.params)
        assert np.allclose(result.yout, ref, atol=1e-6)

        invar_viol = result.calc_invariant_violations()
        assert np.allclose(invar_viol, 0)


@requires('sym', 'pycvodes')
def test_SymbolicSys__by_name__as_autonomous():
    def f(t, y, p):
        k = t**p['e']
        return {
            'a': -k*y['a'],
            'b': +k*y['a']
        }

    def analytic(tout, init_y, p):
        y0ref = init_y[0]*np.exp(-tout**(p[0]+1)/(p[0]+1))
        return np.array([y0ref, init_y[0] - y0ref + init_y[1]]).T

    odes = SymbolicSys.from_callback(
        f, names='ab', param_names='e', dep_by_name=True, par_by_name=True,
        linear_invariants=[{'a': 1, 'b': 1}]
    )

    for odesys in [odes, odes.as_autonomous()]:
        result = odesys.integrate(3, {'a': 2, 'b': 1}, {'e': 2}, integrator='cvode')
        ref = analytic(result.xout, result.yout[0, :], result.params)
        assert np.allclose(result.yout, ref, atol=1e-6)

        invar_viol = result.calc_invariant_violations()
        assert np.allclose(invar_viol, 0)


@requires('sym', 'pycvodes')
def test_SymbolicSys_as_autonomous__scaling():

    # 2 HNO2 -> H2O + NO + NO2; MassAction(EyringHS.fk('dH1', 'dS1'))
    # 2 NO2 -> N2O4; MassAction(EyringHS.fk('dH2', 'dS2'))
    #
    # HNO2 H2O NO NO2 N2O4
    def get_odesys(scaling=1):
        def rhs(t, y, p, backend=math):
            HNO2, H2O, NO, NO2, N2O4 = y
            dH1, dS1, dH2, dS2 = p
            R = 8.314
            T = 300 + 10*backend.sin(0.2*math.pi*t - math.pi/2)
            kB_h = 2.08366e10
            k1 = kB_h*T*backend.exp(dS1/R - dH1/(R*T))/scaling  # bimolecular => scaling**-1
            k2 = kB_h*T*backend.exp(dS2/R - dH2/(R*T))/scaling  # bimolecular => scaling**-1
            r1 = k1*HNO2**2
            r2 = k2*NO2**2
            return [-2*r1, r1, r1, r1 - 2*r2, r2]

        return SymbolicSys.from_callback(rhs, 5, 4, names='HNO2 H2O NO NO2 N2O4'.split(),
                                         param_names='dH1 dS1 dH2 dS2'.split())

    def check(system, scaling=1):
        init_y = [1*scaling, 55*scaling, 0, 0, 0]
        p = [85e3, 10, 70e3, 20]
        return system.integrate(np.linspace(0, 60, 200), init_y, p, integrator='cvode', nsteps=5000)

    def compare_autonomous(scaling):
        odesys = get_odesys(scaling)
        autsys = odesys.as_autonomous()
        res1 = check(odesys, scaling=scaling)
        res2 = check(autsys, scaling=scaling)
        assert np.allclose(res1.yout, res2.yout, atol=1e-6)

    compare_autonomous(1)
    compare_autonomous(1000)
