# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np

import sympy
import pytest

from .. import ODESys
from ..core import integrate_chained
from ..symbolic import SymbolicSys, PartiallySolvedSystem, symmetricsys
from ..util import requires
from ._robertson import run_integration, get_ode_exprs

_yref_1e11 = (0.2083340149701255e-7, 0.8333360770334713e-13, 0.9999999791665050)


@requires('sym', 'sympy', 'pyodeint')
def test_run_integration():
    xout, yout, info = run_integration(integrator='odeint')
    assert info['success'] is True


def _test_goe(symbolic=False, reduced=0, extra_forgive=1, logc=False,
              logt=False, zero_conc=0, zero_time=0, nonnegative=None,
              atol=1e-14, rtol=1e-10, integrator='cvode', nsteps=6000, **kwargs):

    ny, nk = 3, 3
    k = (.04, 1e4, 3e7)
    y0 = (1, zero_conc, zero_conc)
    t0, tend = zero_time, 1e11
    tot0 = np.sum(y0)
    kw = dict(integrator=integrator, atol=atol, rtol=rtol, nsteps=nsteps)
    kw.update(kwargs)

    atol_forgive = {
        0: 5,
        1: 15000,
        2: 7,
        3: 4
    }

    if symbolic:
        _s = SymbolicSys.from_callback(get_ode_exprs(logc=False, logt=False)[0], ny, nk,
                                       nonnegative=nonnegative)
        logexp = (sympy.log, sympy.exp)

        if reduced:
            other1, other2 = [_ for _ in range(3) if _ != (reduced-1)]
            s = PartiallySolvedSystem(_s, lambda x0, y0, p0: {
                _s.dep[reduced-1]: y0[0] + y0[1] + y0[2] - _s.dep[other1] - _s.dep[other2]
            })
        else:
            s = _s

        if logc or logt:
            SS = symmetricsys(logexp if logc else None, logexp if logt else None)
            s = SS.from_other(s)

    else:
        f, j = get_ode_exprs(logc=logc, logt=logt, reduced=reduced)
        if reduced:
            ny -= 1
            k += y0
            y0 = [y0[idx] for idx in range(3) if idx != reduced - 1]

        s = ODESys(f, j, autonomous_interface=not logt)

        if logc:
            y0 = np.log(y0)
        if logt:
            t0 = np.log(t0)
            tend = np.log(tend)

    x, y, i = s.integrate((t0, tend), y0, k, **kw)
    assert i['success'] is True
    if logc and not symbolic:
        y = np.exp(y)
    if reduced and not symbolic:
        y = np.insert(y, reduced-1, tot0 - np.sum(y, axis=1), axis=1)
    assert np.allclose(_yref_1e11, y[-1, :],
                       atol=kw['atol']*atol_forgive[reduced]*extra_forgive,
                       rtol=kw['rtol'])


@requires('sym', 'sympy', 'pycvodes')
def test_get_ode_exprs_symbolic():
    _test_goe(symbolic=True, logc=True, logt=False, zero_conc=1e-20,
              atol=1e-8, rtol=1e-10, extra_forgive=2)
    _test_goe(symbolic=True, logc=True, logt=True, zero_conc=1e-20, zero_time=1e-12,
              atol=1e-8, rtol=1e-12, extra_forgive=2)
    _test_goe(symbolic=True, logc=False, logt=True, zero_conc=0, zero_time=1e-12,
              atol=1e-8, rtol=1e-12, extra_forgive=0.4)
    for reduced in range(4):
        _test_goe(symbolic=True, reduced=reduced)
        if reduced != 2:
            _test_goe(symbolic=True, reduced=reduced, logc=True, logt=False, zero_conc=1e-16,
                      atol=1e-8, rtol=1e-10, extra_forgive=2)
        if reduced == 3:
            _test_goe(symbolic=True, reduced=reduced, logc=True, logt=True, zero_conc=1e-18,
                      zero_time=1e-12, atol=1e-12, rtol=1e-10, extra_forgive=1e-4)  # note extra_forgive

        if reduced != 3:
            _test_goe(symbolic=True, reduced=reduced, logc=False, logt=True, zero_time=1e-12,
                      atol=1e-8, rtol=1e-10, extra_forgive=1)

            _test_goe(symbolic=True, reduced=reduced, logc=False, logt=True, zero_time=1e-9, atol=1e-13, rtol=1e-14)


@requires('sym', 'sympy', 'pycvodes')
def test_get_ode_exprs_ODESys():
    _test_goe(symbolic=False, logc=True, logt=False, zero_conc=1e-20,
              atol=1e-8, rtol=1e-10, extra_forgive=2)
    _test_goe(symbolic=False, logc=True, logt=True, zero_conc=1e-20, zero_time=1e-12,
              atol=1e-8, rtol=1e-12, extra_forgive=2)
    _test_goe(symbolic=False, logc=False, logt=True, zero_conc=0, zero_time=1e-12,
              atol=1e-8, rtol=1e-12, extra_forgive=0.4)
    for reduced in range(4):
        _test_goe(symbolic=False, reduced=reduced, extra_forgive=3)
        if reduced != 2:
            _test_goe(symbolic=False, reduced=reduced, logc=True, logt=False, zero_conc=1e-18,
                      atol=1e-8, rtol=1e-10, extra_forgive=2)
        if reduced == 3:
            _test_goe(symbolic=False, reduced=reduced, logc=True, logt=True, zero_conc=1e-18,
                      zero_time=1e-12, atol=1e-12, rtol=1e-12, extra_forgive=1e-8)  # note extra_forgive

        _test_goe(symbolic=False, reduced=reduced, logc=False, logt=True, zero_time=1e-12,
                  atol=1e-8, rtol=1e-10, extra_forgive=1, nonnegative=True)  # tests RecoverableError

        _test_goe(symbolic=False, reduced=reduced, logc=False, logt=True, zero_time=1e-9, atol=1e-13, rtol=1e-14)


@requires('sym', 'sympy', 'pycvodes')
@pytest.mark.parametrize('reduced_nsteps', [
    (0, [(1, 1705*1.01), (4988*1.01, 1), (100, 1513*1.01), (4988*0.69, 1705*0.69)]),  # pays off in steps!
    (1, [(1, 1563), (100, 1600)]),  # worse than using nothing
    (2, [(1, 1674), (100, 1597*1.01)]),  # no pay back
    (3, [(1, 1591*1.01), (4572*1.01, 1), (100, 1545), (4572*0.66, 1591*0.66)])  # no pay back
])
def test_integrate_chained_robertson(reduced_nsteps):
    rtols = {0: 0.02, 1: 0.1, 2: 0.02, 3: 0.015}
    odes = logsys, linsys = [ODESys(*get_ode_exprs(l, l, reduced=reduced_nsteps[0])) for l in [True, False]]

    def pre(x, y, p):
        return np.log(x), np.log(y), p

    def post(x, y, p):
        return np.exp(x), np.exp(y), p

    logsys.pre_processors = [pre]
    logsys.post_processors = [post]
    zero_time, zero_conc = 1e-10, 1e-18
    init_conc = (1, zero_conc, zero_conc)
    k = (.04, 1e4, 3e7)
    for nsteps in reduced_nsteps[1]:
        y0 = [_ for i, _ in enumerate(init_conc) if i != reduced_nsteps[0] - 1]
        x, y, nfo = integrate_chained(odes, {'nsteps': nsteps, 'return_on_error': [True, False]}, (zero_time, 1e11),
                                      y0, k+init_conc, integrator='cvode', atol=1e-10, rtol=1e-14)
        if reduced_nsteps[0] > 0:
            y = np.insert(y, reduced_nsteps[0]-1, init_conc[0] - np.sum(y, axis=1), axis=1)
        assert np.allclose(_yref_1e11, y[-1, :], atol=1e-16, rtol=rtols[reduced_nsteps[0]])
        assert nfo['success'] is True
        assert nfo['nfev'] > 100
        assert nfo['njev'] > 10

    with pytest.raises(KeyError):
        nfo['asdjklda']


@requires('sym', 'sympy', 'pycvodes')
def test_integrate_chained_multi_robertson():
    odes = logsys, linsys = [ODESys(*get_ode_exprs(l, l)) for l in [True, False]]

    def pre(x, y, p):
        return np.log(x), np.log(y), p

    def post(x, y, p):
        return np.exp(x), np.exp(y), p

    logsys.pre_processors = [pre]
    logsys.post_processors = [post]
    zero_time, zero_conc = 1e-10, 1e-18
    init_conc = (1, zero_conc, zero_conc)
    k = (.04, 1e4, 3e7)

    for sys_iter, kw in [(odes, {'nsteps': [100, 1513*1.01], 'return_on_error': [True, False]}),
                         (odes[1:], {'nsteps': [1705*1.01]})]:
        _x, _y, _nfo = integrate_chained(
            sys_iter, kw, [(zero_time, 1e11)]*3,
            [init_conc]*3, [k+init_conc]*3, integrator='cvode', atol=1e-10, rtol=1e-14)
        assert len(_x) == 3 and len(_y) == 3 and len(_nfo) == 3
        for x, y, nfo in zip(_x, _y, _nfo):
            assert np.allclose(_yref_1e11, y[-1, :], atol=1e-16, rtol=0.02)
            assert nfo['success'] is True
            assert nfo['nfev'] > 100
            assert nfo['njev'] > 10
            assert nfo['nsys'] in (1, 2)
