# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
import numpy as np
from .. import OdeSys

try:
    import dill
except ImportError:
    dill = None


def vdp_f(t, y, p):
    return [y[1], -y[0] + p[0]*y[1]*(1 - y[0]**2)]


def vdp_dfdt(t, y, p):
    return [0, 0]


def vdp_j(t, y, p):
    Jmat = np.zeros((2, 2))
    Jmat[0, 0] = 0
    Jmat[0, 1] = 1
    Jmat[1, 0] = -1 - p[0]*2*y[1]*y[0]
    Jmat[1, 1] = p[0]*(1 - y[0]**2)
    return Jmat


def test_params():
    odes = OdeSys(vdp_f, vdp_j)
    xout, yout, info = odes.integrate([0, 1, 2], [1, 0], params=[2.0],
                                      integrator='scipy')
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    assert info['nfev'] > 0


@pytest.mark.parametrize('solver', ['scipy', 'gsl', 'cvode', 'odeint'])
def test_adaptive(solver):
    odes = OdeSys(vdp_f, vdp_j, vdp_dfdt)
    kwargs = dict(params=[2.0])
    xout, yout, info = odes.adaptive([1, 0], 0, 2, integrator=solver, **kwargs)
    # something is off with odeint (it looks as it is the step
    # size control which is too optimistic).
    assert np.allclose(yout[-1, :], [-1.89021896, -0.71633577],
                       rtol=.2 if solver == 'odeint' else 1e-5)


@pytest.mark.parametrize('solver', ['scipy', 'gsl', 'odeint', 'cvode'])
def test_predefined(solver):
    odes = OdeSys(vdp_f, vdp_j, vdp_dfdt)
    xout = [0, 0.7, 1.3, 2]
    yout, info = odes.predefined([1, 0], xout, params=[2.0], integrator=solver)
    assert np.allclose(yout[-1, :], [-1.89021896, -0.71633577])


def test_pre_post_processors():
    """
    y(x) = A * exp(-k * x)
    dy(x)/dx = -k * A * exp(-k * x) = -k * y(x)

    First transformation:
    v = y/y(x0)
    u = k*x
    ===>
        v(u) = exp(-u)
        dv(u)/du = -v(u)

    Second transformation:
    s = ln(v)
    r = u
    ===>
        s(r) = -r
        ds(r)/dr = -1
    """
    def pre1(x, y, p):
        return x*p[0], y/y[0], [p[0], y[0]]

    def post1(x, y, p):
        return x/p[0], y*p[1], [p[0]]

    def pre2(x, y, p):
        return x, np.log(y), p

    def post2(x, y, p):
        return x, np.exp(y), p

    def dsdr(x, y, p):
        return [-1]

    odesys = OdeSys(dsdr, pre_processors=(pre1, pre2),
                    post_processors=(post2, post1))
    k = 3.7
    A = 42
    tend = 7
    xout, yout, info = odesys.integrate(np.asarray([0, tend]), np.asarray([A]),
                                        [k], atol=1e-12, rtol=1e-12,
                                        name='vode', method='adams')
    yref = A*np.exp(-k*xout)
    assert np.allclose(yout.flatten(), yref)
    assert np.allclose(odesys.internal_yout.flatten(), -odesys.internal_xout)


def test_custom_module():
    from pyodesys.integrators import RK4_example_integartor
    odes = OdeSys(vdp_f, vdp_j)
    xout, yout, info = odes.integrate(
        [0, 2], [1, 0], params=[2.0], integrator=RK4_example_integartor,
        first_step=1e-2)
    # blessed values:
    assert np.allclose(yout[0], [1, 0])
    assert np.allclose(yout[-1], [-1.89021896, -0.71633577])
    assert info['nfev'] == 4*2/1e-2

    xout, yout, info = odes.integrate(
        np.linspace(0, 2, 150), [1, 0], params=[2.0],
        integrator=RK4_example_integartor)

    assert np.allclose(yout[0], [1, 0])
    assert np.allclose(yout[-1], [-1.89021896, -0.71633577])
    assert info['nfev'] == 4*149


@pytest.mark.skipif(dill is None, reason="Test requires dill to be installed")
def test_OdeSys_serialisation():
    _odes = OdeSys(vdp_f, vdp_j)
    s = dill.dumps(_odes)
    odes = dill.loads(s)
    xout, yout, info = odes.integrate([0, 1, 2], [1, 0], params=[2.0],
                                      integrator='scipy')
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    assert info['nfev'] > 0


@pytest.mark.skipif(dill is None, reason="Test requires dill to be installed")
@pytest.mark.parametrize('integrator', ['scipy'])  # 'cvode', 'odeint', 'gsl'
def test_integrate_multiple(integrator):
    # Unofficial API
    odes = OdeSys(vdp_f, vdp_j)
    results = odes._integrate_multiple(
        [0, 2], [[0.5, 0.5], [1, 0], [0, 1]],
        params=([1.0], [2.0], [3.0]),
        integrator=integrator)
    xout, yout, info = results[1][0]
    assert np.allclose(yout[-1, :], [-1.89021896, -0.71633577])
