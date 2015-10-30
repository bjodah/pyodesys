import pytest
import numpy as np
from .. import OdeSys


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
    xout, yout, info = odes.integrate_scipy([0, 1, 2], [1, 0], params=[2.0])
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(yout, ref)
    assert info['nrhs'] > 0


@pytest.mark.parametrize('solver', ['scipy', 'gsl', 'cvode', 'odeint'])
def test_adaptive(solver):
    odes = OdeSys(vdp_f, vdp_j, vdp_dfdt)
    kwargs = dict(params=[2.0])
    xout, yout, info = odes.adaptive(solver, [1, 0], 0, 2, **kwargs)
    # something is off with odeint (it looks as it is the step
    # size control which is too optimistic).
    assert np.allclose(yout[-1, :], [-1.89021896, -0.71633577],
                       rtol=.2 if solver == 'odeint' else 1e-5)


@pytest.mark.parametrize('solver', ['scipy', 'gsl', 'odeint', 'cvode'])
def test_predefined(solver):
    odes = OdeSys(vdp_f, vdp_j, vdp_dfdt)
    xout = [0, 0.7, 1.3, 2]
    yout, info = odes.predefined(solver, [1, 0], xout, params=[2.0])
    assert np.allclose(yout[-1, :], [-1.89021896, -0.71633577])
