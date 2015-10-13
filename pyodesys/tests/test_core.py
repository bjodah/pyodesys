import numpy as np
from .. import OdeSys


def vdp_f(t, y, mu):
    return [y[1], -y[0] + mu*y[1]*(1 - y[0]**2)]


def vdp_j(t, y, mu):
    Jmat = np.zeros((2,2))
    Jmat[0, 0] = 0
    Jmat[0, 1] = 1
    Jmat[1, 0] = -1 - mu*2*y[1]*y[0]
    Jmat[1, 1] = mu*(1 - y[0]**2)
    return Jmat


def test_params():
    odes = OdeSys(vdp_f, vdp_j)
    out = odes.integrate_scipy([0, 1, 2], [1, 0], params=2.0)
    # blessed values:
    ref = [[1, 0], [0.44449086, -1.32847148], [-1.89021896, -0.71633577]]
    assert np.allclose(out[:, 1:], ref)
