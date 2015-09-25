from __future__ import print_function, absolute_import, division

import numpy as np
import sympy as sp

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


def test_OdeSystem__from_callback():
    # Decay chain of 3 species (2 decays)
    # A --[k0=4]--> B --[k1=3]--> C
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    atol, rtol = 1e-11, 1e-11
    odesys = OdeSystem.from_callback(decay_dydt_factory(k), len(k)+1)
    out = odesys.integrate_scipy(tend, y0, atol=atol, rtol=rtol)
    ref = out.copy()
    ref[:, 1:] = np.array(bateman_full(y0, k+[0], ref[:, 0],
                                       exp=np.exp)).T
    assert np.allclose(out, ref, rtol=rtol, atol=atol)


def test_OdeSystem():
    tend, k, y0 = 2, [4, 3], (5, 4, 2)
    y = sp.symarray('y', len(k)+1)
    dydt = decay_dydt_factory(k)
    f = dydt(0, y)
    odesys = OdeSystem(zip(y, f))
    out = odesys.integrate_scipy(tend, y0)
    ref = out.copy()
    ref[:, 1:] = np.array(bateman_full(y0, k+[0], ref[:, 0],
                                       exp=np.exp)).T
    assert np.allclose(out, ref)
