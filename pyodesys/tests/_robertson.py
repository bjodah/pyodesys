# -*- coding: utf-8 -*-

"""
Robertson's stiff chemical kinetics problem (stiff IVP problem)
---------------------------------------------------------------
The fruit fly of chemical kinetics.

Reactions:
    A -> B
B + C -> A + C
B + B -> C + B

Corresponding rates:
r1 = k1*A
r2 = k2*B*C
r3 = k3*B*B

Formulation of system of 3 ODEs for A, B and C:
dA/dt = -r1 + r2
dB/dt = r1 - r2 - r3
dC/dt = r3

Conservation of mass gives us an invariant:
A + B + C = A0 + B0 + C0
=> A = A0 + B0 + C0 - B - C

References
----------
H. H. Robertson, The solution of a set of reaction rate equations, in Numerical
Analysis: An Introduction, J. Walsh, ed., Academic Press, 1966, pp. 178-182.

"""

from __future__ import division, print_function, absolute_import

import math

import numpy as np
import sympy as sp

from ..symbolic import ScaledSys


def get_ode_exprs(logc=False, logt=False, reduced=0):
    """
    reduced:
    0: A, B, C
    1: B, C
    2: A, C
    3: A, B
    """
    if reduced not in (0, 1, 2, 3):
        raise NotImplementedError("What invariant did you have in mind?")

    def dydt(x, y, p, backend=math):
        exp = backend.exp
        k1, k2, k3 = p[:3]
        if reduced:
            A0, B0, C0 = p[3:]
        if logc:
            if reduced == 0:
                expy = A, B, C = map(exp, y)
            elif reduced == 1:
                expy = B, C = map(exp, y)
            elif reduced == 2:
                expy = A, C = map(exp, y)
            elif reduced == 3:
                expy = A, B = map(exp, y)
        else:
            if reduced == 0:
                A, B, C = y
            elif reduced == 1:
                B, C = y
            elif reduced == 2:
                A, C = y
            elif reduced == 3:
                A, B = y

        if reduced == 1:
            A = A0 + B0 + C0 - B - C
        elif reduced == 2:
            B = A0 + B0 + C0 - A - C
        elif reduced == 3:
            C = A0 + B0 + C0 - A - B

        r1 = k1*A
        r2 = k2*B*C
        r3 = k3*B*B
        f = [r2 - r1, r1 - r2 - r3, r3]
        if reduced == 0:
            pass
        elif reduced == 1:
            f = [f[1], f[2]]
        elif reduced == 2:
            f = [f[0], f[2]]
        elif reduced == 3:
            f = [f[0], f[1]]

        if logc:
            f = [f_/ey for ey, f_ in zip(expy, f)]
        if logt:
            ex = exp(x)
            f = [ex*f_ for f_ in f]
        return f
    return dydt


def run_integration(inits=(1, 0, 0), rates=(0.04, 1e4, 3e7), t0=1e-10,
                    tend=1e19, nt=2, logc=False, logt=False, reduced=False,
                    atol=1e-8, rtol=1e-8, zero_conc=1e-23, dep_scaling=1,
                    indep_scaling=1, powsimp=False, **kwargs):

    if nt == 2:
        tout = (t0, tend)
    else:
        tout = np.logspace(np.log10(t0), np.log10(tend), nt)

    odesys = ScaledSys.from_callback(
        get_ode_exprs(logc, logt, reduced),
        2 if reduced else 3, 3, dep_scaling=dep_scaling,
        indep_scaling=indep_scaling,
        exprs_process_cb=(lambda exprs: [
            sp.powsimp(expr.expand(), force=True) for expr in exprs])
        if powsimp else None)

    print(odesys.exprs)

    indices = {0: (0, 1, 2), 1: (1, 2), 2: (0, 2), 3: (0, 1)}[reduced]
    inits = np.array([inits[idx] for idx in indices], dtype=np.float64)
    if logc:
        inits = np.log(inits + zero_conc)
    if logt:
        tout = np.log(tout)
    xout, yout, info = odesys.integrate(tout, inits, rates, atol=atol, rtol=rtol, **kwargs)
    if logc:
        yout = np.exp(yout)
    if logt:
        xout = np.exp(xout)
    if reduced:
        yout = np.insert(yout, reduced-1, 1 - np.sum(yout, axis=1), axis=1)
    return xout, yout, info


def plot(xout, yout, info):
    import matplotlib.pyplot as plt
    for idx in range(yout.shape[1]):
        plt.loglog(xout, yout[:, idx], label='ABC'[idx])
    plt.legend()
    info.pop('internal_xout')
    info.pop('internal_yout')
    print(info)
