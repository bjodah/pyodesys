#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (absolute_import, division, print_function)


"""
Example:
$ python pydy_double_pendulum.py --plot --nt 200
"""

import numpy as np

from pyodesys.symbolic import SymbolicSys
from pyodesys.util import stack_1d_on_left


def get_equations(m_val, g_val, l_val):
    # This function body is copyied from:
    # http://www.pydy.org/examples/double_pendulum.html
    # Retrieved 2015-09-29
    from sympy import symbols
    from sympy.physics.mechanics import (
        dynamicsymbols, ReferenceFrame, Point, Particle, KanesMethod
    )

    q1, q2 = dynamicsymbols('q1 q2')
    q1d, q2d = dynamicsymbols('q1 q2', 1)
    u1, u2 = dynamicsymbols('u1 u2')
    u1d, u2d = dynamicsymbols('u1 u2', 1)
    l, m, g = symbols('l m g')

    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = N.orientnew('B', 'Axis', [q2, N.z])

    A.set_ang_vel(N, u1 * N.z)
    B.set_ang_vel(N, u2 * N.z)

    O = Point('O')
    P = O.locatenew('P', l * A.x)
    R = P.locatenew('R', l * B.x)

    O.set_vel(N, 0)
    P.v2pt_theory(O, N, A)
    R.v2pt_theory(P, N, B)

    ParP = Particle('ParP', P, m)
    ParR = Particle('ParR', R, m)

    kd = [q1d - u1, q2d - u2]
    FL = [(P, m * g * N.x), (R, m * g * N.x)]
    BL = [ParP, ParR]

    KM = KanesMethod(N, q_ind=[q1, q2], u_ind=[u1, u2], kd_eqs=kd)

    try:
        (fr, frstar) = KM.kanes_equations(bodies=BL, loads=FL)
    except TypeError:
        (fr, frstar) = KM.kanes_equations(FL, BL)
    kdd = KM.kindiffdict()
    mm = KM.mass_matrix_full
    fo = KM.forcing_full
    qudots = mm.inv() * fo
    qudots = qudots.subs(kdd)
    qudots.simplify()
    # Edit:
    depv = [q1, q2, u1, u2]
    subs = list(zip([m, g, l], [m_val, g_val, l_val]))
    return zip(depv, [expr.subs(subs) for expr in qudots])


def main(m=1, g=9.81, l=1, q1=.1, q2=.2, u1=0, u2=0, tend=10., nt=200,
         savefig='None', plot=False, savetxt='None', integrator='scipy',
         dpi=100, kwargs="", verbose=False):
    assert nt > 1
    kwargs = dict(eval(kwargs) if kwargs else {})
    odesys = SymbolicSys(get_equations(m, g, l), params=())
    tout = np.linspace(0, tend, nt)
    y0 = [q1, q2, u1, u2]
    xout, yout, info = odesys.integrate(
        tout, y0, integrator=integrator, **kwargs)
    if verbose:
        print(info)
    if savetxt != 'None':
        np.savetxt(stack_1d_on_left(xout, yout), savetxt)
    if plot:
        import matplotlib.pyplot as plt
        odesys.plot_result(xout, yout)
        if savefig != 'None':
            plt.savefig(savefig, dpi=dpi)
        else:
            plt.show()

if __name__ == '__main__':
    try:
        import argh
        argh.dispatch_command(main)
    except ImportError:
        import sys
        if len(sys.argv) > 1:
            import warnings
            warnings.warn("Ignoring parameters run "
                          "'pip install --user argh' to fix.")
        main()
