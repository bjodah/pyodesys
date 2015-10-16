import numpy as np

from pyodesys import SymbolicSys


def _get_equations(m_val, g_val, l_val):
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


class TimeDoublePendulum:

    def setup(self):
        self.odesys = SymbolicSys(_get_equations(1, 9.81, 1))
        self.tout = np.linspace(0, 10., 200)
        self.y0 = [.1, .2, 0, 0]

    def time_integrate_scipy(self):
        self.odesys.integrate('scipy', self.tout, self.y0)

    def time_integrate_gsl(self):
        self.odesys.integrate('gsl', self.tout, self.y0)

    def time_integrate_odeint(self):
        self.odesys.integrate('odeint', self.tout, self.y0)

    def time_integrate_cvode(self):
        self.odesys.integrate('cvode', self.tout, self.y0)
