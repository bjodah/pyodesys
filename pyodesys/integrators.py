# -*- coding: utf-8 -*-
"""
This module is for demonstration purposes only and the integrators here
are not meant for production use. Consider them provisional, i.e., API here
may break without prior deprecation.
"""

import math
import numpy as np


class RK4_example_integartor:
    """
    This is an example of how to implement a custom integrator.
    It uses fixed step size and is usually not useful for real problems.
    """

    with_jacobian = False

    @staticmethod
    def integrate_adaptive(rhs, jac, y0, x0, xend, dx0, **kwargs):
        xspan = xend - x0
        n = int(math.ceil(xspan/dx0))
        yout = [y0[:]]
        xout = [x0]
        k = [np.empty(len(y0)) for _ in range(4)]
        for i in range(0, n+1):
            x, y = xout[-1], yout[-1]
            h = min(dx0, xend-x)
            rhs(x,       y,            k[0])
            rhs(x + h/2, y + h/2*k[0], k[1])
            rhs(x + h/2, y + h/2*k[1], k[2])
            rhs(x + h,   y + h*k[2],   k[3])
            yout.append(y + h/6 * (k[0] + 2*k[1] + 2*k[2] + k[3]))
            xout.append(x+h)
        return np.array(xout), np.array(yout), {'nfev': n*4}

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        x_old = xout[0]
        yout = [y0[:]]
        k = [np.empty(len(y0)) for _ in range(4)]
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            rhs(x_old,       y,            k[0])
            rhs(x_old + h/2, y + h/2*k[0], k[1])
            rhs(x_old + h/2, y + h/2*k[1], k[2])
            rhs(x_old + h,   y + h*k[2],   k[3])
            yout.append(y + h/6 * (k[0] + 2*k[1] + 2*k[2] + k[3]))
            x_old = x
        return np.array(yout), {'nfev': (len(xout)-1)*4}
