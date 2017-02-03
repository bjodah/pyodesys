# -*- coding: utf-8 -*-
"""
This module is for demonstration purposes only and the integrators here
are not meant for production use. Consider them provisional, i.e., API here
may break without prior deprecation.
"""

import math
import warnings

import numpy as np
from .util import import_

lu_factor, lu_solve = import_('scipy.linalg', 'lu_factor', 'lu_solve')


class RK4_example_integrator:
    """
    This is an example of how to implement a custom integrator.
    It uses fixed step size and is usually not useful for real problems.
    """

    with_jacobian = False

    @staticmethod
    def integrate_adaptive(rhs, jac, y0, x0, xend, dx0, **kwargs):
        if kwargs:
            warnings.warn("Ignoring keyword-argumtents: %s" % ', '.join(kwargs.keys()))
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
        if kwargs:
            warnings.warn("Ignoring keyword-argumtents: %s" % ', '.join(kwargs.keys()))
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


class EulerForward_example_integrator:

    with_jacobian = False
    integrate_adaptive = None

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        if kwargs:
            warnings.warn("Ignoring keyword-argumtents: %s" % ', '.join(kwargs.keys()))
        x_old = xout[0]
        yout = [y0[:]]
        f = np.empty(len(y0))
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            rhs(x_old, y, f)
            yout.append(y + h*f)
            x_old = x
        return np.array(yout), {'nfev': (len(xout)-1)}


class Midpoint_example_integrator:

    with_jacobian = False
    integrate_adaptive = None

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        if kwargs:
            warnings.warn("Ignoring keyword-argumtents: %s" % ', '.join(kwargs.keys()))
        x_old = xout[0]
        yout = [y0[:]]
        f = np.empty(len(y0))
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            rhs(x_old, y, f)
            dy_efw = h*f
            rhs(x_old + h/2, y + dy_efw/2, f)
            yout.append(y + h*f)
            x_old = x
        return np.array(yout), {'nfev': (len(xout)-1)}


class EulerBackward_example_integrator:

    with_jacobian = True
    integrate_adaptive = None

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        if kwargs:
            warnings.warn("Ignoring keyword-argumtents: %s" % ', '.join(kwargs.keys()))
        x_old = xout[0]
        yout = [y0[:]]
        f = np.empty(len(y0))
        j = np.empty((len(y0), len(y0)))
        I = np.eye(len(y0))
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            jac(x_old, y, j)
            lu_piv = lu_factor(h*j - I)
            rhs(x, y, f)
            ynew = y + f*h
            norm_delta_ynew = float('inf')
            while norm_delta_ynew > 1e-12:
                rhs(x, ynew, f)
                delta_ynew = lu_solve(lu_piv, ynew - y - f*h)
                ynew += delta_ynew
                norm_delta_ynew = np.sqrt(np.sum(np.square(delta_ynew)))

            yout.append(ynew)
            x_old = x
        return np.array(yout), {'nfev': (len(xout)-1)}


class Trapezoidal_example_integrator:

    with_jacobian = True
    integrate_adaptive = None

    @staticmethod
    def integrate_predefined(rhs, jac, y0, xout, **kwargs):
        if kwargs:
            warnings.warn("Ignoring keyword-argumtents: %s" % ', '.join(kwargs.keys()))
        x_old = xout[0]
        yout = [y0[:]]
        f = np.empty(len(y0))
        j = np.empty((len(y0), len(y0)))
        I = np.eye(len(y0))
        for i, x in enumerate(xout[1:], 1):
            y = yout[-1]
            h = x - x_old
            jac(x_old, y, j)
            lu_piv = lu_factor(h*j - I)
            rhs(x, y, f)
            euler_fw_dy = f*h
            ynew = y + euler_fw_dy
            norm_delta_ynew = float('inf')
            while norm_delta_ynew > 1e-12:
                rhs(x, ynew, f)
                delta_ynew = lu_solve(lu_piv, ynew - y - f*h)
                ynew += delta_ynew
                norm_delta_ynew = np.sqrt(np.sum(np.square(delta_ynew)))

            yout.append((ynew + y + euler_fw_dy)/2)
            x_old = x
        return np.array(yout), {'nfev': (len(xout)-1)}
