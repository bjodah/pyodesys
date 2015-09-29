import numpy as np
import sympy as sp


def stack_1d_on_left(x, y):
    return np.hstack((np.asarray(x).reshape(len(x), 1),
                      np.asarray(y)))


def banded_jacobian(y, x, ml, mu):
    """
    Calculates a banded version of the jacobian

    Compatible with the format requested by
    scipy.integrate.ode
    """
    ny = len(y)
    nx = len(x)
    packed = np.zeros((mu+ml+1, nx), dtype=object)

    def set(ri, ci, val):
        packed[ri-ci+mu, ci] = val

    for ri in range(ny):
        for ci in range(max(0, ri-ml), min(nx, ri+mu+1)):
            set(ri, ci, y[ri].diff(x[ci]))
    return sp.ImmutableMatrix(packed)
