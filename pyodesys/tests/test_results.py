# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import math

import numpy as np

from .. import ODESys
from .test_core import sine, sine_jac


def test_Result_at():
    odesys = ODESys(sine, sine_jac)
    A, k = 2, 3
    result = odesys.integrate(np.linspace(0, 1, 17), [0, A*k], [k], atol=1e-8, rtol=1e-8)
    x_probe = 2**-0.5
    assert result.info['success']
    ref = np.array([A*math.sin(k*x_probe), A*math.cos(k*x_probe)*k])
    est, est_err = result.at(x_probe)
    real_err = np.abs(est - ref)
    assert np.all(real_err < est_err)
    assert np.allclose(ref, est, atol=1e-6, rtol=1e-6)
