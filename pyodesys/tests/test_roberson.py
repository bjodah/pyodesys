# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from ._robertson import run_integration  # , get_ode_exprs


def test_run_integration():
    xout, yout, info = run_integration(integrator='odeint')
    assert info['success'] is True
