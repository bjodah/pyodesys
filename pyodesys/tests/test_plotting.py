# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from pyodesys import OdeSys
from .test_core import vdp_f, vdp_j


def test_plot_result():
    odes = OdeSys(vdp_f, vdp_j)
    odes.integrate('scipy', [0, 1, 2], [1, 0], params=[2.0])
    odes.plot_result()


def test_plot_result_interpolation():
    odes = OdeSys(vdp_f, vdp_j)
    odes.integrate('cvode', [0, 1, 2], [1, 0], params=[2.0], nderiv=1)
    odes.plot_result(interpolate=True)
