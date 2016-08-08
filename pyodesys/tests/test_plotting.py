# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest

try:
    import matplotlib
except ImportError:
    matplotlib = None

from pyodesys import OdeSys
from .test_core import vdp_f, vdp_j


@pytest.mark.skipif(matplotlib is None, reason='package matplotlib missing')
def test_plot_result():
    odes = OdeSys(vdp_f, vdp_j)
    odes.integrate([0, 1, 2], [1, 0], params=[2.0], integrator='scipy')
    odes.plot_result()


@pytest.mark.skipif(matplotlib is None, reason='package matplotlib missing')
def test_plot_result_interpolation():
    odes = OdeSys(vdp_f, vdp_j)
    odes.integrate([0, 1, 2], [1, 0], params=[2.0], nderiv=1,
                   integrator='cvode')
    odes.plot_result(interpolate=True)
